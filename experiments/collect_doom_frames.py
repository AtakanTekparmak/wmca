"""
Collect DOOM frames using ViZDoom for world model training.

Storage-efficient: stores sequential frames as uint8 with episode boundary
markers. At load time, consecutive frames within the same episode become
(frame_t, action_t, frame_{t+1}) tuples for world model training.

Usage:
    uv run --with vizdoom,numpy python experiments/collect_doom_frames.py
    uv run --with vizdoom,numpy python experiments/collect_doom_frames.py -n 10000 --pad-256

Output (in experiments/doom_data/):
    frames.npy         - (N+1, 3, H, W) uint8  [sequential frames]
    actions.npy        - (N,) int16             [action at each step]
    episode_starts.npy - (E,) int64             [frame indices where new episodes begin]
    metadata.json      - collection parameters and button mapping

To build training pairs: for each i where episode_starts doesn't mark i+1
as a new episode, the tuple is (frames[i], actions[i], frames[i+1]).
Normalize frames to [0,1] float32 at train time via frames / 255.0.

Storage: ~23 GB for 100K frames (uint8), vs ~184 GB with float32 pairs.

GameNGen comparison notes:
    - GameNGen uses 320x240 padded to 320x256 (16 zero rows on top)
    - GameNGen conditions on last 64 actions (action history window)
    - GameNGen uses the commercial DOOM WAD; we use freedoom2 (free equivalent)
    - Action space: 9 binary buttons -> 512 possible combinations
    - GameNGen trains at 20 fps (every other tic); we collect every tic (35 fps)
"""

import argparse
import json
import os
import time

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────

# Buttons used for gameplay (matches GameNGen's action subset)
BUTTON_NAMES = [
    "ATTACK",
    "USE",
    "SPEED",
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
]
NUM_BUTTONS = len(BUTTON_NAMES)  # 9 -> 512 possible action combos

WIDTH = 320
HEIGHT = 240
EPISODE_TIMEOUT = 2100  # ~60 seconds at 35 tics/second


def action_to_int(action_vec: list[int]) -> int:
    """Encode binary action vector as a single integer (bitfield)."""
    return sum(a << i for i, a in enumerate(action_vec))


def make_game(pad_to_256: bool = False) -> "vizdoom.DoomGame":
    """Create and configure a ViZDoom game instance (headless)."""
    import vizdoom as vzd

    game = vzd.DoomGame()
    game.set_doom_game_path(os.path.join(os.path.dirname(vzd.__file__), "freedoom2.wad"))
    game.set_window_visible(False)
    game.set_sound_enabled(False)

    res = vzd.ScreenResolution.RES_320X256 if pad_to_256 else vzd.ScreenResolution.RES_320X240
    game.set_screen_resolution(res)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_episode_timeout(EPISODE_TIMEOUT)

    button_map = {
        "ATTACK": vzd.Button.ATTACK,
        "USE": vzd.Button.USE,
        "SPEED": vzd.Button.SPEED,
        "MOVE_FORWARD": vzd.Button.MOVE_FORWARD,
        "MOVE_BACKWARD": vzd.Button.MOVE_BACKWARD,
        "MOVE_LEFT": vzd.Button.MOVE_LEFT,
        "MOVE_RIGHT": vzd.Button.MOVE_RIGHT,
        "TURN_LEFT": vzd.Button.TURN_LEFT,
        "TURN_RIGHT": vzd.Button.TURN_RIGHT,
    }
    for name in BUTTON_NAMES:
        game.add_available_button(button_map[name])

    game.init()
    return game


def collect_frames(
    n_transitions: int,
    output_dir: str,
    chunk_size: int = 10_000,
    seed: int = 42,
    pad_to_256: bool = False,
) -> None:
    """
    Collect n_transitions steps. Stores N+1 frames (to form N pairs),
    N actions, and episode boundary indices.

    Uses chunked saving to keep memory bounded: at most chunk_size frames
    in RAM at a time, flushed to compressed .npz chunks on disk, then
    merged into final .npy files at the end.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)

    game = make_game(pad_to_256=pad_to_256)
    h_out = 256 if pad_to_256 else HEIGHT
    frame_shape = (3, h_out, WIDTH)

    # We need n_transitions + 1 frames total (the +1 is the final next_frame).
    # But we save in chunks, so we track frame_count and action_count separately.
    # frame_count = action_count + 1 at the end.

    chunk_frames = np.zeros((chunk_size, *frame_shape), dtype=np.uint8)
    chunk_actions = np.zeros(chunk_size, dtype=np.int16)

    frame_count = 0  # total frames stored
    action_count = 0  # total actions stored
    episodes = 0
    chunk_idx = 0
    chunk_frame_files = []
    chunk_action_files = []
    episode_starts = [0]  # first episode always starts at frame 0
    frames_in_chunk = 0
    actions_in_chunk = 0

    def flush_chunk(n_frames_to_flush: int, n_actions_to_flush: int):
        nonlocal chunk_idx, frames_in_chunk, actions_in_chunk
        if n_frames_to_flush == 0:
            return
        cf = os.path.join(output_dir, f"chunk_frames_{chunk_idx:04d}.npy")
        ca = os.path.join(output_dir, f"chunk_actions_{chunk_idx:04d}.npy")
        np.save(cf, chunk_frames[:n_frames_to_flush])
        np.save(ca, chunk_actions[:n_actions_to_flush])
        chunk_frame_files.append(cf)
        chunk_action_files.append(ca)
        chunk_idx += 1
        frames_in_chunk = 0
        actions_in_chunk = 0

    def store_frame(buf: np.ndarray):
        nonlocal frame_count, frames_in_chunk
        # buf is (H, W, 3) uint8 -> (3, H, W) uint8
        chunk_frames[frames_in_chunk] = buf.transpose(2, 0, 1)
        frames_in_chunk += 1
        frame_count += 1
        if frames_in_chunk >= chunk_size:
            flush_chunk(frames_in_chunk, actions_in_chunk)

    def store_action(action_int: int):
        nonlocal action_count, actions_in_chunk
        chunk_actions[actions_in_chunk] = action_int
        actions_in_chunk += 1
        action_count += 1

    t_start = time.time()
    last_report = t_start

    # Start first episode, store initial frame
    game.new_episode()
    episodes += 1
    state = game.get_state()
    store_frame(state.screen_buffer)

    while action_count < n_transitions:
        if game.is_episode_finished():
            game.new_episode()
            episodes += 1
            state = game.get_state()
            if state is None:
                continue
            episode_starts.append(frame_count)
            store_frame(state.screen_buffer)
            continue

        # Random action
        action = rng.randint(0, 2, NUM_BUTTONS).tolist()
        store_action(action_to_int(action))
        game.make_action(action)

        if game.is_episode_finished():
            # No next frame available; we'll start a new episode next iteration.
            # The last action has no corresponding next_frame within this episode.
            # We store a "terminal" marker by not storing a frame here.
            # Adjust: remove the orphan action (no next_frame for it).
            actions_in_chunk -= 1
            action_count -= 1
            continue

        state = game.get_state()
        if state is None:
            actions_in_chunk -= 1
            action_count -= 1
            continue

        store_frame(state.screen_buffer)

        # Progress
        now = time.time()
        if now - last_report >= 5.0:
            elapsed = now - t_start
            fps = action_count / elapsed if elapsed > 0 else 0
            eta = (n_transitions - action_count) / fps if fps > 0 else 0
            mem_mb = (frame_count * np.prod(frame_shape)) / 1e6
            print(
                f"  [{action_count:>8d}/{n_transitions}] "
                f"{fps:.0f} fps | "
                f"ep: {episodes} | "
                f"frames: {frame_count} | "
                f"chunks: {chunk_idx} | "
                f"ETA: {eta:.0f}s"
            )
            last_report = now

    game.close()

    # Flush remaining
    flush_chunk(frames_in_chunk, actions_in_chunk)

    # Merge chunks
    print(f"Merging {chunk_idx} chunks...")
    all_frames = [np.load(f) for f in chunk_frame_files]
    all_actions = [np.load(f) for f in chunk_action_files]

    frames = np.concatenate(all_frames, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    ep_starts = np.array(episode_starts, dtype=np.int64)

    np.save(os.path.join(output_dir, "frames.npy"), frames)
    np.save(os.path.join(output_dir, "actions.npy"), actions)
    np.save(os.path.join(output_dir, "episode_starts.npy"), ep_starts)

    # Clean up chunks
    for f in chunk_frame_files + chunk_action_files:
        os.remove(f)

    elapsed = time.time() - t_start
    frames_mb = os.path.getsize(os.path.join(output_dir, "frames.npy")) / 1e6
    actions_mb = os.path.getsize(os.path.join(output_dir, "actions.npy")) / 1e6

    metadata = {
        "n_transitions": int(actions.shape[0]),
        "n_frames": int(frames.shape[0]),
        "n_episodes": int(ep_starts.shape[0]),
        "resolution": [WIDTH, h_out],
        "pad_to_256": pad_to_256,
        "storage_format": "uint8, CHW, [0, 255] -- normalize to float32 [0,1] at train time",
        "action_encoding": "bitfield of 9 binary buttons, stored as int16",
        "num_buttons": NUM_BUTTONS,
        "buttons": BUTTON_NAMES,
        "num_possible_actions": 2**NUM_BUTTONS,
        "wad": "freedoom2.wad (bundled with vizdoom)",
        "episode_timeout_tics": EPISODE_TIMEOUT,
        "seed": seed,
        "collection_time_seconds": round(elapsed, 1),
        "avg_fps": round(action_count / elapsed, 1),
        "file_sizes_mb": {
            "frames.npy": round(frames_mb, 1),
            "actions.npy": round(actions_mb, 1),
        },
        "total_size_mb": round(frames_mb + actions_mb, 1),
        "how_to_use": (
            "Training pairs: (frames[i], actions[i], frames[i+1]) for all i "
            "where i+1 is NOT in episode_starts. Normalize: frames.astype(float32) / 255.0"
        ),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone. {action_count} transitions across {episodes} episodes in {elapsed:.1f}s ({action_count/elapsed:.0f} fps)")
    print(f"  frames.npy:         {frames.shape} uint8 ({frames_mb:.1f} MB)")
    print(f"  actions.npy:        {actions.shape} int16 ({actions_mb:.1f} MB)")
    print(f"  episode_starts.npy: {ep_starts.shape}")
    print(f"  Total: {frames_mb + actions_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Collect DOOM frames via ViZDoom for world model training"
    )
    parser.add_argument(
        "-n", "--n-frames", type=int, default=100_000,
        help="Number of transitions (frame->action->next_frame) to collect",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, default="experiments/doom_data",
        help="Output directory",
    )
    parser.add_argument("--chunk-size", type=int, default=10_000, help="Frames per chunk (memory control)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--pad-256", action="store_true", help="Use 320x256 (GameNGen native resolution)")
    args = parser.parse_args()

    h = 256 if args.pad_256 else HEIGHT
    per_frame_bytes = 3 * h * WIDTH
    total_gb = (args.n_frames + 1) * per_frame_bytes / 1e9

    print(f"Collecting {args.n_frames:,} DOOM transitions")
    print(f"  Resolution: {WIDTH}x{h}")
    print(f"  Buttons: {NUM_BUTTONS} ({2**NUM_BUTTONS} possible actions)")
    print(f"  Output: {args.output_dir}")
    print(f"  Estimated size: {total_gb:.1f} GB (uint8)")
    print(f"  Seed: {args.seed}")
    print()

    collect_frames(
        n_transitions=args.n_frames,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        seed=args.seed,
        pad_to_256=args.pad_256,
    )


if __name__ == "__main__":
    main()
