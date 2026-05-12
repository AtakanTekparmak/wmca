import ale_py; import gymnasium; gymnasium.register_envs(ale_py)
"""5-Atari WFM: generate data for 5 Atari games, pre-train on 4, hold out 1."""
import numpy as np, torch, torch.nn as nn, time, os, sys
import gymnasium as gym

DEVICE = torch.device('mps')
GRID = 32; EPOCHS = 100

# 5 diverse Atari games
GAMES = ['Pong', 'Breakout', 'SpaceInvaders', 'Freeway', 'Asterix']

def play_game(name, episodes=200, steps_per_ep=100):
    """Play random policy, collect (obs, next_obs, action) triples."""
    env = gym.make(f'ALE/{name}-v5', render_mode=None)
    rng = np.random.default_rng(42)
    frames, nframes, actions = [], [], []
    na = env.action_space.n
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        for _ in range(steps_per_ep):
            a = rng.integers(0, na)
            nobs, _, terminated, truncated, _ = env.step(a)
            # Convert to grayscale, resize to 64×64
            obs_g = np.mean(obs, axis=-1)  # RGB -> grayscale
            nobs_g = np.mean(nobs, axis=-1)
            frames.append(obs_g.astype(np.float32) / 255.0)
            nframes.append(nobs_g.astype(np.float32) / 255.0)
            actions.append(a)
            obs = nobs
            if terminated or truncated: break
        if ep % 50 == 0: print(f'  {name}: {ep}/{episodes}')
    env.close()
    frames = np.stack(frames)[:, np.newaxis]  # (N, 1, H, W)
    nframes = np.stack(nframes)[:, np.newaxis]
    actions = np.array(actions, dtype=np.int64)
    return frames, nframes, actions

class GridAE(nn.Module):
    def __init__(self, hid=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1,hid,3,1,1),nn.ReLU(),nn.Conv2d(hid,1,3,1,1),nn.Sigmoid())
        self.dec = nn.Sequential(nn.Conv2d(1,hid,3,1,1),nn.ReLU(),nn.Conv2d(hid,1,3,1,1),nn.Sigmoid())
    def encode(self,x): return self.enc(x)
    def forward(self,x): return self.dec(self.enc(x))

def make_af(actions, na, H, W):
    vals = (actions.astype(np.float32) + 1.0) / na
    return np.tile(vals[:, None, None, None], (1, 1, H, W))

def pad32(X):
    ph=(32-X.shape[2])//2; pw=(32-X.shape[3])//2
    return np.pad(X,((0,0),(0,0),(ph,32-X.shape[2]-ph),(pw,32-X.shape[3]-pw)))

class WFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cml = CML2DMultiR(in_channels=2,K=32,steps=15,seed=42,gate_mode='uniform')
        self.nca = nn.Sequential(nn.Conv2d(4,64,3,1,1),nn.ReLU(),nn.Conv2d(64,64,3,1,1),nn.ReLU(),nn.Conv2d(64,1,1))
    def forward(self,x):
        with torch.no_grad(): co=self.cml(x)
        return co[:,:1]+self.nca(torch.cat([x,co],dim=1))

# Step 1: Generate data for all 5 games
print("=== STEP 1: Generate Atari data ===")
data = {}
for game in GAMES:
    print(f"\n--- {game} ---")
    frames, nframes, actions = play_game(game, episodes=200, steps_per_ep=50)
    
    # Train AE
    n = len(frames); n_train = int(n * 0.8)
    X_ae = torch.from_numpy(frames[:n_train]).float().to(DEVICE)
    ae = GridAE().to(DEVICE); opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    for _ in range(10):
        perm = torch.randperm(len(X_ae), device=DEVICE)
        for i in range(0, len(perm), 8):
            ix = perm[i:i+8]; opt.zero_grad()
            nn.functional.mse_loss(ae(X_ae[ix]), X_ae[ix]).backward(); opt.step()
    ae.eval()
    
    # Encode
    with torch.no_grad():
        lats = ae.encode(torch.from_numpy(frames).float().to(DEVICE)).cpu().numpy()
        nlats = ae.encode(torch.from_numpy(nframes).float().to(DEVICE)).cpu().numpy()
    recon = ae(torch.from_numpy(frames[:64]).float().to(DEVICE)).cpu()
    psnr = 10*np.log10(1.0/nn.functional.mse_loss(recon, torch.from_numpy(frames[:64]).float()).item())
    
    na = env.action_space.n if "env" in dir() else actions.max()+1
    H, W = lats.shape[2], lats.shape[3]; na = env.action_space.n if 'env' in dir() else actions.max()+1
    af = make_af(actions, na, H, W)
    X = pad32(np.concatenate([lats, af], axis=1))
    Y = pad32(nlats)
    data[game] = {'X': X, 'Y': Y, 'na': na}
    print(f"  {len(lats)} samples, PSNR={psnr:.1f}dB, {na} actions")

# Step 2: Pre-train on 4, hold out Breakout
print(f"\n=== STEP 2: WFM pre-training on 4 games ===")
holdout = 'Breakout'
train_games = [g for g in GAMES if g != holdout]
print(f"Pre-train: {train_games}")
print(f"Hold-out: {holdout}")

from wmca.modules.hybrid import CML2DMultiR
model = WFM().to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
t0 = time.time()

for ep in range(EPOCHS):
    model.train()
    for game in train_games:
        Xt, Yt = data[game]['X'], data[game]['Y']
        n = min(len(Xt), 5000)
        xt = torch.from_numpy(Xt[:n]).float().to(DEVICE); yt = torch.from_numpy(Yt[:n]).float().to(DEVICE)
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, 8):
            ix = perm[i:i+8]; opt.zero_grad()
            nn.functional.mse_loss(model(xt[ix]), yt[ix]).backward(); opt.step()
    if ep % 20 == 0 or ep == EPOCHS-1:
        print(f"  ep {ep+1:3d}/{EPOCHS} [{((time.time()-t0)/60):.0f}m]")

os.makedirs('experiments/results', exist_ok=True)
torch.save(model.state_dict(), 'experiments/results/wfm_5env.pt')
print(f"Saved. Time: {(time.time()-t0)/60:.0f}m")

# Step 3: Zero-shot + FT on Breakout
print(f"\n=== STEP 3: Transfer to {holdout} ===")
Xb, Yb = data[holdout]['X'], data[holdout]['Y']
n_ft = int(len(Xb) * 0.7)
Xft = torch.from_numpy(Xb[:n_ft]).float().to(DEVICE); Yft = torch.from_numpy(Yb[:n_ft]).float().to(DEVICE)
Xfv = torch.from_numpy(Xb[n_ft:n_ft+256]).float().to(DEVICE); Yfv = torch.from_numpy(Yb[n_ft:n_ft+256]).float().to(DEVICE)

model.eval()
with torch.no_grad(): z = nn.functional.mse_loss(model(Xfv), Yfv).item()
print(f"Zero-shot: {z:.2e}")

mf = WFM().to(DEVICE); mf.load_state_dict(torch.load('experiments/results/wfm_5env.pt'))
ot = torch.optim.Adam(mf.parameters(), lr=1e-4)
for ep in range(30):
    mf.train(); perm = torch.randperm(n_ft, device=DEVICE)
    for i in range(0, n_ft, 8):
        ix = perm[i:i+8]; ot.zero_grad()
        nn.functional.mse_loss(mf(Xft[ix]), Yft[ix]).backward(); ot.step()
mf.eval()
with torch.no_grad(): ft = nn.functional.mse_loss(mf(Xfv), Yfv).item()
print(f"FT 30ep: {ft:.2e}")

ms = WFM().to(DEVICE); os_ = torch.optim.Adam(ms.parameters(), lr=1e-3)
for ep in range(100):
    ms.train(); perm = torch.randperm(n_ft, device=DEVICE)
    for i in range(0, n_ft, 8):
        ix = perm[i:i+8]; os_.zero_grad()
        nn.functional.mse_loss(ms(Xft[ix]), Yft[ix]).backward(); os_.step()
ms.eval()
with torch.no_grad(): s_ = nn.functional.mse_loss(ms(Xfv), Yfv).item()
print(f"Scratch 100ep: {s_:.2e}")

print(f"\n{'='*60}")
print(f"RESULTS (pre-train: {len(train_games)} games, hold-out: {holdout}):")
print(f"  Pre-train envs: {train_games}")
print(f"  Zero-shot:       {z:.2e}")
print(f"  Fine-tuned 30ep: {ft:.2e}")
print(f"  From-scratch 100ep: {s_:.2e}")
print(f"  FT vs scratch:   {s_/ft:.1f}x {'WIN!' if ft < s_ else 'LOSS'}")
