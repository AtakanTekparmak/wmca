# Wandb initialization and logging utilities for WMCA experiments

import os
from pathlib import Path
from dotenv import load_dotenv

def init_wandb(run_name: str, config: dict, tags: list[str] | None = None):
    """Initialize a wandb run. Loads .env from project root."""
    import wandb

    # Load .env from project root
    env_path = Path(__file__).parent.parent.parent.parent / ".env"
    load_dotenv(env_path)

    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "wmca"),
        name=run_name,
        config=config,
        tags=tags or [],
    )
