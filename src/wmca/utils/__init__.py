import os
import torch


def pick_device() -> torch.device:
    force_cpu = os.environ.get("FORCE_CPU", "").strip().lower() in ("1", "true", "yes", "y")
    if (not force_cpu) and torch.cuda.is_available():
        print("Using CUDA Acceleration!")
        return torch.device("cuda")
    if (not force_cpu) and torch.backends.mps.is_available():
        print("Using Apple MPS Acceleration!")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")
