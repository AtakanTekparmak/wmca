"""AutumnBench environment suite.

To add a new environment:
1. Create a subclass of AutumnEnv in this package
2. Implement CELL_TYPES, _init_grid(), _step_dynamics()
3. Register it in AUTUMN_ENVS dict below
4. It automatically gets a benchmark generator via make_autumn_benchmark()
"""
from __future__ import annotations

from wmca.envs.autumn.base import AutumnEnv  # noqa: F401
from wmca.envs.autumn.disease_spreading import DiseaseSpreadingEnv  # noqa: F401
from wmca.envs.autumn.gravity import GravityEnv  # noqa: F401
from wmca.envs.autumn.water_flow import WaterFlowEnv  # noqa: F401

AUTUMN_ENVS: dict[str, type[AutumnEnv]] = {
    "autumn_disease": DiseaseSpreadingEnv,
    "autumn_gravity": GravityEnv,
    "autumn_water": WaterFlowEnv,
}

# Default data generation parameters per environment.
# Keys here are split into two groups:
#   - data_gen params: episode_length, action_policy (consumed by data_gen)
#   - env params: everything else (forwarded to env constructor)
# n_trajectories is NOT here -- it's an explicit parameter of _generate().
_AUTUMN_DEFAULTS: dict[str, dict] = {
    "autumn_disease": {
        "episode_length": 30,
        "action_policy": "noop",  # pure dynamics prediction by default
        "p_infect": 0.3,
        "p_recover": 0.1,
        "p_lose_immunity": 0.0,
        "initial_infected": 3,
        "wall_density": 0.1,
    },
    "autumn_gravity": {
        "episode_length": 20,
        "action_policy": "noop",
        "n_floating_blocks": 8,
        "n_obstacle_walls": 4,
        "use_agent": False,
    },
    "autumn_water": {
        "episode_length": 30,
        "action_policy": "noop",
        "n_sources": 2,
        "wall_density": 0.15,
        "container": True,
    },
}


def make_autumn_benchmark(env_name: str):
    """Factory: creates a benchmarks.py-compatible generator for a registered env.

    Returns a callable with signature:
        (grid_size, n_trajectories, seed, device, **kwargs) -> BenchmarkData
    """
    if env_name not in AUTUMN_ENVS:
        raise ValueError(
            f"Unknown autumn env '{env_name}'. "
            f"Available: {list(AUTUMN_ENVS.keys())}"
        )

    env_class = AUTUMN_ENVS[env_name]
    defaults = _AUTUMN_DEFAULTS.get(env_name, {})

    def _generate(
        grid_size: int = env_class.GRID_SIZE,
        n_trajectories: int = 500,
        seed: int = 42,
        device: str = "cpu",
        **kwargs,
    ):
        from wmca.envs.autumn.data_gen import generate_autumn_transitions

        # Merge defaults with user overrides
        gen_kwargs = dict(defaults)
        gen_kwargs.update(kwargs)

        # Extract data_gen-level params
        episode_length = gen_kwargs.pop("episode_length", 30)
        action_policy = gen_kwargs.pop("action_policy", "noop")

        # Remaining kwargs go to env constructor
        result = generate_autumn_transitions(
            env_class=env_class,
            n_trajectories=n_trajectories,
            episode_length=episode_length,
            action_policy=action_policy,
            seed=seed,
            grid_size=grid_size,
            device=device,
            **gen_kwargs,
        )

        # Override the meta name with the registry name
        result.meta["name"] = env_name

        return result

    _generate.__doc__ = f"Generate benchmark data for {env_name}."
    return _generate


__all__ = [
    "AutumnEnv",
    "DiseaseSpreadingEnv",
    "GravityEnv",
    "WaterFlowEnv",
    "AUTUMN_ENVS",
    "make_autumn_benchmark",
]
