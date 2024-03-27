from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
    load_env_plugins as _load_env_plugins,
)

# Hook to load plugins from entry points
_load_env_plugins()



register(
    id="TrainContinuous-v0",
    entry_point="gym.envs.train:trainEnv",
    max_episode_steps=100000,
    reward_threshold=200.0,
)


