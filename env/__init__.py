from gymnasium.envs.registration import register

register(
    id='GridWorld-d-v0',
    entry_point='gridworld_env.gridworld_env_d:GridWorldEnvD',
)
