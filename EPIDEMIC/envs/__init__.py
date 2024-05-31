from gymnasium.envs.registration import register
register(
    id='envs/GridWorldEnv-v0',
    entry_point='envs.GridWorldEnv:GridWorldEnv',
)
print("Registered")