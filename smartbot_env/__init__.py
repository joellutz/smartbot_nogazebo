from gym.envs.registration import register

register(
    id='SmartBotEnv-v0',
    entry_point='smartbot_env.smartbot_env:SmartBotEnv',
)