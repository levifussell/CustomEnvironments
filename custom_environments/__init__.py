from gym.envs.registration import register

register(
        id='predator-prey-v0',
        entry_point='custom_environments.envs:PredatorPreyEnv'
        )

register(
        id='predator-prey-v2',
        entry_point='custom_environments.envs:PredatorPrey2'
        )
