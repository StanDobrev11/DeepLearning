import gymnasium as gym

from stable_baselines3 import PPO
from gymnasium.envs.registration import register, registry
# from stable_baselines3.common.monitor import Monitor

def training_loop():
    if 'MarineEnv-v0' not in registry:
        register(
            id='MarineEnv-v0',
            entry_point='environments:MarineEnv',  # String reference to the class
        )

    env = gym.make(
        id='MarineEnv-v0',
        render_mode='rgb_array',
        continuous=True,
    )

    kwargs = {

    }

    model = PPO(
        'MlpPolicy',
        env=env,
        verbose=1,
        device='cpu',
        **kwargs
    )

    model.learn(total_timesteps=int(1e5), progress_bar=True)

    return model



if __name__ == '__main__':
    pass
