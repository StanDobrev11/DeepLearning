# import torch
# import tensorflow as tf
# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#
# print(torch.cuda.is_available())
# print(tf.__version__)
# print(tf.keras.__version__)
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="rgb_array")
observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated

env.close()