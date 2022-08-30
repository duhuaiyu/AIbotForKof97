# import numpy as np
# from sb3_contrib import RecurrentPPO
# from stable_baselines3.common.evaluation import evaluate_policy
# model = RecurrentPPO("MlpLstmPolicy", "CartPole-v1", verbose=1)
# model.learn(5000)
# env = model.get_env()
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, warn=False)
# print(mean_reward)
# model.save("ppo_recurrent")
# del model # remove to demonstrate saving and loading
# model = RecurrentPPO.load("ppo_recurrent")
# obs = env.reset()
# # cell and hidden state of the LSTM
# lstm_states = None
# num_envs = 1
# # Episode start signals are used to reset the lstm states
# episode_starts = np.ones((num_envs,), dtype=bool)
# while True:
#   action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
#   obs, rewards, dones, info = env.step(action)
#   episode_starts = dones
#   env.render()

import gym
import numpy as np
from gym import spaces, utils
from MAMEToolkit.emulator import Emulator
from MAMEToolkit.emulator import Address
from KofSteps import *
from KofActions import Actions
import cv2
import math
import random
import string
from typing import Any, Dict
import pandas as pd
import gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from Kof97EnvironmentSR import Kof98EnvironmentLSTM
from Kof97EnvironmentSR import Kof98EnvironmentV2
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
video_folder = 'logs/videos/'
video_length = 100
roms_path = "roms/"


env = DummyVecEnv([lambda: Kof98EnvironmentV2(roms_path=roms_path, CH="CH0",reset_delay=16)])

# Record the video starting at the first step
env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix="PPO-Kyo Kusanagi")
obs = env.reset()
model = PPO.load("Kof97_PPO_V2_t3_1", env=env)
# models = {
#     "Multi CNN VS Kyo Kusanagi" : {"model":'Kof97_PPO_V2_t3_1', "env":env},
#     # "Multi CNN VS Goro Daimon" : {"model":'Kof97_PPO_V2_t3_1_Transfer_CH2_2', "env":env_CH2},
#     # "Multi CNN VS Mai Shiranui" : {"model":'Kof97_PPO_V2_t3_1_transfer_CH16', "env":env_CH16},
#     # "Multi CNN VS Billy" : {"model":'Kof97_PPO_V2_t3_1_transfer_CH26', "env":env_CH26},
#     # "LSTM VS Kyo Kusanagi" : {"model":'Kof97_PPO_LSTM_V3', "env":lstm_env},
# }

while True:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render(mode="rgb_array")
    if done:
      obs = env.reset()
      break
env.close()