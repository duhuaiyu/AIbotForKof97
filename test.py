import random
import time

import numpy

from Kof97EnvironmentSR import Kof98EnvironmentLSTM,ComboRewardCalculatorV2
from MAMEToolkit.sf_environment import Environment
import cv2
import os
import numpy as np
from stable_baselines3.common.env_checker import check_env
temp_dir = '/home/duhuaiyu/kof_temp'
roms_path = "roms/"  # Replace this with the path to your ROMs


#env.start()
i = 0
#actions = [(6,8),(6,8),(6,8),(6,8),(6,8),(0,8),(0,8),(0,8),(8,0),(8,0),(8,8),(8,8),(8,8),(8,8),(8,8),(8,8),(8,8),(8,8),(8,8),(8,8),(8,8),(8,8),(8,8),(8,8)]
actions = [(6,8),(6,8),(7,8),(7,8),(0,8),(0,0),(0,0),(8,0),(8,0),(8,8),(8,8),(8,8),(8,8),(8,8),(8,8)]

# while True:
#     env = Kof97Environment("env1", roms_path, debug=True)
#     res = env.start()
#     print(res["2P_CH"])
#     env.close()
#     if res["2P_CH"] == 0:
#         break
#actions = [(0,8),(0,0)]
# while True:
#     frame, reward, round_done, stage_done, game_done = env.step(6, 8)
#     if not round_done:
#         break
env = Kof98EnvironmentLSTM()
print(env.observation_space)
res = env.reset()
print(res.shape)
check_env(env)
print("check over")
i = 0
res = []
while i<20:
    obs, reward, done, info = env.step(random.randint(0, 48))
    res.append(obs)
    #if not round_done:
        #for frame in frames:
        #print('save file')
        #cv2.imwrite(os.path.join(temp_dir, str(i) +".png"),cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    #print("game_done",game_done)
    if done:
        env.reset()
    i = i +1
array= np.asarray(res[0:20])
print(res)