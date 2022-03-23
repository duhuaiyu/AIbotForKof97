import random
import time
from Kof97Environment import Kof97Environment
from MAMEToolkit.sf_environment import Environment
import cv2
import os
import numpy as np
temp_dir = '/home/duhuaiyu/kof_temp'
roms_path = "roms/"  # Replace this with the path to your ROMs
env = Kof97Environment("env1", roms_path)

#env.start()
i = 0
#actions = [(6,8),(6,8),(6,8),(6,8),(6,8),(0,8),(0,8),(0,8),(8,0),(8,0),(8,8),(8,8),(8,8),(8,8),(8,8),(8,8),(8,8),(8,8),(8,8),(8,8),(8,8),(8,8),(8,8),(8,8)]
actions = [(6,8),(6,8),(7,8),(7,8),(0,8),(0,0),(0,0),(8,0),(8,0),(8,8),(8,8),(8,8),(8,8),(8,8),(8,8)]
env.start()
#actions = [(0,8),(0,0)]
# while True:
#     frame, reward, round_done, stage_done, game_done = env.step(6, 8)
#     if not round_done:
#         break
while True:
    # move_action = random.randint(0, 8)
    # attack_action = random.randint(0, 7)
    # time.sleep(1)
    index = i % len(actions)
    move_action = actions[index][0]
    attack_action = actions[index][1]
    #print (move_action,attack_action)
    frame, reward, round_done, stage_done, game_done = env.step(move_action, attack_action)

    #if not round_done:
        #for frame in frames:
        #print('save file')
        #cv2.imwrite(os.path.join(temp_dir, str(i) +".png"),cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    #print("game_done",game_done)
    if game_done:
        env.reset()
    elif stage_done:
        env.reset()
    elif round_done:
        #print("round done!!")
        env.reset()
    i = i +1