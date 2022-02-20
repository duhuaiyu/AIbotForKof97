import random
from Kof97Environment import Kof97Environment
from MAMEToolkit.sf_environment import Environment
import cv2
import os
temp_dir = '/home/duhuaiyu/kof_temp'
roms_path = "roms/"  # Replace this with the path to your ROMs
env = Kof97Environment("env1", roms_path)
env.start()
i = 0
while True:
    move_action = random.randint(0, 8)
    attack_action = random.randint(0, 7)
    frames, reward, round_done, stage_done, game_done = env.step(move_action, attack_action)
    # for frame in frames:
    #     print('save file')
    #     cv2.imwrite(os.path.join(temp_dir, str(i) +".png"),cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if game_done:
        env.new_game()
    elif stage_done:
        env.next_stage()
    elif round_done:
        env.next_round()
    i = i +1