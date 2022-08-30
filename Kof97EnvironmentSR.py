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
from os import listdir
from os.path import isfile, join
from collections import deque
from stable_baselines3.common.envs import SimpleMultiObsEnv

class RewardCalculator:
    def get_reward(self, info ,last_info):
        pass

class DefaultRewardCalculator(RewardCalculator):
    def __init__(self, distance_rate, time_rate, damage_rate):
        self.distance_rate = distance_rate
        self.time_rate = time_rate
        self.damage_rate = damage_rate

    def get_reward(self,info ,last_info):
        p1_diff = (last_info["healthP1"] - info["healthP1"])
        p1_diff = 0 if p1_diff < 0 else p1_diff
        p2_diff = (last_info["healthP2"] - info["healthP2"])
        p2_diff = 0 if p2_diff < 0 else p2_diff
        damage_reward = (p2_diff-p1_diff )* self.damage_rate
        distance = np.abs(info["1P_x"] - info["2P_x"])
        #print(distance)
        if distance <= 150:
            distance = 0
        else:
            distance = distance -150

        distance_reward = - distance* self.distance_rate
        time_reward = -1 * self.time_rate
        #print(f"damage_reward:{damage_reward}, distance_reward:{distance_reward},time_reward{time_reward}")
        reward = damage_reward + distance_reward + time_reward
        #print(f"reward: {reward}")
        return reward

class ComboRewardCalculator(DefaultRewardCalculator):
    def __init__(self, distance_rate, time_rate, damage_rate):
        super(ComboRewardCalculator, self).__init__(distance_rate, time_rate, damage_rate)

    def get_reward(self,info ,last_info):
        p1_diff = (last_info["healthP1"] - info["healthP1"])
        p1_diff = 0 if p1_diff < 0 else p1_diff
        p2_diff = (last_info["healthP2"] - info["healthP2"])
        p2_diff = 0 if p2_diff < 0 else p2_diff
        combo = info["2P_combo"]


        damage_reward = (p2_diff-p1_diff )* self.damage_rate * (1 if combo == 0 else math.exp(combo-1))
        #print(combo,math.exp(combo-1),p2_diff-p1_diff ,damage_reward )
        distance = np.abs(info["1P_x"] - info["2P_x"])
        #print(distance)
        if distance <= 150:
            distance = 0
        else:
            distance = distance -150

        distance_reward = - distance* self.distance_rate

        time_reward = -1 * self.time_rate
        info["damage_reward"] = damage_reward
        info["distance_reward"] = distance_reward
        info["time_reward"] = time_reward
        #print(f"combo:{combo}, damage_reward:{damage_reward}, distance_reward:{distance_reward},time_reward{time_reward}")
        reward = damage_reward + distance_reward + time_reward
        #print(f"reward: {reward}")
        return reward

class ComboRewardCalculatorV2(DefaultRewardCalculator):
    def __init__(self, distance_rate, time_rate, damage_rate_1p,damage_rate_2p):
        super(ComboRewardCalculatorV2, self).__init__(distance_rate, time_rate, damage_rate_1p)
        self.damage_rate_1p = damage_rate_1p
        self.damage_rate_2p = damage_rate_2p

    def get_reward(self,info ,last_info):
        p1_diff = (last_info["healthP1"] - info["healthP1"])
        p2_diff = (last_info["healthP2"] - info["healthP2"])
        combo_1p = info["2P_combo"] # here exchange the 1P 2P notation for convention
        combo_2p = info["1P_combo"]
        p1_damage = p2_diff * self.damage_rate_1p * (1 if combo_1p == 0 else combo_1p)
        p2_damage = p1_diff * self.damage_rate_2p * (1 if combo_2p == 0 else combo_2p)
        damage_reward = p1_damage- p2_damage

        pow_state_1p = info["1P_pow_state"]
        _, pow_num = divmod(pow_state_1p,16)

        pow_state_1p_last = last_info["1P_pow_state"]
        _, pow_num_last = divmod(pow_state_1p_last,16)
        if pow_num < pow_num_last:
            pow_reward = -5
        else:
            pow_reward = 0
        #print(combo,math.exp(combo-1),p2_diff-p1_diff ,damage_reward )
        distance = np.abs(info["1P_x"] - info["2P_x"])
        #print(distance)
        if distance <= 150:
            distance = 0
        else:
            distance = distance -150

        distance_reward = - distance* self.distance_rate

        time_reward = -1 * self.time_rate
        info["damage_reward"] = damage_reward
        info["distance_reward"] = distance_reward
        info["time_reward"] = time_reward
        info["pow_reward"] = pow_reward
        #print(f"p1_diff:{p1_diff},p2_diff:{p2_diff}, 1p_health:{info['healthP1']},2p_health:{info['healthP2']},")
        #print(f"combo_1p:{combo_1p},combo_2p:{combo_2p} damage_reward:{damage_reward}, distance_reward:{distance_reward},time_reward{time_reward}")
        reward = damage_reward + distance_reward + time_reward+pow_reward
        if info["playing"] != 32:
            if info["healthP1"]< info["healthP2"]:
                info["win_loss"] = "LOSE"
                #print("LOSE")
                return -100
            else:
                #print("Win")
                info["win_loss"] = "WIN"
                return 100
        else:
            info["win_loss"] = "UNKONWN"
        #print(f"reward: {reward}")
        return reward

class ComboRewardCalculatorV3(DefaultRewardCalculator):
    def __init__(self, distance_rate, time_rate, damage_rate_1p,damage_rate_2p):
        super(ComboRewardCalculatorV3, self).__init__(distance_rate, time_rate, damage_rate_1p)
        self.damage_rate_1p = damage_rate_1p
        self.damage_rate_2p = damage_rate_2p

    def get_reward(self,info ,last_info):
        p1_diff = (last_info["healthP1"] - info["healthP1"])
        p2_diff = (last_info["healthP2"] - info["healthP2"])
        combo_1p = info["2P_combo"] # here exchange the 1P 2P notation for convention
        combo_2p = info["1P_combo"]
        p1_damage = p2_diff * self.damage_rate_1p * (1 if combo_1p == 0 else combo_1p)
        p2_damage = p1_diff * self.damage_rate_2p * (1 if combo_2p == 0 else combo_2p)
        damage_reward = p1_damage- p2_damage

        pow_state_1p = info["1P_pow_state"]
        _, pow_num = divmod(pow_state_1p,16)

        pow_state_1p_last = last_info["1P_pow_state"]
        _, pow_num_last = divmod(pow_state_1p_last,16)
        if pow_num < pow_num_last:
            pow_reward = -5
        else:
            pow_reward = 0
        #print(combo,math.exp(combo-1),p2_diff-p1_diff ,damage_reward )
        distance = np.abs(info["1P_x"] - info["2P_x"])
        #print(distance)
        if distance <= 150:
            distance = 0
        else:
            distance = distance -150

        distance_reward = - distance* self.distance_rate

        time_reward = -1 * self.time_rate
        info["damage_reward"] = damage_reward
        info["distance_reward"] = distance_reward
        info["time_reward"] = time_reward
        info["pow_reward"] = pow_reward
        #print(f"p1_diff:{p1_diff},p2_diff:{p2_diff}, 1p_health:{info['healthP1']},2p_health:{info['healthP2']},")
        #print(f"combo_1p:{combo_1p},combo_2p:{combo_2p} damage_reward:{damage_reward}, distance_reward:{distance_reward},time_reward{time_reward}")
        reward = damage_reward + distance_reward + time_reward+pow_reward
        if info["playing"] != 32:
            if info["healthP1"]< info["healthP2"]:
                info["win_loss"] = "LOSE"
                #print("LOSE")
                # return -100
            else:
                #print("Win")
                info["win_loss"] = "WIN"
                # return 100
        else:
            info["win_loss"] = "UNKONWN"
        #print(f"reward: {reward}")
        return reward


class Kof98Environment(gym.Env):
    MAX_VELOCITY = 30
    move_actions = {
            0: [Actions.P1_LEFT],
            1: [Actions.P1_LEFT, Actions.P1_UP],
            2: [Actions.P1_UP],
            3: [Actions.P1_UP, Actions.P1_RIGHT],
            4: [Actions.P1_RIGHT],
            5: [Actions.P1_RIGHT, Actions.P1_DOWN],
            6: [Actions.P1_DOWN],
            7: [Actions.P1_DOWN, Actions.P1_LEFT],
            8: []
        }

    attack_actions = {
        0: [Actions.P1_A],
        1: [Actions.P1_B],
        2: [Actions.P1_C],
        3: [Actions.P1_D],
        4: [Actions.P1_A, Actions.P1_B],
        5: [Actions.P1_C, Actions.P1_D],
        6: [Actions.P1_A, Actions.P1_B, Actions.P1_C],
        7: [Actions.P1_A, Actions.P1_B, Actions.P1_C, Actions.P1_D],
        8: []
    }
    def setup_memory_addresses(self):
        return {
            "playing": Address('0x10A83E', 'u8'),
            "input": Address('0x300000', 'u8'),
            "2Frame": Address('0x10DA44', 'u8'),
            "healthP1": Address('0x108239', 's8'),
            "healthP2": Address('0x108439', 's8'),
            "1P_x": Address('0x108118', 'u16'),
            "1P_y": Address('0x108120', 'u16'),
            "1P_pow_value": Address('0x1081e8', 'u8'),
            "1P_pow_state": Address('0x10825f', 'u8'), #0-3 bits, power start count.
                                                     # 4-7 bits, max state: 00, off, 10. start, 20, in max power, 30 off
            "1P_combo": Address('0x1082ce', 'u8'), # 1P be comboed count


            "2P_x": Address('0x108318', 'u16'),
            "2P_y": Address('0x108320', 'u16'),
            "2P_pow_value": Address('0x1083e8', 'u8'),
            "2P_pow_state": Address('0x10845f', 'u8'),
            "2P_combo": Address('0x1084ce', 'u8'), # 2P be comboed count
            "background1_x": Address('0x10b0ca', 'u16'),
            "1P_Action": Address('0x108172', 'u16'),
            "2P_Action": Address('0x108372', 'u16'),

            # action sequence
            "action_seq_0": Address('0x10E7B0', 'u64'),
            "action_seq_1": Address('0x10E7B8', 'u64'),
            "action_seq_2": Address('0x10E7C0', 'u64'),
            "action_seq_3": Address('0x10E7C8', 'u64'),
            "action_seq_4": Address('0x10E7D0', 'u64'),
            # match time
            "time": Address('0x10A83A', 'u8'),
        }
    def __init__(self,roms_path = "roms/", basic_saving_dir = "/home/duhuaiyu/KOF97Saves", CH = "CH26",rewardCalculator = ComboRewardCalculatorV3(distance_rate=0.01,time_rate=0.01,damage_rate_1p=1,damage_rate_2p=1.3),env_param={}):


        render = True
        env_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=15))
        action_space_dic = {}
        action_space_dic.update(self.move_actions)
        index = len(self.move_actions)
        for i in range(0,4):
            for j in range(0,len(self.move_actions)):
                #print(j)
                action_space_dic[index] = self.move_actions[j].copy()+self.attack_actions[i]
                index = index +1
        action_space_dic[index] = [Actions.P1_LEFT,Actions.P1_A, Actions.P1_B]
        index = index + 1
        action_space_dic[index] = [Actions.P1_RIGHT, Actions.P1_A, Actions.P1_B]
        index = index + 1
        action_space_dic[index] = [Actions.P1_A, Actions.P1_B, Actions.P1_C]
        index = index + 1
        action_space_dic[index] = [Actions.P1_A, Actions.P1_B, Actions.P1_C, Actions.P1_D]
        self.action_space_dic = action_space_dic
        self.action_space = spaces.Discrete(len(action_space_dic))
        observation_space_dim = 0
        observation_space_dim+= 9 # health, x, y , pow, pow star,  MAX Mode, combo
        observation_space_dim+= 512 # action space
        observation_space_dim = observation_space_dim * 2 # 1P, 2P
        observation_space_dim += 1 # towards, 0 left, 1 right
        observation_space_dim += (12*8) # input sequence, 8bits, 12 step
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(observation_space_dim, ), dtype=np.float32)
        print("env_id:"+env_id)
        self.emu = Emulator(env_id, roms_path, "kof97", self.setup_memory_addresses(),debug=True,frame_ratio=2,render=render,**env_param)
        self.rewardCalculator = rewardCalculator
        self.basic_saving_dir = basic_saving_dir
        self.CH=CH
        self.full_saving_path = join(basic_saving_dir,CH)
        self.save_files = [f for f in listdir(self.full_saving_path) if isfile(join(self.full_saving_path, f))]
        self.metadata = {"render.modes":["rgb_array"]}
        #print(self.save_files)


    def reset(self):
        path = join(self.full_saving_path,random.choice(self.save_files))
        #print(path)
        self.emu.console.writeln(f'manager:machine():load("{path}")')
        #self.emu.console.writeln('manager:machine():load("/home/duhuaiyu/kof97save_CH0")')
        #self.emu.console.writeln('manager:machine():load("/home/duhuaiyu/kof97save02")')
        return self.preprocess(self.wait_for_fight_start())
    def render(self, mode="rgb_array"):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.2
        if mode == "human":
            observation = self.observation
            image = np.zeros((130,310,3), np.uint8)
            image.fill(255)
            # print(image)
            thickness = 1
            offset = 9
            black_color = (0, 0, 0)
            color_1p = (255, 0, 0)
            color_2p = (0,255,0)
            # draw health bar
            image = cv2.rectangle(image,(5, 5) , (105,10), black_color, thickness)
            image = cv2.rectangle(image,(195, 5) , (295,10), black_color, thickness)
            image = cv2.rectangle(image,(5, 5) , (5+int(observation[0]*100),10), black_color, -1)
            image = cv2.rectangle(image,(195, 5) , (195+int(observation[0+offset]*100),10), black_color, -1)


            # draw location
            h_0 = 110
            w_0 = 5
            #1P
            x_1p = w_0+int(observation[1] * 300)
            y_1p = h_0 - int(observation[2] * 100)
            image = cv2.circle(image, (x_1p,y_1p), 3, color_1p, -1)
            image = cv2.putText(image,str(np.argmax(observation[18:18+512])) ,(x_1p,y_1p - 5),font,fontScale,black_color,1)
            #print(observation[7],observation[8]  )
            cv2.line(image,(x_1p,y_1p),(int(x_1p+observation[7]*Kof98Environment.MAX_VELOCITY) ,int(y_1p-observation[8]*Kof98Environment.MAX_VELOCITY) ),color_2p)

            x_2p = w_0+int(observation[1+offset] * 300)
            y_2p = h_0 - int(observation[2+offset] * 100)
            image = cv2.circle(image, (x_2p,y_2p), 3, color_2p, -1)
            image = cv2.putText(image,str(np.argmax(observation[531:531+512])),(x_2p,y_2p - 5),font,fontScale,black_color,1)
            cv2.line(image,(x_2p,y_2p),(int(x_2p+observation[7+offset]*Kof98Environment.MAX_VELOCITY) ,int(y_2p-observation[8+offset]*Kof98Environment.MAX_VELOCITY) ),color_2p)

            # draw pow
            y_start = 115
            image = cv2.rectangle(image,(5, y_start+5) , (105,y_start+10), black_color, thickness)
            image = cv2.rectangle(image,(195, y_start+5) , (295,y_start+10), black_color, thickness)
            image = cv2.circle(image, (110,y_start+7), 3, black_color, 1 if observation[5] == 0 else -1)
            #print(observation[4])
            if int(observation[4]) == 1:
                image = cv2.putText(image,"Max",(5,y_start),font,fontScale,black_color,1)

            image = cv2.rectangle(image,(5, y_start+5) , (5+int(observation[3]*100),y_start+10), black_color, -1)
            image = cv2.rectangle(image,(195, y_start+5) , (195+int(observation[3+offset]*100),y_start+10), black_color, -1)

            image = cv2.circle(image, (300,y_start+7), 3, black_color,1 if observation[5+offset] == 0 else -1)
            if int(observation[4+offset]) == 1:
                image = cv2.putText(image,"Max",(195,y_start),font,fontScale,black_color,1)

            # combo
            y_start = 50
            #print(observation[6],observation[6+offset])
            if int(observation[6]) == 1:
                image = cv2.putText(image,"Combo",(250,y_start),font,fontScale,black_color,1)
            if int(observation[6+offset]) == 1:
                image = cv2.putText(image,"Combo",(50,y_start),font,fontScale,black_color,1)
            #print(observation[18])
            print(f"1P act:{self.last_info['1P_Action']}, health: {self.last_info['healthP1']}, loc:({self.last_info['1P_x']},{self.last_info['1P_y']}), energy {self.last_info['1P_pow_value']}, reward:{self.last_info['reward']}")
            print(f"2P act:{self.last_info['2P_Action']}, health: {self.last_info['healthP2']}, loc:({self.last_info['2P_x']},{self.last_info['2P_y']}), energy {self.last_info['2P_pow_value']} ,toward:{self.last_info['toward']}")
            image = cv2.putText(image,"->" if observation[18] == 0 else "<-",(150,125),font,fontScale,black_color,1)
            return image
        elif mode=="rgb_array":
            return self.last_info["frame"]
        else:
            print(f"1P act:{self.last_info['1P_Action']}, health: {self.last_info['healthP1']}, loc:({self.last_info['1P_x']},{self.last_info['1P_y']}), energy {self.last_info['1P_pow_value']}, reward:{self.last_info['reward']}")
            print(f"2P act:{self.last_info['2P_Action']}, health: {self.last_info['healthP2']}, loc:({self.last_info['2P_x']},{self.last_info['2P_y']}), energy {self.last_info['2P_pow_value']} ,toward:{self.last_info['toward']}")
            print(f" damage_reward:{self.last_info['damage_reward']}, distance_reward:{self.last_info['distance_reward']},time_reward:{self.last_info['time_reward']},pow_reward:{self.last_info.get('pow_reward')}")



    def step(self, action):
        actions = self.action_space_dic[action]
        data = self.sub_step(actions)
        done = self.check_done(data)
        reward = self.rewardCalculator.get_reward(data,self.last_info)
        data["reward"] = reward
        data["action"] = action
        self.process_toward(data) # 0 left, 1 right, default 1P is facing right
        observation = self.preprocess(data)
        self.last_info = data
        return observation, reward, done, data

    def process_toward(self, data):
        last_toward = self.last_info["toward"]
        toward = last_toward
        if int(last_toward) == 0 :
            if data["1P_x"] <= data["2P_x"]:
                #print("1P<=2P")
                toward = 0
            else:
                #print("1P>2P")
                toward = 1
        else:
            if data["1P_x"] >= data["2P_x"]:
                #print("1P>=2P")
                toward = 1
            else:
                #print("1P<2P")
                toward = 0
        data["toward"] = toward
        #print(data["toward"],toward)
    def preprocess(self,data):
        observation = np.array([])
        observation = np.append(observation,[data["healthP1"]/103])
        observation = np.append(observation,[data["1P_x"]/736])
        observation = np.append(observation,[data["1P_y"]/120])
        observation = np.append(observation,[data["1P_pow_value"]/128])
        #print(data["1P_pow_value"],data["2P_pow_value"])
        pow_state_1p = data["1P_pow_state"]
        state, pow_num = divmod(pow_state_1p,16)
        #print(f"state {state},pow_num {pow_num}, {pow_state_1p}")
        observation = np.append(observation,[0 if state == 0 else 1])
        observation = np.append(observation,[0 if pow_num == 0 else 1])
        observation = np.append(observation,[0 if data["1P_combo"] == 0 else 1])
        # 1p velocity
        observation = np.append(observation,[(data["1P_x"] - self.last_info["1P_x"])/Kof98Environment.MAX_VELOCITY])
        observation = np.append(observation,[(data["1P_y"] - self.last_info["1P_y"])/Kof98Environment.MAX_VELOCITY])

        observation = np.append(observation,[data["healthP2"]/103])
        observation = np.append(observation,[data["2P_x"]/736])
        observation = np.append(observation,[data["2P_y"]/120])
        observation = np.append(observation,[data["2P_pow_value"]/128])
        pow_state_2p = data["2P_pow_state"]
        state, pow_num = divmod(pow_state_2p,16)
        observation = np.append(observation,[0 if state == 0 else 1])
        observation = np.append(observation,[0 if pow_num == 0 else 1])
        observation = np.append(observation,[0 if data["2P_combo"] == 0 else 1])
        # 1p velocity
        observation = np.append(observation,[(data["2P_x"] - self.last_info["2P_x"])/Kof98Environment.MAX_VELOCITY])
        observation = np.append(observation,[(data["2P_y"] - self.last_info["2P_y"])/Kof98Environment.MAX_VELOCITY])


        observation = np.append(observation,[data["toward"]])
        observation = np.append(observation,self.one_hot(512,data["1P_Action"]))
        observation = np.append(observation,self.one_hot(512,data["2P_Action"]))
        observation = np.append(observation,self.process_action_sequence(data))
        self.observation = observation
        return observation
    def int_to_b_array(self,value):
        return [int(x) for x in bin(value)[2:].zfill(8)]

    def process_action_sequence(self,info):
        res = []
        for i in range(4,0,-1):
            action_seq = info["action_seq_"+str(i)]
            #print(hex(action_seq))
            for _ in range(0,8):
                res.append(action_seq%64)
                action_seq = action_seq >>8
        binary_res = []
        last_action = None
        same_count = 0
        #print(res)
        keeped_count = 0
        for action in res:
            if keeped_count>=12:
                break
            if action == last_action and same_count<=2:
                same_count +=1
            else:
                binary_res.append(self.int_to_b_array(action))
                same_count = 0
                keeped_count +=1
            last_action = action
        #binary_res.reverse()
        for i in range(len(binary_res),12):
            binary_res.append([0,0,0,0,0,0,0,0])
        return list(np.array(binary_res).flat)

    def one_hot(self, num, index):
        res = np.zeros(num)
        res[index] = 1
        return res

    def wait_for_fight_start(self):
        data = self.emu.step([])
        # print("playing", data["playing"], 'healthP1',
        #       data["healthP1"], 'healthP2', data["healthP2"])
        i = 0
        while data["playing"] != 32:
            # if i % 2 == 1 :
            #     data = self.emu.step([Actions.P1_LEFT.value])
            # else:
            #     data = self.emu.step([Actions.P1_LEFT.value,Actions.P1_A.value])
            data = self.emu.step([])
            i = i +1
        data["action"] = 8
        data["reward"] = 0
        data["toward"] = 0
        data["damage_reward"] = 0
        data["distance_reward"] = 0
        data["time_reward"] = 0
        data["pow_reward"] = 0
        self.last_info = data
        return data
    def sub_step(self, actions):
        data = self.emu.step([action.value for action in actions])
        return data
    def check_done(self, data):
        if data["playing"] == 32:
            return False
        else:
            return True
    def close(self):
        self.emu.close()

# updated environment
# flip left right when character facing different direction
class Kof98EnvironmentV2(Kof98Environment):

    act_queue_length = 36
    input_queue_length = 40
    def __init__(self,roms_path = "roms/", basic_saving_dir = "/home/duhuaiyu/KOF97Saves", CH = "CH0",reset_delay = 0 ,rewardCalculator = ComboRewardCalculatorV3(distance_rate=0.01,time_rate=0.01,damage_rate_1p=1,damage_rate_2p=1.3),env_param={}):
        super(Kof98EnvironmentV2, self).__init__(roms_path,basic_saving_dir,CH,rewardCalculator,env_param)
        self.p1_act_seque = deque([ [0 for j in range (512)] for i in range(Kof98EnvironmentV2.act_queue_length)] ,maxlen=Kof98EnvironmentV2.act_queue_length)
        self.p2_act_seque = deque([ [0 for j in range (512)] for i in range(Kof98EnvironmentV2.act_queue_length)] ,maxlen=Kof98EnvironmentV2.act_queue_length)
        self.act_img_size = [Kof98EnvironmentV2.act_queue_length , 512 , 1]
        self.input_size = [Kof98EnvironmentV2.input_queue_length, 256, 1]
        self.vector_size = 2 * 9 + 1
        self.reset_delay = reset_delay
        self.observation_space = gym.spaces.Dict(
            spaces={
                "vec": gym.spaces.Box(-1, 1, (self.vector_size,), dtype=np.float64),
                "act_1p": gym.spaces.Box(0, 255, self.act_img_size, dtype=np.uint8),
                "act_2p": gym.spaces.Box(0, 255, self.act_img_size, dtype=np.uint8),
                "inputs": gym.spaces.Box(0, 255, self.input_size, dtype=np.uint8)
            }
        )

    def sub_step(self, actions):
        toward = self.last_info["toward"]
        data = self.emu.step([self.flip_action(action,toward) for action in actions])
        return data

    def flip_action(self,action, toward):
        if toward == 0:
            return action.value
        elif action == Actions.P1_LEFT:
            #print("flip left")
            return Actions.P1_RIGHT.value
        elif action == Actions.P1_RIGHT:
            #print("flip left")
            return Actions.P1_LEFT.value
        else:
            return action.value

    def preprocess(self,data):
        observation = np.array([])
        observation = np.append(observation,[data["healthP1"]/103])
        observation = np.append(observation,[data["1P_x"]/736])
        observation = np.append(observation,[data["1P_y"]/120])
        observation = np.append(observation,[data["1P_pow_value"]/128])
        #print(data["1P_pow_value"],data["2P_pow_value"])
        pow_state_1p = data["1P_pow_state"]
        state, pow_num = divmod(pow_state_1p,16)
        #print(f"state {state},pow_num {pow_num}, {pow_state_1p}")
        observation = np.append(observation,[0 if state == 0 else 1])
        observation = np.append(observation,[0 if pow_num == 0 else 1])
        observation = np.append(observation,[0 if data["1P_combo"] == 0 else 1])
        # 1p velocity
        observation = np.append(observation,[(data["1P_x"] - self.last_info["1P_x"])/Kof98Environment.MAX_VELOCITY])
        observation = np.append(observation,[(data["1P_y"] - self.last_info["1P_y"])/Kof98Environment.MAX_VELOCITY])

        observation = np.append(observation,[data["healthP2"]/103])
        observation = np.append(observation,[data["2P_x"]/736])
        observation = np.append(observation,[data["2P_y"]/120])
        observation = np.append(observation,[data["2P_pow_value"]/128])
        pow_state_2p = data["2P_pow_state"]
        state, pow_num = divmod(pow_state_2p,16)
        observation = np.append(observation,[0 if state == 0 else 1])
        observation = np.append(observation,[0 if pow_num == 0 else 1])
        observation = np.append(observation,[0 if data["2P_combo"] == 0 else 1])
        # 1p velocity
        observation = np.append(observation,[(data["2P_x"] - self.last_info["2P_x"])/Kof98Environment.MAX_VELOCITY])
        observation = np.append(observation,[(data["2P_y"] - self.last_info["2P_y"])/Kof98Environment.MAX_VELOCITY])
        observation = np.append(observation,abs(data["1P_x"]-data["2P_x"])/512)
        self.p1_act_seque.append(self.one_hot(512,data["1P_Action"])*255)
        self.p2_act_seque.append(self.one_hot(512, data["2P_Action"]) * 255)

        # observation = np.append(observation,[data["toward"]])
        # observation = np.append(observation,self.one_hot(512,data["1P_Action"]))
        # observation = np.append(observation,self.one_hot(512,data["2P_Action"]))
        # observation = np.append(observation,self.process_action_sequence(data))
        self.observation = observation
        return {
            "vec" : observation,
            "act_1p": np.expand_dims(np.array(self.p1_act_seque), axis=2),
            "act_2p": np.expand_dims(np.array(self.p2_act_seque),axis=2),
            "inputs": np.expand_dims(np.array(self.process_action_sequence(data)), axis=2),
        }

    def process_action_sequence(self,info):
        binary_res = []
        for i in range(4,-1,-1):
            action_seq = info["action_seq_"+str(i)]
            for _ in range(0,8):
                binary_res.append(self.one_hot(256, action_seq%64))
                action_seq = action_seq >>8
        return binary_res

    def reset(self):
        self.p1_act_seque = deque([[0 for j in range(512)] for i in range(Kof98EnvironmentV2.act_queue_length)],
                                  maxlen=Kof98EnvironmentV2.act_queue_length)
        self.p2_act_seque = deque([[0 for j in range(512)] for i in range(Kof98EnvironmentV2.act_queue_length)],
                                  maxlen=Kof98EnvironmentV2.act_queue_length)
        path = join(self.full_saving_path,random.choice(self.save_files))
        self.emu.console.writeln(f'manager:machine():load("{path}")')
        return self.preprocess(self.wait_for_fight_start())

class Kof98EnvironmentLSTM(Kof98Environment):

    def setup_memory_addresses(self):
        return {
            "playing": Address('0x10A83E', 'u8'),
            "input": Address('0x300000', 'u8'),
            "2Frame": Address('0x10DA44', 'u8'),
            "healthP1": Address('0x108239', 's8'),
            "healthP2": Address('0x108439', 's8'),
            "1P_x": Address('0x108118', 'u16'),
            "1P_y": Address('0x108120', 'u16'),
            "1P_pow_value": Address('0x1081e8', 'u8'),
            "1P_pow_state": Address('0x10825f', 'u8'), #0-3 bits, power start count.
                                                     # 4-7 bits, max state: 00, off, 10. start, 20, in max power, 30 off
            "1P_combo": Address('0x1082ce', 'u8'), # 1P be comboed count


            "2P_x": Address('0x108318', 'u16'),
            "2P_y": Address('0x108320', 'u16'),
            "2P_pow_value": Address('0x1083e8', 'u8'),
            "2P_pow_state": Address('0x10845f', 'u8'),
            "2P_combo": Address('0x1084ce', 'u8'), # 2P be comboed count
            "background1_x": Address('0x10b0ca', 'u16'),
            "1P_Action": Address('0x108172', 'u16'),
            "2P_Action": Address('0x108372', 'u16'),

            # match time
            "time": Address('0x10A83A', 'u8'),

            # action sequence
            # "action_seq_0": Address('0x10E7B0', 'u64'),
            # "action_seq_1": Address('0x10E7B8', 'u64'),
            # "action_seq_2": Address('0x10E7C0', 'u64'),
            # "action_seq_3": Address('0x10E7C8', 'u64'),
            # "action_seq_4": Address('0x10E7D0', 'u64'),
        }

    def __init__(self,roms_path = "roms/", basic_saving_dir = "/home/duhuaiyu/KOF97Saves", CH = "CH0",rewardCalculator = ComboRewardCalculatorV3(distance_rate=0.01,time_rate=0.01,damage_rate_1p=1,damage_rate_2p=1.3),env_param={}):
        super(Kof98EnvironmentLSTM, self).__init__(roms_path,basic_saving_dir,CH,rewardCalculator,env_param)
        observation_space_dim = 0
        observation_space_dim+= 9 # health, x, y , pow, pow star,  MAX Mode, combo
        observation_space_dim+= 512 # action space
        observation_space_dim = observation_space_dim * 2 # 1P, 2P
        observation_space_dim += 1  # 1P 2P distance
        observation_space_dim += 1 # towards, 0 left, 1 right
        observation_space_dim += self.action_space.n # LSTM last input
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(observation_space_dim,), dtype=np.float64)


    def sub_step(self, actions):
        toward = self.last_info["toward"]
        data = self.emu.step([self.flip_action(action,toward) for action in actions])
        return data

    def flip_action(self,action, toward):
        if toward == 0:
            return action.value
        elif action == Actions.P1_LEFT:
            #print("flip left")
            return Actions.P1_RIGHT.value
        elif action == Actions.P1_RIGHT:
            #print("flip left")
            return Actions.P1_LEFT.value
        else:
            return action.value

    def preprocess(self,data):
        observation = np.array([])
        observation = np.append(observation,[data["healthP1"]/103])
        observation = np.append(observation,[data["1P_x"]/736])
        observation = np.append(observation,[data["1P_y"]/120])
        observation = np.append(observation,[data["1P_pow_value"]/128])
        #print(data["1P_pow_value"],data["2P_pow_value"])
        pow_state_1p = data["1P_pow_state"]
        state, pow_num = divmod(pow_state_1p,16)
        #print(f"state {state},pow_num {pow_num}, {pow_state_1p}")
        observation = np.append(observation,[0 if state == 0 else 1])
        observation = np.append(observation,[0 if pow_num == 0 else 1])
        observation = np.append(observation,[0 if data["1P_combo"] == 0 else 1])
        # 1p velocity
        observation = np.append(observation,[(data["1P_x"] - self.last_info["1P_x"])/Kof98Environment.MAX_VELOCITY])
        observation = np.append(observation,[(data["1P_y"] - self.last_info["1P_y"])/Kof98Environment.MAX_VELOCITY])

        observation = np.append(observation,[data["healthP2"]/103])
        observation = np.append(observation,[data["2P_x"]/736])
        observation = np.append(observation,[data["2P_y"]/120])
        observation = np.append(observation,[data["2P_pow_value"]/128])
        pow_state_2p = data["2P_pow_state"]
        state, pow_num = divmod(pow_state_2p,16)
        observation = np.append(observation,[0 if state == 0 else 1])
        observation = np.append(observation,[0 if pow_num == 0 else 1])
        observation = np.append(observation,[0 if data["2P_combo"] == 0 else 1])
        # 1p velocity
        observation = np.append(observation,[(data["2P_x"] - self.last_info["2P_x"])/Kof98Environment.MAX_VELOCITY])
        observation = np.append(observation,[(data["2P_y"] - self.last_info["2P_y"])/Kof98Environment.MAX_VELOCITY])
        observation = np.append(observation, abs(data["1P_x"] - data["2P_x"]) / 512)

        observation = np.append(observation,[data["toward"]])
        observation = np.append(observation,self.one_hot(512,data["1P_Action"]))
        observation = np.append(observation,self.one_hot(512,data["2P_Action"]))
        observation = np.append(observation,self.one_hot(self.action_space.n, data["action"]))
        self.observation = observation
        return observation

