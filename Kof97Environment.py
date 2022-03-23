from MAMEToolkit.emulator import Emulator
from MAMEToolkit.emulator import Address
from KofSteps import *
from KofActions import Actions
from collections import deque
from PIL import Image, ImageOps
import torchvision.transforms as ts
import time
import cv2


# Combines the data of multiple time steps
def add_rewards(old_data, new_data):
    for k in old_data.keys():
        if "rewards" in k:
            for player in old_data[k]:
                new_data[k][player] += old_data[k][player]
    return new_data


# Returns the list of memory addresses required to train on Street Fighter
def setup_memory_addresses():
    return {
        "playing": Address('0x10A83E', 'u8'),
        "fighting2": Address('0x10A83F', 'u8'),
        "input": Address('0x300000', 'u8'),
        "2Frame": Address('0x10DA44', 'u8'),
        # "winsP1": Address('0x02011383', 'u8'),
        # "winsP2": Address('0x02011385', 'u8'),
        # "healthP1": Address('0x108251', 'u8'),
        # "healthP2": Address('0x108451', 'u8'),
        "healthP1": Address('0x108239', 'u8'),
        "healthP2": Address('0x108439', 'u8'),
        "1P_x": Address('0x108118', 'u16'),
        "1P_y": Address('0x108120', 'u16'),
        "2P_x": Address('0x108318', 'u16'),
        "2P_y": Address('0x108320', 'u16'),
        "1P_Action": Address('0x108172', 'u16'),
        "2P_Action": Address('0x108372', 'u16'),
        # "action_queue_1": Address('0x10E79C', 'u8'),
        # "action_queue_2": Address('0x10E79D', 'u8'),
        # "action_queue_3": Address('0x10E79E', 'u8'),
        # "action_queue_4": Address('0x10E79F', 'u8'),
        # "action_queue_5": Address('0x10E7A0', 'u8'),
        # "action_queue_6": Address('0x10E7A1', 'u8'),
        # "action_queue_7": Address('0x10E7A2', 'u8'),
        # "action_queue_8": Address('0x10E7A3', 'u8'),
        #
        # "action_queue_9": Address('0x10E7A4', 'u8'),
        # "action_queue_10": Address('0x10E7A5', 'u8'),
        # "action_queue_11": Address('0x10E7A6', 'u8'),
        # "action_queue_12": Address('0x10E7A7', 'u8'),
        # "action_queue_13": Address('0x10E7A8', 'u8'),
        # "action_queue_14": Address('0x10E7A9', 'u8'),
        # "action_queue_15": Address('0x10E7AA', 'u8'),
        # "action_queue_16": Address('0x10E7AB', 'u8'),
        #
        # "action_queue_17": Address('0x10E7AC', 'u8'),
        # "action_queue_18": Address('0x10E7AD', 'u8'),
        # "action_queue_19": Address('0x10E7AE', 'u8'),
        # "action_queue_20": Address('0x10E7AF', 'u8'),
        # "action_queue_21": Address('0x10E7B0', 'u8'),
        # "action_queue_22": Address('0x10E7B1', 'u8'),
        # "action_queue_23": Address('0x10E7B2', 'u8'),
        # "action_queue_24": Address('0x10E7B3', 'u8'),
        #
        # "action_queue_25": Address('0x10E7B4', 'u8'),
        # "action_queue_26": Address('0x10E7B5', 'u8'),
        # "action_queue_27": Address('0x10E7B6', 'u8'),
        # "action_queue_28": Address('0x10E7B7', 'u8'),
        # "action_queue_29": Address('0x10E7B8', 'u8'),
        # "action_queue_30": Address('0x10E7B9', 'u8'),
        # "action_queue_31": Address('0x10E7BA', 'u8'),
        # "action_queue_32": Address('0x10E7BB', 'u8'),
        #
        # "action_queue_33": Address('0x10E7BC', 'u8'),
        # "action_queue_34": Address('0x10E7BD', 'u8'),
        # "action_queue_35": Address('0x10E7BE', 'u8'),
        # "action_queue_36": Address('0x10E7BF', 'u8'),
        # "action_queue_37": Address('0x10E7C0', 'u8'),
        # "action_queue_38": Address('0x10E7C1', 'u8'),
        # "action_queue_39": Address('0x10E7C2', 'u8'),
        # "action_queue_40": Address('0x10E7C3', 'u8'),
        #
        # "action_queue_41": Address('0x10E7C4', 'u8'),
        # "action_queue_42": Address('0x10E7C5', 'u8'),
        # "action_queue_43": Address('0x10E7C6', 'u8'),
        # "action_queue_44": Address('0x10E7C7', 'u8'),
        # "action_queue_45": Address('0x10E7C8', 'u8'),
        # "action_queue_46": Address('0x10E7C9', 'u8'),
        # "action_queue_47": Address('0x10E7CA', 'u8'),
        # "action_queue_48": Address('0x10E7CB', 'u8'),
        #
        # "action_queue_49": Address('0x10E7CC', 'u8'),
        # "action_queue_50": Address('0x10E7CD', 'u8'),
        # "action_queue_51": Address('0x10E7CE', 'u8'),
        # "action_queue_52": Address('0x10E7CF', 'u8'),
        #
        # "action_queue_53": Address('0x10E7D0', 'u8'),
        # "action_queue_54": Address('0x10E7D1', 'u8'),
        # "action_queue_55": Address('0x10E7D2', 'u8'),
        # "action_queue_56": Address('0x10E7D3', 'u8'),
        #
        # "action_queue_57": Address('0x10E7D4', 'u8'),
        # "action_queue_58": Address('0x10E7D5', 'u8'),
        # "action_queue_59": Address('0x10E7D6', 'u8'),
        # "action_queue_60": Address('0x10E7D7', 'u8'),
        #
        # "direction1": Address('0x10E7D8', 'u8'),
        # "direction2": Address('0x10E7D9', 'u8'),
        #
        # "lastTime": Address('0x10E7DA', 'u16'),
    }



# Converts and index (action) into the relevant movement action Enum, depending on the player
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

def index_to_move_action(action):
    return move_actions[action]


# Converts and index (action) into the relevant attack action Enum, depending on the player
def index_to_attack_action(action):
    return attack_actions[action]


action_space = {}
action_space.update(move_actions)
index = len(move_actions)
for i in range(0,4):
    for j in range(0,len(move_actions)):
        #print(j)
        action_space[index] = index_to_move_action(j).copy()+index_to_attack_action(i)
        index = index +1
action_space[index] = [Actions.P1_LEFT,Actions.P1_A, Actions.P1_B]
index = index + 1
action_space[index] = [Actions.P1_RIGHT, Actions.P1_A, Actions.P1_B]
index = index + 1
action_space[index] = [Actions.P1_A, Actions.P1_B, Actions.P1_C]
index = index + 1
action_space[index] = [Actions.P1_A, Actions.P1_B, Actions.P1_C, Actions.P1_D]

def index_to_action(action):
    return action_space[action]


# The Street Fighter specific interface for training an agent against the game
class Kof97Environment(object):

    # env_id - the unique identifier of the emulator environment, used to create fifo pipes
    # difficulty - the difficult to be used in story mode gameplay
    # frame_ratio, frames_per_step - see Emulator class
    # render, throttle, debug - see Console class
    def __init__(self, env_id, roms_path, difficulty=3, frame_ratio=1, frames_per_step=1,  render=True, throttle=False, frame_skip=0, sound=False, debug=False, binary_path=None, frames_in_state = 5, preprocess = None, writer = None):
        self.difficulty = difficulty
        self.frame_ratio = frame_ratio
        self.frames_per_step = frames_per_step
        self.throttle = throttle
        self.env_id = env_id
        self.emu = Emulator(env_id, roms_path, "kof97", setup_memory_addresses(), frame_ratio=frame_ratio, render=render, throttle=throttle, frame_skip=frame_skip, sound=sound, debug=debug, binary_path=binary_path)
        self.started = False
        self.expected_health = {"P1": 0, "P2": 0}
        self.expected_wins = {"P1": 0, "P2": 0}
        self.round_done = False
        self.stage_done = False
        self.game_done = True
        self.stage = 1
        self.round = 0
        self.cumulateReward = 0
        self.roundReward = 0
        self.curr_episode = 0
        self.frame_queue = deque(maxlen=frames_in_state)
        self.writer = writer
        if preprocess is None:
            self.preprocess = ts.Compose([
                ts.Resize([128,128]),
                ts.Grayscale(),
                ts.ToTensor(),
                ts.Normalize(
                    mean=[0.5],
                    std=[0.5]
                )
            ]
            )
        else:
            self.preprocess = preprocess

    # Runs a set of action steps over a series of time steps
    # Used for transitioning the emulator through non-learnable gameplay, aka. title screens, character selects
    def run_steps(self, steps):
        for step in steps:
            # time.sleep(1)
            for i in range(step["wait"]):
                self.emu.step([])
            self.emu.step([action.value for action in step["actions"]])

    # Must be called first after creating this class
    # Sends actions to the game until the learnable gameplay starts
    # Returns the first few frames of gameplay
    def start(self):
        if self.throttle:
            for i in range(int(250/self.frame_ratio)):
                self.emu.step([])
        #self.run_steps(set_difficulty(self.frame_ratio, self.difficulty))
        self.run_steps(start_game(self.frame_ratio))
        frames = self.wait_for_waiting_scream()
        frames = self.wait_for_fight_start()
        self.started = True
        self.round = 0
        self.cumulateReward = 0
        self.curr_episode = self.curr_episode + 1
        #return self.frame_queue
        return self.frame_queue


    def wait_for_waiting_scream(self):
        i = 0
        data = self.emu.step([])
        while data["playing"] == 0:
            if i % 2 == 1 :
                data = self.emu.step([Actions.P1_LEFT.value])
            else:
                data = self.emu.step([Actions.P1_LEFT.value,Actions.P1_A.value])

    # Observes the game and waits for the fight to start
    def wait_for_fight_start(self):
        data = self.emu.step([])
        image = self.preprocess(Image.fromarray(data["frame"], 'RGB'))
        self.frame_queue.append(image)
        # print("playing", data["playing"], 'healthP1',
        #       data["healthP1"], 'healthP2', data["healthP2"])
        i = 0
        while data["playing"] != 32:
            if i % 2 == 1 :
                data = self.emu.step([Actions.P1_LEFT.value])
            else:
                data = self.emu.step([Actions.P1_LEFT.value,Actions.P1_A.value])
            # game stop
            # print("playing", data["playing"], 'healthP1',
            #       data["healthP1"], 'healthP2', data["healthP2"])
            image = self.preprocess(Image.fromarray(data["frame"], 'RGB'))
            self.frame_queue.append(image)
            if data["playing"] == 0 :
                print("Game restart")
                print("all rewards:" , self.cumulateReward, "Total round:", self.round)
                if self.writer is not None:
                    self.writer.add_scalar("Train_{}/CumulateReward".format(self.env_id), self.cumulateReward, self.curr_episode)
                    self.writer.add_scalar("Train_{}/TotalRound".format(self.env_id), self.round, self.curr_episode)
                    self.writer.add_scalar("Train_{}/AverageReward".format(self.env_id), self.cumulateReward/self.round, self.curr_episode)
                return self.start()
            i = i +1

        self.expected_health = {"P1": data["healthP1"], "P2": data["healthP2"]}
        data = self.gather_frames([])
        #return data["frame"]
        return self.frame_queue

    def reset(self):
        if self.game_done:
            return self.new_game()
        elif self.stage_done:
            return self.next_stage()
        elif self.round_done:
            return self.next_round()
        else:
            raise EnvironmentError("Reset called while gameplay still running")

    def loadRest(self,file):
        self.emu.console.writeln("manager:machine():load("+file+")")

    # To be called when a round finishes
    # Performs the necessary steps to take the agent to the next round of gameplay
    def next_round(self):
        print("Next round")
        self.round = self.round + 1
        self.round_done = False
        self.expected_health = {"P1": 0, "P2": 0}
        return self.wait_for_fight_start()

    def wait_for_next_round(self):
        while True:
            frame, reward, round_done, stage_done, game_done = self.step(8, 0)
            if not round_done:
                return frame, reward, round_done, stage_done, game_done
            # if game_done:
            #     self.new_game()
    # To be called when a game finishes
    # Performs the necessary steps to take the agent(s) to the next game and resets the necessary book keeping variables
    def next_stage(self):
        self.wait_for_continue()
        self.run_steps(next_stage(self.frame_ratio))
        self.expected_health = {"P1": 0, "P2": 0}
        self.expected_wins = {"P1": 0, "P2": 0}
        self.round_done = False
        self.stage_done = False
        return self.wait_for_fight_start()

    def new_game(self):
        self.wait_for_continue()
        self.run_steps(start_game(self.frame_ratio))
        self.expected_health = {"P1": 0, "P2": 0}
        self.expected_wins = {"P1": 0, "P2": 0}
        self.round_done = False
        self.stage_done = False
        self.game_done = False
        self.stage = 1
        self.cumulateReward = 0
        return self.wait_for_fight_start()

    # Steps the emulator along until the screen goes black at the very end of a game
    def wait_for_continue(self):
        data = self.emu.step([])
        if self.frames_per_step == 1:
            while data["frame"].sum() != 0:
                data = self.emu.step([])
        else:
            while data["frame"][0].sum() != 0:
                data = self.emu.step([])

    # Steps the emulator along until the round is definitely over
    def run_till_victor(self, data):
        while self.expected_wins["P1"] == data["winsP1"] and self.expected_wins["P2"] == data["winsP2"]:
            data = add_rewards(data, self.sub_step([]))
        self.expected_wins = {"P1":data["winsP1"], "P2":data["winsP2"]}
        return data

    # Checks whether the round or game has finished
    def check_done(self, data):
        #print('input', data["input"], '2Frame', data["2Frame"],"fighting", data["fighting"], 'healthP1', data["healthP1"],'healthP2', data["healthP2"])
        # print("playing", data["playing"], 'healthP1',
        #       data["healthP1"], 'healthP2', data["healthP2"])
        # "1P_x": Address('0x108118', 'u8'),
        # "1P_y": Address('0x108120', 'u8'),
        # "2P_x": Address('0x108318', 'u8'),
        # "2P_y": Address('0x108320', 'u8'),
        # "1P_Action": Address('0x108172', 'u8'),
        # "2P_Action": Address('0x108372', 'u8'),
        print("1P_x:",data["1P_x"],
              "1P_y:",data["1P_y"],
              "2P_x:",data["2P_x"],
              "2P_y:",data["2P_y"],
              "1P_Action:",data["1P_Action"],
              "2P_Action:",data["2P_Action"]
              )
        self.cumulateReward = self.cumulateReward + data["rewards"]
        self.roundReward = self.roundReward + data["rewards"]
        self.check_action_queue(data)
        if data["playing"] == 0:
            self.game_done = True
        else:
            self.game_done = False
        if data["playing"] != 32:
            # data = self.run_till_victor(data)
            if data["healthP1"] < data["healthP2"]:
                print("P1 Lose")
            else:
                print("P1 Win")
            self.round_done = True
            print("Round Reward" , self.roundReward)
            self.roundReward = 0

        else:
            self.round_done = False
            # if data["winsP1"] == 2:
            #     self.stage_done = True
            #     self.stage += 1
            # if data["winsP2"] == 2:
            #     self.game_done = True
        return data
    def check_action_queue(self,data):
        pass
        # res = ""
        # for i in range(1,61):
        #     res = res + str(data["action_queue_"+str(i)]) + ","
        #     if i %10 ==0:
        #         res = res + " "
        #print("AQ:" + res)
    # Collects the specified amount of frames the agent requires before choosing an action
    def gather_frames(self, actions):
        data = self.sub_step(actions)
        image = self.preprocess(Image.fromarray(data["frame"], 'RGB'))
        self.frame_queue.append(image)
        frames = [data["frame"]]
        for i in range(self.frames_per_step - 1):
            data = add_rewards(data, self.sub_step(actions))
            frames.append(data["frame"])
            image = self.preprocess(Image.fromarray(data["frame"], 'RGB'))
            self.frame_queue.append(image)
        #data["frame"] = frames[0] if self.frames_per_step == 1 else frames
        data["frame"] = frames
        return data

    # Steps the emulator along by one time step and feeds in any actions that require pressing
    # Takes the data returned from the step and updates book keeping variables
    def sub_step(self, actions):
        data = self.emu.step([action.value for action in actions])
        data["healthP1"] = data["healthP1"] - 255 if data["healthP1"] > 103 else data["healthP1"]
        data["healthP2"] = data["healthP2"] - 255 if data["healthP2"] > 103 else data["healthP2"]
        p1_diff = (self.expected_health["P1"] - data["healthP1"])
        p2_diff = (self.expected_health["P2"] - data["healthP2"])
        # p1_diff = 0
        # p2_diff = 0
        self.expected_health = {"P1": data["healthP1"], "P2": data["healthP2"]}
        #self.expected_health = 0

        rewards = {
            "P1": (p2_diff-p1_diff),
            "P2": (p1_diff-p2_diff)
        }

        data["rewards"] = rewards["P1"]
        # self.frame_queue.append(data["frame"] )
        #if data["rewards"] != 0:
        #    print("rewards",rewards,"healthP1",data["healthP1"],"healthP2",data["healthP2"],"playing",data["playing"])
        return data

    # Steps the emulator along by the requested amount of frames required for the agent to provide actions
    def step(self, move_action, attack_action):
        if self.started:
            #if not self.round_done and not self.stage_done and not self.game_done:
            actions = []
            actions += index_to_move_action(move_action)
            actions += index_to_attack_action(attack_action)
            data = self.gather_frames(actions)
            data = self.check_done(data)
            #return data["frame"], data["rewards"], self.round_done, self.stage_done, self.game_done
            return self.frame_queue, data["rewards"], self.round_done, self.stage_done, self.game_done
            #else:
            #    raise EnvironmentError("Attempted to step while characters are not fighting")
        else:
            raise EnvironmentError("Start must be called before stepping")
    def step_single_action(self, action):
        if self.started:
            #if not self.round_done and not self.stage_done and not self.game_done:
            actions = index_to_action(action)
            data = self.gather_frames(actions)
            data = self.check_done(data)
            #return data["frame"], data["rewards"], self.round_done, self.stage_done, self.game_done
            return self.frame_queue, data["rewards"], self.round_done, self.stage_done, self.game_done
            # else:
            #     raise EnvironmentError("Attempted to step while characters are not fighting")
        else:
            raise EnvironmentError("Start must be called before stepping")
    # Safely closes emulator
    def close(self):
        self.emu.close()
