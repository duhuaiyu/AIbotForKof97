import gym
import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv,VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from multiprocessing import Process, freeze_support, set_start_method
from Kof97EnvironmentSR import Kof98Environment,ComboRewardCalculatorV3
from stable_baselines3.common.evaluation import evaluate_policy
from Callback import VideoRecorderCallback
def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = create_env()
        # env = gym.make(env_id)
        env.seed(seed + rank)
        # env = Monitor(env, filename=sub_dir)
        return env
    set_random_seed(seed)
    return _init

def create_env():
    env_res = Kof98Environment(rewardCalculator=ComboRewardCalculatorV3(distance_rate=0.01, time_rate=0.01, damage_rate_1p=1, damage_rate_2p=1.3))
    return env_res

env_str = "product"

if __name__ == '__main__':
    # env = gym.make("kof97:kof97-v0")
    # env.reset()
    set_start_method('forkserver', force=True)
    env_id = "kof97:kof97-v1"
    num_cpu = 12 # Number of use
    # Create the vectorized environment
    env = VecMonitor(SubprocVecEnv([make_env(env_id, i) for i
                         in range(num_cpu)]))
    log_dir = f"ts-log/{env_str}"
    os.makedirs(log_dir, exist_ok=True)
    # env = VecMonitor(env,filename=log_dir)
    # Stable Baselines provides you with make_vec_env() helper
    # which does exactly the previous steps for you.
    # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
    evl_env = Monitor(create_env())
    # params = {'n_steps': 1040, 'gamma': 0.8265071569055529, 'learning_rate': 2.7479590167434578e-05,
    #           'clip_range': 0.3333600055564347, 'gae_lambda': 0.8935368177780704}

    # Trail 47 2021-04-24
    # params ={'batch_size': 16919, 'gamma': 0.8697867805123972, 'learning_rate': 1.8681700054486652e-05,
    #  'clip_range': 0.1487316140897702, 'gae_lambda': 0.9494857182533598}
    # trail = 47

    # params ={'batch_size': 19042, 'gamma': 0.8607458351483496, 'learning_rate': 2.2514439904812116e-05,
    #  'clip_range': 0.11385091895656232, 'gae_lambda': 0.8911754858643272}
    # trail = 65

    params ={'batch_size': 28815, 'gamma': 0.9254533886157719, 'learning_rate': 1.958232389849691e-05,
     'clip_range': 0.3991136793393521, 'gae_lambda': 0.9893567426332022}
    trail = 71
    model = PPO('MlpPolicy', env, verbose=1,tensorboard_log=log_dir,**params)
    model = PPO.load(f"opt_2/trial_{trail}_best_model.zip", env=env)
    video_recorder = VideoRecorderCallback(evl_env, render_freq=200_000,n_eval_episodes=10)
    model.learn(total_timesteps=50_000_000,tb_log_name="PPO_71_Train", reset_num_timesteps=True, callback=video_recorder)
    print("finish learn")
    env.close()
    evl_env.close()
    print("finish close")
    model.save("Kof97_PPO_P1")
    # obs = env.reset()
    # for _ in range(10000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()

