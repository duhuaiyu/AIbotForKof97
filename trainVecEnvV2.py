import gym
import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv,VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.preprocessing import is_image_space
from multiprocessing import Process, freeze_support, set_start_method
from Kof97EnvironmentSR import Kof98EnvironmentV2,ComboRewardCalculatorV3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.torch_layers import CombinedExtractor
from Callback import VideoRecorderCallback
from stable_baselines3.common.callbacks import CheckpointCallback
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
    env_res = Kof98EnvironmentV2()
    return env_res

env_str = "V2_env"

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
    # evl_env = Monitor(create_env())
    params ={'batch_size': 5000,  'learning_rate': 1.058232389849691e-04}
    trail = 71
    #model = PPO('MultiInputPolicy', env, verbose=0,tensorboard_log=log_dir,**params)
    #model = PPO.load(f"opt_2/trial_{trail}_best_model.zip", env=env)
    model = PPO.load(f"Kof97_PPO_V2_t3_1_transfer_CH16", env=env)
    model.learning_rate = 3.058232389849691e-04
    checkpoint_callback = CheckpointCallback(save_freq=1_000_00, save_path='./logs/',
                                             name_prefix='Kof97_PPO_V2_t3_1_transfer_CH26')
    model.learn(total_timesteps=30_000_000,tb_log_name="PPO_V2_t3_transfer_CH26", reset_num_timesteps=True, callback=checkpoint_callback)
    print("finish learn")
    env.close()
    # evl_env.close()
    print("finish close")
    model.save("Kof97_PPO_V2_t3_1_transfer_CH26")
    # obs = env.reset()
    # for _ in range(10000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()

