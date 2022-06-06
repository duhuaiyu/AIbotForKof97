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
from Kof97EnvironmentSR import Kof98EnvironmentLSTM,ComboRewardCalculatorV3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.torch_layers import CombinedExtractor
from Callback import VideoRecorderCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO
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
    env_res = Kof98EnvironmentLSTM()
    return env_res

env_str = "LSTM_env"

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
    #params ={'batch_size': 5000,  'learning_rate': 1.958232389849691e-05}
    params = {'batch_size': 5000, 'learning_rate': 1.058232389849691e-04}
    trail = 71
    model = RecurrentPPO('MlpLstmPolicy', env, verbose=0,tensorboard_log=log_dir,**params)
    checkpoint_callback = CheckpointCallback(save_freq=100_000, save_path='./logs/',
                                             name_prefix='kof_model_lstm_t3')
    model.learn(total_timesteps=30_000_000,tb_log_name="PPO_LSTM_t2", reset_num_timesteps=True, callback=checkpoint_callback)
    print("finish learn")
    env.close()
    model.save("Kof97_PPO_LSTM_V3")


