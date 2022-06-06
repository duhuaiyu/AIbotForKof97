# Importing the optimzation frame - HPO
import optuna
# PPO algo for RL
from stable_baselines3 import PPO
# Bring in the eval policy method for metric calculation
from stable_baselines3.common.evaluation import evaluate_policy
# Import the sb3 monitor for logging
from stable_baselines3.common.monitor import Monitor
# Import the vec wrappers to vectorize and frame stack
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
# Import os to deal with filepaths
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv,VecMonitor
import os
from Kof97EnvironmentSR import Kof98Environment,ComboRewardCalculatorV3
from stable_baselines3.common.utils import set_random_seed
from multiprocessing import Process, freeze_support, set_start_method

LOG_DIR = './logs_2/'
OPT_DIR = './opt_2/'

# Function to return test hyperparameters - define the object function
def optimize_ppo(trial):
    return {
        #'n_steps':trial.suggest_int('n_steps', 258, 2048),
        'batch_size': trial.suggest_int('batch_size', 10240, 102400),
        'gamma':trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate':trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),
        'clip_range':trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda':trial.suggest_uniform('gae_lambda', 0.8, 0.99)
    }
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
# Run a training loop and return mean reward

if __name__ == '__main__':
    set_start_method('forkserver', force=True)
    # Creating the experiment
    set_start_method('forkserver', force=True)
    # Create environment
    # env = StreetFighter()
    # env = Monitor(env, LOG_DIR)
    # env = DummyVecEnv([lambda: env])
    # env = VecFrameStack(env, 4, channels_order='last')
    env_id = "kof97:kof97-v1"
    num_cpu = 12  # Number of use
    env = SubprocVecEnv([make_env(env_id, i) for i
                         in range(num_cpu)])
    env = VecMonitor(env, LOG_DIR)
    study = optuna.create_study(direction='maximize')
    def optimize_agent(trial):
        try:
            model_params = optimize_ppo(trial)

            env.reset()
            print("reset")
            # Create algo
            model = PPO('MlpPolicy', env, tensorboard_log=LOG_DIR, verbose=1, **model_params)
            #model.learn(total_timesteps=30000)
            model.learn(total_timesteps=100000)

            # Evaluate model
            mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
            print("mean_reward")
            print(mean_reward)
            # env.close()

            SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(trial.number))
            model.save(SAVE_PATH)

            return mean_reward

        except Exception as e:
            return -1000
    study.optimize(optimize_agent, n_trials=100, n_jobs=1)
    print("best_params:")
    print(study.best_params)
    print("best_trial:")
    print(study.best_trial)