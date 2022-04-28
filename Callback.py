from typing import Any, Dict

import gym
import torch as th
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from stable_baselines3.common.logger import TensorBoardOutputFormat
from datetime import datetime
import matplotlib.pyplot as plt
from stable_baselines3.common.logger import Figure
import os

class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 10, deterministic: bool = True, save_prefix = "Kof97", save_path = "./models/test"):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic
        os.makedirs(save_path, exist_ok=True)
        self.save_prefix =save_prefix
        self.save_path = save_path


    def _on_training_start(self):
        output_formats = self.logger.output_formats
        # Save reference to tensorboard formatter object
        # note: the failure case (not formatter found) is not handled here, should be done with try/except.
        self.tb_formatter = next(
            formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))
    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            print(f"evaluate:{self.n_calls}")
            screens = { i : [] for i in range(0,self._n_eval_episodes) }
            win_episodes = []
            combos={i: 0 for i in range(2,13)}
            self.model.save(os.path.join(self.save_path,f"{self.save_prefix}_{self.n_calls}_{datetime.now()}"))
            def eval_callback(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                # statistic combos
                combo_num = _locals["info"]["2P_combo"]
                if combo_num == 0 and _globals.get("combo_num") is not None and _globals.get("combo_num") > 0:
                    combos[_globals.get("combo_num")] += 1
                _globals["combo_num"] = combo_num
                current_episode = _locals["episode_counts"][0]

                # statistic win_loss
                if _locals["dones"][0]:
                    if _locals["info"].get("win_loss") == "WIN":
                        win_episodes.append(current_episode)

                # grab_screens
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens[current_episode].append(screen.transpose(2, 0, 1))

            mean_reward, std_reward = evaluate_policy(
                self.model,
                self._eval_env,
                callback=eval_callback,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            print(mean_reward,std_reward)
            print(win_episodes)
            print(combos)
            self.tb_formatter.writer.add_scalar("eval/mean_reward", mean_reward, self.n_calls)
            self.tb_formatter.writer.add_scalar("eval/std_reward", std_reward, self.n_calls)
            self.tb_formatter.writer.add_scalar("eval/win_rate", len(win_episodes)/self._n_eval_episodes, self.n_calls)
            figure = plt.figure()
            #plt.bar(myDictionary.keys(), myDictionary.values(), width, color='g')
            names = list(combos.keys())
            values = list(combos.values())
            # plt.bar(range(len(data)), values, tick_label=names)
            figure.add_subplot().bar(names, values, tick_label=names, )
            plt.title('Combos', fontsize=18)
            # Close the figure after logging it
            self.logger.record(f"combo/figure{self.n_calls}", Figure(figure, close=True),
                               exclude=("stdout", "log", "json", "csv"))
            plt.close()
            self.tb_formatter.writer.flush()
            for key, value in screens.items():
                self.logger.record(
                    f"records_{self.n_calls}/video_{key}",
                    Video(th.ByteTensor(np.array([value])), fps=60),
                    exclude=("stdout", "log", "json", "csv"),
                )
        return True