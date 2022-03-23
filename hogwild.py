from multiprocessing import Process
import random
from Kof97Environment import Kof97Environment


def run_env(worker_id, roms_path):
    env = Kof97Environment(f"env{worker_id}", roms_path)
    env.start()
    while True:
        move_action = random.randint(0, 7)
        attack_action = random.randint(0, 7)
        frames, reward, round_done, stage_done, game_done = env.step(move_action, attack_action)
        if game_done:
            env.new_game()
        elif stage_done:
            env.next_stage()
        elif round_done:
            env.next_round()


workers = 2
# Environments must be created outside of the threads
roms_path = "roms/"  # Replace this with the path to your ROMs
threads = [Process(target=run_env, args=(i, roms_path)) for i in range(workers)]
[thread.start() for thread in threads]