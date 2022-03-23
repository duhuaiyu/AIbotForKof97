import os
#os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from CNNModel import ActorCritic
from optimizer import GlobalAdam
from process import local_train
import torch.multiprocessing as _mp
import shutil
# pip install gym_super_mario_bros
import time

def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for Super Mario Bros""")
    # parser.add_argument("--world", type=int, default=1)
    # parser.add_argument("--stage", type=int, default=1)
    # parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument("--num_local_steps", type=int, default=100)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=10000, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/kof97_default_new")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_from_previous_stage", type=bool, default=True,
                        help="Load weight from previous trained stage")
    parser.add_argument("--use_gpu", type=bool, default=True)
    args = parser.parse_args()
    return args


def train(opt):
    torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    mp = _mp.get_context("spawn")
    #env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)#游戏环境配置
    global_model = ActorCritic(5, 49)
    if opt.use_gpu:
        global_model.cuda()
    global_model.share_memory()
    if opt.load_from_previous_stage:
        #file_ = "{}/kof97_{}_{}".format(opt.saved_path, previous_world, previous_stage)
        file_ = "{}/kof97_10000".format(opt.saved_path)
        if os.path.isfile(file_):
            global_model.load_state_dict(torch.load(file_))

    optimizer = GlobalAdam(global_model.parameters(), lr=opt.lr)
    
    #local_train(0, opt, global_model, optimizer, True)
    #local_test(opt.num_processes, opt, global_model)

    processes = []
    for index in range(opt.num_processes):
        if index == 0:
            process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer, True))
        else:
            process = mp.Process(target=local_train, args=(index, opt, global_model, optimizer))
        process.start()
        time.sleep(1)
        processes.append(process)
    # process = mp.Process(target=local_test, args=(opt.num_processes, opt, global_model))
    # process.start()
    processes.append(process)
    for process in processes:
        process.join()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
