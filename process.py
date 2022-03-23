import torch
from Kof97Environment import Kof97Environment
from CNNModel import ActorCritic
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from tensorboardX import SummaryWriter
import numpy as np
import timeit


def local_train(index, opt, global_model, optimizer, save=False):
    torch.manual_seed(123 + index)
    if save:
        start_time = timeit.default_timer()
    writer = SummaryWriter(opt.log_path)
    temp_dir = '/home/duhuaiyu/kof_temp'
    roms_path = "roms/"  # Replace this with the path to your ROMs
    env = Kof97Environment("env"+str(index), roms_path, writer=writer)
    state = env.start()

    local_model = ActorCritic(5, 49)
    if opt.use_gpu:
        local_model.cuda()
    local_model.train()
    round_done = True
    #state, reward, round_done, stage_done, game_done = env.wait_for_next_round()
    #state = torch.from_numpy(state)
    # state = torch.tensor(np.array(state), dtype=torch.float)
    # state = state.permute(0, 3, 1, 2)  done = True
    state = torch.stack(tuple(state), dim=1)
    if opt.use_gpu:
        state = state.cuda()

    curr_step = 0
    curr_episode = 0
    while True:
        if save:
            if curr_episode % opt.save_interval == 0 and curr_episode > 0:
                print("Process {} save model".format(index))
                torch.save(global_model.state_dict(),
                           "{}/kof97_{}".format(opt.saved_path,curr_episode))
            print("Process {}. Episode {}".format(index, curr_episode))
        curr_episode += 1
        local_model.load_state_dict(global_model.state_dict())
        if round_done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if opt.use_gpu:
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()

        log_policies = []
        values = []
        rewards = []
        entropies = []

        for _ in range(opt.num_local_steps):
            curr_step += 1
            logits, value, h_0, c_0 = local_model(state, h_0, c_0)#return self.actor_linear(hx), self.critic_linear(hx), hx, cx#隐层和记忆单元
            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)#计算当前熵值

            m = Categorical(policy)#采样
            action = m.sample().item()

            #state, reward, done, _ = env.step(action)
            state, reward, round_done, stage_done, game_done = env.step_single_action(action)
            state = torch.stack(tuple(state), dim=1)
            #state = torch.from_numpy(state)
            # state = torch.tensor(np.array(state), dtype=torch.float)
            #state = state.permute(0, 3, 1, 2)
            if opt.use_gpu:
                state = state.cuda()
            # if curr_step > opt.num_global_steps:
            #     done = True


            if round_done:
                curr_step = 0
                state = env.reset()
                state = torch.stack(tuple(state), dim=1)
                # state = torch.tensor(np.array(state), dtype=torch.float)
                # state = state.permute(0, 3, 1, 2)
                if opt.use_gpu:
                    state = state.cuda()

            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)

            if round_done:
                break

        R = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            R = R.cuda()
        if not round_done:
            _, R, _, _ = local_model(state, h_0, c_0)#这个R相当于最后一次的V值，第二个返回值是critic网络的

        gae = torch.zeros((1, 1), dtype=torch.float)#额外的处理，为了减小variance
        if opt.use_gpu:
            gae = gae.cuda()
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R

        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() - value.detach()#Generalized Advantage Estimator 带权重的折扣项
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * opt.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - opt.beta * entropy_loss
        writer.add_scalar("Train_{}/Loss".format(index), total_loss, curr_episode)
        optimizer.zero_grad()
        total_loss.backward()

        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad

        optimizer.step()

        if curr_episode == int(opt.num_global_steps / opt.num_local_steps):
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                print('The code runs for %.2f s ' % (end_time - start_time))
            return


# def local_test(index, opt, global_model):
#     torch.manual_seed(123 + index)
#     env, num_states, num_actions = create_train_env(opt.world, opt.stage, opt.action_type)
#     local_model = ActorCritic(num_states, num_actions)
#     local_model.eval()
#     state = torch.from_numpy(env.reset())
#     done = True
#     curr_step = 0
#     actions = deque(maxlen=opt.max_actions)
#     while True:
#         curr_step += 1
#         if done:
#             local_model.load_state_dict(global_model.state_dict())
#         with torch.no_grad():
#             if done:
#                 h_0 = torch.zeros((1, 512), dtype=torch.float)
#                 c_0 = torch.zeros((1, 512), dtype=torch.float)
#             else:
#                 h_0 = h_0.detach()
#                 c_0 = c_0.detach()
#
#         logits, value, h_0, c_0 = local_model(state, h_0, c_0)
#         policy = F.softmax(logits, dim=1)
#         action = torch.argmax(policy).item()
#         state, reward, done, _ = env.step(action)
#         env.render()
#         actions.append(action)
#         if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
#             done = True
#         if done:
#             curr_step = 0
#             actions.clear()
#             state = env.reset()
#         state = torch.from_numpy(state)
