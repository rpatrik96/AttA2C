import math

import numpy as np
import torch
import torch.nn.functional as F

def play_episode(net, env, num_envs, is_cuda=True) :
    num_steps = 10000

    # variables reset
    episode_rewards = [[] for _ in range(num_envs)]
    actions = [[] for _ in range(num_envs)]
    a_t_log_probs = [[] for _ in range(num_envs)]
    values = [[] for _ in range(num_envs)]
    episode_dones = [[] for _ in range(num_envs)]

    """Environment reset"""
    obs = env.reset()

    net.feat_enc_net.reset_lstm(num_envs)

    for step in range(num_steps) :
        # 1. reorder dimensions for nn.Conv2d (batch, ch_in, width, height)
        # 2. convert numpy array to _normalized_ FloatTensor
        s_t = (torch.from_numpy(obs.transpose((0, 3, 1, 2))).float() / 255.).cuda()

        if is_cuda :
            s_t.cuda()

        """Interact with the environments """
        # call A2C
        a_t, log_p_a_t, value = net.get_action(s_t)
        # interact
        obs, rewards, dones, _ = env.step(a_t.cpu().numpy())

        num_finished = 0
        for i in range(num_envs) :
            episode_rewards[i].append(rewards[i])
            actions[i].append(a_t[i])
            a_t_log_probs[i].append(log_p_a_t[i])
            values[i].append(value[i])
            episode_dones[i].append(dones[i])

            if True in episode_dones[i] :
                num_finished += 1

        # break if all envs are finished
        # print(num_finished)
        # if dones.sum() == num_envs :
        if num_finished == num_envs :
            break

    return episode_rewards, actions, a_t_log_probs, values, episode_dones

def train(net, env, optimizer, num_envs, num_epoch=20, rollout_size=8, is_cuda=True) :
    num_backwards = 0
    """--------------------------------"""
    """Training"""
    """--------------------------------"""
    for epoch in range(num_epoch) :

        """--------------------------------"""
        """Epsiode"""
        """--------------------------------"""
        rewards, actions, a_t_log_probs, values, dones = play_episode(net, env, num_envs, is_cuda)

        """process data/env"""

        env_idx = 0
        for r, a_t, a_t_log_p, val, done in zip(rewards, actions, a_t_log_probs, values, dones) :

            if True in done :
                end_idx = done.index(True) + 1
                r = r[:end_idx]
                a_t = a_t[:end_idx]
                a_t_log_p = a_t_log_p[:end_idx]
                val = val[:end_idx]

            num_frames = len(r)

            num_updates = math.ceil(num_frames / rollout_size)

            num_backwards += num_updates
            with torch.no_grad() :
                print("Env ", env_idx, " reward: ", np.array(r).sum(), " #backwards: ", num_backwards)
                env_idx += 1

            # discount rewards
            r = discount_reward(r)

            for i in range(num_updates) :

                optimizer.zero_grad()
                loss = a2c_loss(r[i * rollout_size :(i + 1) * rollout_size],
                                a_t_log_p[i * rollout_size :(i + 1) * rollout_size],
                                val[i * rollout_size :(i + 1) * rollout_size])

                loss.backward(retain_graph=True)
                optimizer.step()

def discount_reward(rewards, discount=0.99) :
    # normalization function
    def normalize(data) :
        return (data - data.mean()) / (data.std() + 10e-9)

    """Setup"""
    # placeholder list to avoid dynamic list allocation with insert
    r_discounted = [0] * len(rewards)
    # reversed indices
    rev_idx = list(range(len(rewards) - 1, -1, -1))

    """Calculate discounted rewards"""
    R = 0  # cumulated reward
    for i in rev_idx :
        R = rewards[i] + discount * R
        r_discounted[i] = R

    # convert to tensor and normalize
    r_discounted = normalize(torch.tensor(r_discounted).cuda())  # tensorize

    return r_discounted

def a2c_loss(rewards, a_t_log_prob, values) :
    policy_losses = []
    value_losses = []

    # grab the log probability of the action taken, the value associated to it
    # and the reward
    for log_prob, value, R in zip(a_t_log_prob, values, rewards) :

        # calculate advantage
        # i.e. how good was the estimate of the value of the current state
        advantage = R - value.item()

        # weight the deviation of the predicted value (of the state) from the
        # actual reward (=advantage) with the negative log probability of the action
        # taken (- needed as log(p) p in [0;1] < 0)
        policy_losses.append(-log_prob * advantage)

        # the value loss weights the squared difference between the actual
        # and predicted rewards
        value_losses.append(F.mse_loss(value, torch.tensor([R]).cuda()))

    # return the a2c loss
    # which is the sum of the actor (policy) and critic (advantage) losses
    # due to the fact that batches can be shorter (e.g. if an env is finished already)
    # MEAN is used instead of SUM
    loss = torch.stack(policy_losses).mean() + torch.stack(value_losses).mean()

    return loss
