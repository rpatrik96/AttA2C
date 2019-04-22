import torch

from src.rollout import Rollout

def train(net, env, optimizer, num_envs, num_epoch=20, rollout_size=8, is_cuda=True) :
    num_backwards = 0
    """--------------------------------"""
    """Training"""
    """--------------------------------"""
    num_steps = 10000

    # variables reset
    # number of columns: num_envs
    logger = Rollout(rollout_size, num_envs, is_cuda)

    for epoch in range(num_epoch) :

        """Environment reset"""
        obs = env.reset()

        net.feat_enc_net.reset_lstm(num_envs)

        for _ in range(num_steps) :

            for step in range(rollout_size) :
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

                logger.insert(step, rewards, a_t, log_p_a_t, value, dones)

            # Note:
            # get the estimate of the final reward
            # that's why we have the CRITIC
            # 1. estimate final reward
            # 2. zero out for finished envs
            s_t = (torch.from_numpy(obs.transpose((0, 3, 1, 2))).float() / 255.).cuda()
            _, _, final_value = net.get_action(s_t)

            # print("rollout finished")

            r_disc = logger.compute_reward(final_value)
            # update for each env
            num_finished = 0
            for i in range(num_envs) :
                if logger.dones[:, i][-1] :
                    num_finished += 1

                # discount rewards
                # r = discount_reward(rollout_rewards[i], rollout_dones[i], final_value[i])
                optimizer.zero_grad()
                print(final_value)
                loss = a2c_loss(r_disc[:, i], logger.log_probs[:,i], logger.values[:,i])

                loss.backward(retain_graph=True)
                optimizer.step()

                # print("Env: ", i,  "loss: ", loss.item())

            # break if all envs are finished
            if num_finished == num_envs :
                print("episode finished ", epoch)
                break

def a2c_loss(rewards, log_prob, values) :
    # grab the log probability of the action taken, the value associated to it
    # and the reward
    # for log_prob, value, R in zip(a_t_log_prob, values, rewards) :

    # calculate advantage
    # i.e. how good was the estimate of the value of the current state
    advantage = rewards - values

    # weight the deviation of the predicted value (of the state) from the
    # actual reward (=advantage) with the negative log probability of the action
    # taken (- needed as log(p) p in [0;1] < 0)
    policy_loss = -log_prob * advantage.detach()

    # the value loss weights the squared difference between the actual
    # and predicted rewards
    value_loss = advantage.pow(2).cuda()

    # return the a2c loss
    # which is the sum of the actor (policy) and critic (advantage) losses
    # due to the fact that batches can be shorter (e.g. if an env is finished already)
    # MEAN is used instead of SUM
    loss = policy_loss.mean() + value_loss.mean()

    return loss
