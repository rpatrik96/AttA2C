import torch

from src.rollout import Rollout

# todo: model save + reaftor into a class? + LSTM reset

def train(net, env, optimizer, num_envs, rollout_size=8, is_cuda=True) :
    num_steps = 10000

    # variables reset
    # number of columns: num_envs
    logger = Rollout(rollout_size, num_envs, is_cuda)

    """Environment reset"""
    obs = env.reset()

    net.feat_enc_net.reset_lstm(num_envs)

    for _ in range(num_steps) :

        final_value = rollout(env, is_cuda, logger, net, obs, rollout_size)

        optimizer.zero_grad()
        loss = a2c_loss(logger.compute_reward(final_value), logger.log_probs, logger.values)
        loss.backward(retain_graph=True)
        optimizer.step()

        # it stores a lot of data which let's the graph
        # grow out of memory, so it is crucial to reset
        logger.reset_buffers()

    env.close()

def rollout(env, is_cuda, logger, net, obs, rollout_size) :
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
        # env.render()

        logger.insert(step, rewards, a_t, log_p_a_t, value, dones)

    # Note:
    # get the estimate of the final reward
    # that's why we have the CRITIC --> estimate final reward
    # detach, as the final value will only be used as a
    with torch.no_grad() :
        s_t = (torch.from_numpy(obs.transpose((0, 3, 1, 2))).float() / 255.).cuda()
        _, _, final_value = net.get_action(s_t)
    return final_value

def a2c_loss(rewards, log_prob, values) :

    # calculate advantage
    # i.e. how good was the estimate of the value of the current state
    advantage = rewards - values

    # weight the deviation of the predicted value (of the state) from the
    # actual reward (=advantage) with the negative log probability of the action
    # taken (- needed as log(p) p in [0;1] < 0)
    policy_loss = (-log_prob * advantage.detach()).mean()

    # the value loss weights the squared difference between the actual
    # and predicted rewards
    value_loss = advantage.pow(2).mean()

    # return the a2c loss
    # which is the sum of the actor (policy) and critic (advantage) losses
    # due to the fact that batches can be shorter (e.g. if an env is finished already)
    # MEAN is used instead of SUM
    #todo: include entropy?
    loss = policy_loss + value_loss

    return loss
