import torch

from storage import RolloutStorage

# todo: model save

class Runner(object) :

    def __init__(self, net, env, optimizer, num_envs, rollout_size=8, num_steps=500000, is_cuda=True) :
        super().__init__()

        # constants
        self.num_envs = num_envs
        self.rollout_size = rollout_size
        self.num_steps = num_steps

        self.is_cuda = torch.cuda.is_available() and is_cuda

        # objects
        self.net = net
        self.env = env
        self.optimizer = optimizer
        self.storage = RolloutStorage(self.rollout_size, self.num_envs, (84, 84), 4, self.is_cuda)

        if self.is_cuda :
            self.net = self.net.cuda()

    def train(self) :

        """Environment reset"""
        obs = self.env.reset()
        self.storage.states[0].copy_(self.storage.obs2tensor(obs))

        for _ in range(self.num_steps) :

            final_value = self.episode_rollout()

            self.optimizer.zero_grad()
            loss = a2c_loss(self.storage.compute_reward(final_value), self.storage.log_probs, self.storage.values)
            loss.backward(retain_graph=True)
            self.optimizer.step()

            # it stores a lot of data which let's the graph
            # grow out of memory, so it is crucial to reset
            self.storage.after_update()

        self.env.close()

    def episode_rollout(self) :
        for step in range(self.rollout_size) :

            """Interact with the environments """
            # call A2C
            a_t, log_p_a_t, value = self.net.get_action(self.storage.get_state(step))
            # interact
            obs, rewards, dones, _ = self.env.step(a_t.cpu().numpy())
            # self.env.render()

            self.storage.insert(step, rewards, obs, a_t, log_p_a_t, value, dones)
            self.net.reset_recurrent_buffers(reset_indices=dones)

        # Note:
        # get the estimate of the final reward
        # that's why we have the CRITIC --> estimate final reward
        # detach, as the final value will only be used as a
        with torch.no_grad() :
            _, _, final_value = self.net.get_action(self.storage.get_state(step + 1))
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
    # todo: include entropy?
    loss = policy_loss + value_loss

    return loss
