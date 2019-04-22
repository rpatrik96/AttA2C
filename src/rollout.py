import torch

class Rollout(object) :
    def __init__(self, rollout_size, num_envs, is_cuda=True) :
        """

        :param rollout_size: number of steps after the policy gets updated
        :param num_envs: number of environments to train on parallel
        :param is_cuda: flag whether to use CUDA
        """
        super().__init__()

        self.rollout_size = rollout_size
        self.num_envs = num_envs
        self.is_cuda = is_cuda

        # initialize the buffers with zeros
        self.reset_buffers()

    def reset_buffers(self) :
        """
        Creates and/or resets the buffers - each of size (rollout_size, num_envs) -
        storing: - rewards
                 - actions
                 - log probabilities
                 - values
                 - dones

         NOTE: calling this function after a `.backward()` ensures that all data
         not needed in the future (which may `requires_grad()`) gets freed, thus
         avoiding memory leak
        :return:
        """
        self.rewards = self._generate_buffer((self.rollout_size, self.num_envs))
        self.actions = self._generate_buffer((self.rollout_size, self.num_envs))
        self.log_probs = self._generate_buffer((self.rollout_size, self.num_envs))
        self.values = self._generate_buffer((self.rollout_size, self.num_envs))
        self.dones = self._generate_buffer((self.rollout_size, self.num_envs))

    def _generate_buffer(self, size) :
        """
        Generates a `torch.zeros` tensor with the specified size.

        :param size: size of the tensor (tuple)
        :return:  tensor filled with zeros of 'size'
                    on the device specified by self.is_cuda
        """
        if self.is_cuda :
            return torch.zeros(size).cuda()
        else :
            return torch.zeros(size)

    def insert(self, step, rewards, actions, log_probs, values, dones) :
        """
        Inserts new data into the log for each environment at index step

        :param step: index of the step
        :param rewards: numpy array of the rewards
        :param actions: tensor of the actions
        :param log_probs: tensor of the log probabilities
        :param values: tensor of the values
        :param dones: numpy array of the dones (boolean)
        :return:
        """
        self.rewards[step].copy_(torch.from_numpy(rewards))
        self.actions[step].copy_(actions)
        self.log_probs[step].copy_(log_probs)
        self.values[step].copy_(values)
        self.dones[step].copy_(torch.ByteTensor(dones.data))

    def compute_reward(self, final_value, discount=0.99) :
        """
        Computes the discounted reward while respecting - if the episode
        is not done - the estimate of the final reward from that state (i.e.
        the value function passed as the argument `final_value`)

        :param env_idx: index of the environment
        :param final_value: estimate of the final reward by the critic
        :param discount: discount factor
        :return:
        """

        # normalization function
        def normalize(data) :
            return (data - data.mean()) / (data.std() + 10e-9)

        """Setup"""
        # placeholder tensor to avoid dynamic allocation with insert
        r_discounted = self._generate_buffer((self.rollout_size, self.num_envs))

        """Calculate discounted rewards"""
        # setup the reward chain
        # if the rollout has brought the env to finish
        # then we proceed with 0 as final reward (there is nothing to gain in that episode)
        # but if we did not finish, then we use our estimate

        # masked_scatter_ copies from #1 where #0 is 1 -> but we need scattering, where
        # the episode is not finished, thus the (1-x)
        R = self._generate_buffer(self.num_envs).masked_scatter((1 - self.dones[-1]).byte(), final_value)

        for i in reversed(range(self.rollout_size)) :
            # the reward can only change if we are within the episode
            # i.e. while done==True, we use 0
            # NOTE: this update rule also can handle, if a new episode has started during the rollout
            # in that case an intermediate value will be 0
            # todo: add GAE
            R = self._generate_buffer(self.num_envs).masked_scatter((1 - self.dones[-1]).byte(),
                                                                    self.rewards[i] + discount * R)

            r_discounted[i] = R

        # normalize and return
        return normalize(r_discounted)
