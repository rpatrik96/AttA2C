import torch

class Rollout(object) :
    def __init__(self, rollout_size, num_envs, is_cuda=True) :
        super().__init__()

        self.rollout_size = rollout_size
        self.num_envs = num_envs
        self.is_cuda = is_cuda

        """Buffers"""
        """storing:
                    - rewards
                    - actions
                    - log probabilities
                    - values
                    - dones
                    for each environment (length is rollout_size)
        """

        self.rewards = self._generate_buffer((self.rollout_size, self.num_envs))
        self.actions = self._generate_buffer((self.rollout_size, self.num_envs))
        self.log_probs = self._generate_buffer((self.rollout_size, self.num_envs))
        self.values = self._generate_buffer((self.rollout_size, self.num_envs))
        self.dones = self._generate_buffer((self.rollout_size, self.num_envs))

    def _generate_buffer(self, size) :
        """

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

        :param env_idx: index of the environment
        :param final_value: estimate of the final reward by the critic
        :param discount: discount factor
        :return:
        """

        # normalization function
        def normalize(data) :
            return (data - data.mean()) / (data.std() + 10e-9)

        """Setup"""
        # placeholder list to avoid dynamic list allocation with insert
        r_discounted = self._generate_buffer((self.rollout_size, self.num_envs))
        # reversed indices
        rev_idx = list(range(self.rollout_size - 1, -1, -1))

        """Calculate discounted rewards"""
        # setup the reward chain
        # if the rollout has brought the env to finish
        # then we proceed with 0 as final reward (there is nothing to gain)
        # but if we did not finish, then we use our estimate
        # masked_scatter_ copies from #1 where #0 is 1 -> but we need scattering, where
        # the episode is not finished
        from pdb import set_trace
        R = self._generate_buffer(self.num_envs).masked_scatter((1 - self.dones[-1]).byte(), final_value)

        for i in rev_idx :
            # the reward can only change if we are within the episode
            # i.e. while done==True, we use 0
            # set_trace()
            R = self._generate_buffer(self.num_envs).masked_scatter((1 - self.dones[-1]).byte(),
                                                                    self.rewards[i] + discount * R)

            r_discounted[i] = R

        # normalize and return
        return normalize(r_discounted)
