import argparse
from enum import Enum
from os import makedirs
from os.path import isdir, isfile, join

import numpy as np
import pandas as pd
import torch


class AttentionType(Enum):
    SINGLE_ATTENTION = 0
    DOUBLE_ATTENTION = 1


class AttentionTarget(Enum):
    NONE = 0
    ICM = 1
    A2C = 2


class RewardType(Enum):
    INTRINSIC_AND_EXTRINSIC = 0
    INTRINSIC_ONLY = 1  # currently not used


class HyperparamScheduler(object):

    def __init__(self, init_val, end_val=0.0, tau=20000, threshold=1e-5):
        super().__init__()

        self.init_val = init_val
        self.end_val = end_val
        self.value = self.init_val
        self.cntr = 0
        self.tau = tau
        self.threshold = threshold

    def step(self):
        self.cntr += 1

        if self.value > self.threshold:
            self.value = self.end_val + (self.init_val - self.end_val) * np.exp(-self.cntr / self.tau)
        else:
            self.value = 0.0

    def save(self, group):
        """

        :param group: the reference to the group level hierarchy of a .hdf5 file to save the data
        :return:
        """
        for key, val in self.__dict__.items():
            group.create_dataset(key, data=val)


class NetworkParameters(object):
    def __init__(self, env_name: str, num_envs: int, n_stack: int, rollout_size: int = 5, num_updates: int = 2500000,
                 max_grad_norm: float = 0.5, curiosity_coeff: HyperparamScheduler = HyperparamScheduler(0.0, 0.0),
                 icm_beta: float = 0.2, value_coeff: float = 0.5, entropy_coeff: float = 0.02,
                 attention_target: AttentionTarget = AttentionTarget.NONE,
                 attention_type: AttentionType = AttentionType.SINGLE_ATTENTION,
                 reward_type: RewardType = RewardType.INTRINSIC_ONLY):
        self.env_name = env_name
        self.num_envs = num_envs
        self.n_stack = n_stack
        self.rollout_size = rollout_size
        self.num_updates = num_updates
        self.max_grad_norm = max_grad_norm
        self.curiosity_coeff = curiosity_coeff
        self.icm_beta = icm_beta
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.attention_target = attention_target
        self.attention_type = attention_type
        self.reward_type = reward_type

    def save(self, data_dir, timestamp):
        param_dict = {**self.__dict__, **self.curiosity_coeff.__dict__, "timestamp": timestamp}
        del param_dict["curiosity_coeff"]

        df_path = join(data_dir, "params.tsv")

        pd.DataFrame.from_records([param_dict]).to_csv(
            df_path,
            sep='\t',
            index=False,
            header=True if not isfile(df_path) else False,
            mode='a')


def get_args():
    """
    Function for handling command line arguments

    :return: parsed   command line arguments
    """
    parser = argparse.ArgumentParser(description='Curiosity-driven deep RL with A2C+ICM')

    # training
    parser.add_argument('--train', action='store_true', default=True,
                        help='train flag (False->load model)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='CUDA flag')
    parser.add_argument('--log-dir', type=str, default="../log",
                        help='log directory for Tensorboard')
    parser.add_argument('--seed', type=int, default=42, metavar='SEED',
                        help='random seed')
    parser.add_argument('--max-grad_norm', type=float, default=.5, metavar='MAX_GRAD_NORM',
                        help='threshold for gradient clipping')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')

    # environment
    parser.add_argument('--idx', type=int, default=4, metavar='IDX',
                        help='index of the configuration to start from')
    parser.add_argument('--env-name', type=str, default='PongNoFrameskip-v4',
                        help='environment name')
    parser.add_argument('--num-envs', type=int, default=4, metavar='NUM_ENVS',
                        help='number of parallel environments')
    parser.add_argument('--n-stack', type=int, default=4, metavar='N_STACK',
                        help='number of frames stacked')
    parser.add_argument('--rollout-size', type=int, default=5, metavar='ROLLOUT_SIZE',
                        help='rollout size')
    parser.add_argument('--num-updates', type=int, default=2500000, metavar='NUM_UPDATES',
                        help='number of updates')

    # model coefficients
    parser.add_argument('--curiosity-coeff', type=float, default=.02, metavar='CURIOSITY_COEFF',
                        help='curiosity-based exploration coefficient')
    parser.add_argument('--icm-beta', type=float, default=.2, metavar='ICM_BETA',
                        help='beta for the ICM module')
    parser.add_argument('--value-coeff', type=float, default=.5, metavar='VALUE_COEFF',
                        help='value loss weight factor in the A2C loss')
    parser.add_argument('--entropy-coeff', type=float, default=.02, metavar='ENTROPY_COEFF',
                        help='entropy loss weight factor in the A2C loss')

    # Argument parsing
    return parser.parse_args()


def make_dir(dirname):
    if not isdir(dirname):
        makedirs(dirname)


def load_and_eval(agent, env):
    agent.load_state_dict(torch.load("a2c_best_loss"))
    agent.eval()

    obs = env.reset()
    for i in range(1000):
        tensor = torch.from_numpy(obs.transpose((0, 3, 1, 2))).float() / 255.
        tensor = tensor.cuda() if torch.cuda.is_available() else tensor
        action, _, _ = agent.get_action(tensor)
        obs, rewards, dones, info = env.step(action)
        env.render()
