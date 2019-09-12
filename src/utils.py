from enum import Enum
from os import makedirs, listdir
from os.path import isdir, isfile, join, dirname, abspath

import matplotlib.pyplot as plt
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


def print_plot_details():
    plt.xlabel("Rollout")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


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
                 max_grad_norm: float = 0.5, curiosity_coeff: float = 0.0,
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
        param_dict = {**self.__dict__, "timestamp": timestamp}

        df_path = join(data_dir, "params.tsv")

        pd.DataFrame.from_records([param_dict]).to_csv(
            df_path,
            sep='\t',
            index=False,
            header=True if not isfile(df_path) else False,
            mode='a')


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


def merge_tables():
    # iterate over tables
    log_dir = join(dirname(dirname(abspath(__file__))), "log")

    for env_dir in listdir(log_dir):
        stocks = []
        data_dir = join(log_dir, env_dir)
        for table in listdir(data_dir):
            if table.endswith(".tsv"):
                stock_df = pd.read_csv(join(data_dir, table), sep="\t")
                stocks.append(stock_df)
        pd.concat(stocks, axis=0, sort=True).to_csv(join(data_dir, "params.tsv"), sep="\t", index=False)


if __name__ == '__main__':
    merge_tables()
