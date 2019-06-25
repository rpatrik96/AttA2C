import argparse
from dataclasses import dataclass

import h5py
import numpy as np
import torch


class LogData(object):
    def __init__(self):
        self.mean = []
        self.std = []
        self.min = []
        self.max = []

    def log(self, sample):
        self.mean.append(sample.mean())
        self.std.append(sample.std())
        self.min.append(sample.min())
        self.max.append(sample.max())

    def save(self, group):
        group.create_dataset("mean", data=self.mean)
        group.create_dataset("std", data=self.std)
        group.create_dataset("min", data=self.min)
        group.create_dataset("max", data=self.max)

    def load(self, group):
        # read in parameters
        # [()] is needed to read in the whole array if you don't do that,
        #  it doesn't read the whole data but instead gives you lazy access to sub-parts
        #  (very useful when the array is huge but you only need a small part of it).
        # https://stackoverflow.com/questions/10274476/how-to-export-hdf5-file-to-numpy-using-h5py
        self.mean = group["mean"][()]
        self.std = group["std"][()]
        self.min = group["min"][()]
        self.max = group["max"][()]


class TemporalLogger(object):
    def __init__(self):
        super().__init__()
        self.rewards = LogData()
        self.features = LogData()

    def log(self, reward, feature):
        self.rewards.log(reward)
        self.features.log(feature)

    def save(self, path):
        with h5py.File(path + '.hdf5', 'w') as f:
            rewards = f.create_group("rewards")
            features = f.create_group("features")

            self.rewards.save(rewards)
            self.features.save(features)

    def load(self, path):
        with h5py.File(path + '.hdf5', 'r') as f:
            print(f['rewards'])
            self.rewards.load(f['rewards'])
            self.features.load(f['features'])


@dataclass
class NetworkParameters:
    num_envs: int
    n_stack: int
    rollout_size: int = 5
    num_updates: int = 2500000
    max_grad_norm: float = 0.5
    curiosity_coeff: float = 0.03
    icm_beta: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.02


class HyperparamScheduler(object):

    def __init__(self, init_val, end_val, tau=20000):
        super().__init__()

        self.init_val = init_val
        self.end_val = end_val
        self.param = self.init_val
        self.cntr = 0
        self.tau = tau

    def step(self):
        self.cntr += 1

        self.param = (self.init_val - self.end_val) * np.exp(-self.cntr / self.tau)


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
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='log with Tensorboard')
    parser.add_argument('--log-dir', type=str, default="../../log/curiosity_loss_scheduled",
                        help='log directory for Tensorboard')
    parser.add_argument('--seed', type=int, default=42, metavar='SEED',
                        help='random seed')
    parser.add_argument('--max-grad_norm', type=float, default=.5, metavar='MAX_GRAD_NORM',
                        help='threshold for gradient clipping')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')

    # environment
    parser.add_argument('--env-name', type=str, default='PongNoFrameskip-v4',
                        help='environment name')
    parser.add_argument('--num-envs', type=int, default=16, metavar='NUM_ENVS',
                        help='number of parallel environemnts')
    parser.add_argument('--n-stack', type=int, default=4, metavar='N_STACK',
                        help='number of frames stacked')
    parser.add_argument('--rollout-size', type=int, default=5, metavar='ROLLOUT_SIZE',
                        help='rollout size')
    parser.add_argument('--num-updates', type=int, default=2500000, metavar='NUM_UPDATES',
                        help='number of updates')

    # model coefficients
    parser.add_argument('--curiosity-coeff', type=float, default=.015, metavar='CURIOSITY_COEFF',
                        help='curiosity-based exploration coefficient')
    parser.add_argument('--icm-beta', type=float, default=.2, metavar='ICM_BETA',
                        help='beta for the ICM module')
    parser.add_argument('--value-coeff', type=float, default=.5, metavar='VALUE_COEFF',
                        help='value loss weight factor in the A2C loss')
    parser.add_argument('--entropy-coeff', type=float, default=.02, metavar='ENTROPY_COEFF',
                        help='entropy loss weight factor in the A2C loss')

    # Argument parsing
    return parser.parse_args()


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
