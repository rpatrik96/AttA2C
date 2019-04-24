# stable baselines
import torch.optim as optim
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack

from src.model import A2CNet
from src.train import Runner

# constants
NUM_ENV = 16
SEED = 42
N_STACK = 4

if __name__ == '__main__' :

    """Environment"""
    # create the atari environments
    # NOTE: this wrapper automatically resets each env if the episode is done
    env = make_atari_env('PongNoFrameskip-v4', num_env=NUM_ENV, seed=SEED)
    env = VecFrameStack(env, n_stack=N_STACK)

    """Model"""
    a2c = A2CNet(n_stack=N_STACK, num_envs=NUM_ENV, num_actions=env.action_space.n)
    a2c.cuda()
    optimizer = optim.Adam(a2c.parameters())

    """Training"""
    runner = Runner(a2c, env, optimizer, NUM_ENV, is_cuda=True)
    runner.train()
