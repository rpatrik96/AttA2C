# stable baselines
import torch.optim as optim
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack

from src.model import A2CNet
from src.train import train

# constants
NUM_ENV = 4
SEED = 42
N_STACK = 4

if __name__ == '__main__' :

    # create the atari environments
    env = make_atari_env('PongNoFrameskip-v4', num_env=NUM_ENV, seed=SEED)
    env = VecFrameStack(env, n_stack=N_STACK)

    a2c = A2CNet(env.action_space.n, n_stack=N_STACK)
    a2c.cuda()
    optimizer = optim.Adam(a2c.parameters())

    train(a2c, env, optimizer, NUM_ENV, is_cuda=True)
