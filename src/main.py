from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack

from agent import ICMAgent
from train import Runner

# constants
NUM_ENVS = 16
SEED = 42
N_STACK = 4

if __name__ == '__main__':

    """Environment"""
    # create the atari environments
    # NOTE: this wrapper automatically resets each env if the episode is done
    env_name = 'PongNoFrameskip-v4'
    env = make_atari_env(env_name, num_env=NUM_ENVS, seed=SEED)
    env = VecFrameStack(env, n_stack=N_STACK)

    """Training"""
    agent = ICMAgent(N_STACK, NUM_ENVS, env.action_space.n)


    runner = Runner(agent, env, NUM_ENVS, N_STACK, is_cuda=True)
    runner.train()

#     import torch
#     a2c.load_state_dict(torch.load("a2c_best_loss"))
#     a2c.eval()
#
#     obs = env.reset()
#     for i in range(1000) :
#         tensor = torch.from_numpy(obs.transpose((0, 3, 1, 2))).float() / 255.
#         tensor =  tensor.cuda() if torch.cuda.is_available() else tensor
#         action, _,_ = a2c.get_action(tensor)
#         obs, rewards, dones, info = env.step(action)
#         env.render()
# #
