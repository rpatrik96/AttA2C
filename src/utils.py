import argparse

import torch


def get_args():
    """
    Function for handling command line arguments

    :return: parsed   command line arguments
    """
    parser = argparse.ArgumentParser(description='Curisoity-driven deep RL with A2C+ICM')

    # training
    parser.add_argument('--train', action='store_true', default=True,
                        help='train flag (False->load model)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='CUDA flag')
    parser.add_argument('--tensorboard', action='store_true', default=True,
                        help='log with Tensorboard')
    parser.add_argument('--log-dir', type=str, default="../../log/curiosity_loss_pampam",
                        help='log directory for Tensorboard')
    parser.add_argument('--seed', type=int, default=42, metavar='SEED',
                        help='random seed')
    parser.add_argument('--max-grad_norm', type=float, default=.5, metavar='MAX_GRAD_NORM',
                        help='threshold for gradient clipping')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')

    # environment
    parser.add_argument('--num_envs', type=int, default=16, metavar='NUM_ENVS',
                        help='number of parallel environemnts')
    parser.add_argument('--n-stack', type=int, default=4, metavar='N_STACK',
                        help='number of frames stacked')
    parser.add_argument('--rollout-size', type=int, default=5, metavar='ROLLOUT_SIZE',
                        help='rollout size')
    parser.add_argument('--num-updates', type=int, default=2500000, metavar='NUM_UPDATES',
                        help='number of updates')

    # model coefficients
    parser.add_argument('--curiosity-coeff', type=float, default=.03, metavar='CURIOSITY_COEFF',
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
