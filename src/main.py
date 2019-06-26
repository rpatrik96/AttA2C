from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack

from agent import ICMAgent
from train import Runner
from utils import get_args, load_and_eval, NetworkParameters

# constants


if __name__ == '__main__':

    """Argument parsing"""
    args = get_args()

    """Environment"""
    # create the atari environments
    # NOTE: this wrapper automatically resets each env if the episode is done
    env = make_atari_env(args.env_name, num_env=args.num_envs, seed=args.seed)
    env = VecFrameStack(env, n_stack=args.n_stack)

    """Agent"""
    agent = ICMAgent(args.n_stack, args.num_envs, env.action_space.n, lr=args.lr)

    if args.train:
        """Train"""
        param = NetworkParameters(args.env_name, args.num_envs, args.n_stack, args.rollout_size, args.num_updates,
                                  args.max_grad_norm, args.curiosity_coeff, args.icm_beta, args.value_coeff,
                                  args.entropy_coeff)
        runner = Runner(agent, env, args.tensorboard, args.log_dir, args.cuda, args.seed)
        runner.train()

    else:
        """Eval"""
        load_and_eval(agent, env)
