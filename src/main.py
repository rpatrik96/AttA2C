from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack

from agent import ICMAgent
from args import get_args
from logger import Renderer
from train import Runner
# constants
from utils import AttentionTarget, AttentionType, RewardType
from utils import NetworkParameters, set_random_seeds

if __name__ == '__main__':

    """Argument parsing"""
    args = get_args()

    set_random_seeds(args.seed)

    env_names = ["PongNoFrameskip-v4",  # "PongNoFrameskip-v4",
                 "BreakoutNoFrameskip-v4",  # "BreakoutNoFrameskip-v4",
                 "SeaquestNoFrameskip-v4"]  # , "SeaquestNoFrameskip-v4"]

    cur_idx = 0
    num_train = 0

    if args.train:
        for env_name in env_names:
            for attn_target in AttentionTarget:
                for attn_type in AttentionType:
                    if (
                            attn_target == AttentionTarget.A2C or attn_target == AttentionTarget.ICM_LOSS) and attn_type == AttentionType.DOUBLE_ATTENTION:
                        break

                    print(env_name, attn_target, attn_type)

                    # manage start index
                    cur_idx += 1

                    """Environment"""
                    # create the atari environments
                    # NOTE: this wrapper automatically resets each env if the episode is done
                    env = make_atari_env(env_name, num_env=args.num_envs, seed=args.seed)
                    env = VecFrameStack(env, n_stack=args.n_stack)

                    # env = set_env_seed(env, args.seed)

                    """Agent"""
                    agent = ICMAgent(args.n_stack, args.num_envs, env.action_space.n, attn_target, attn_type,
                                     lr=args.lr)

                    if cur_idx > args.idx and num_train < args.num_train:
                        """Train"""

                        # skip if index not achieved
                        num_train += 1
                        param = NetworkParameters(env_name, args.num_envs, args.n_stack, args.rollout_size,
                                                  args.num_updates, args.max_grad_norm, args.icm_beta,
                                                  args.value_coeff, args.entropy_coeff, attn_target, attn_type,
                                                  RewardType.INTRINSIC_AND_EXTRINSIC)
                        runner = Runner(agent, env, param, args.cuda, args.seed, args.log_dir)
                        runner.train()


                    elif num_train == args.num_train:
                        exit()

                    if attn_target == AttentionTarget.NONE:
                        break

    else:
        rend = Renderer(args.env, args.variant, args.log_dir)
        rend.render(args.num_frames, args.seed)
