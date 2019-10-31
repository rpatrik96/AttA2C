from os.path import abspath, dirname, join

import imageio
import numpy as np
import pandas as pd
import torch
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack

from agent import ICMAgent
from utils import make_dir, series_indexer, label_enum_converter, instance2label


class Renderer(object):

    def __init__(self, env, variant, log_dir) -> None:
        super().__init__()
        # sanity check
        if variant not in [0, 4]:
            raise ValueError(f"Invalid variant, got {variant}, should be 0 or 4")
        self.env_name = f"{env.capitalize()}NoFrameskip-v{variant}"
        self.log_dir = log_dir
        self.data_dir = join(self.log_dir, self.env_name)
        self.render_dir = join(dirname(dirname(abspath(__file__))), join("gifs", self.env_name))
        make_dir(self.render_dir)

        self.params_df = pd.read_csv(join(self.data_dir, "params.tsv"), "\t")

    def render(self, steps=2500, seed=42):
        for timestamp in self.params_df.timestamp:
            # query parameters
            instance = self.params_df[self.params_df.timestamp == timestamp]
            n_stack = series_indexer(instance["n_stack"])

            attn_target_enum = label_enum_converter(series_indexer(instance['attention_target']))
            attn_type_enum = label_enum_converter(series_indexer(instance['attention_type']))

            # filenames for loading
            log_points = (.25, .5, .75, .99)
            files2load = [f"agent_best_loss_{timestamp}", f"agent_best_reward_{timestamp}",
                          *[f"agent_step_{i}_{timestamp}" for i in log_points]]

            # name conversion for GIF save
            label = instance2label(instance)
            gif_name = label.lower()
            gif_name = gif_name.replace(", ", "_")
            gif_name = gif_name.replace(" ", "_")
            files2save = [f"{gif_name}_best_loss", f"{gif_name}_best_reward",
                          *[f"{gif_name}_step_{i}" for i in log_points]]

            # iterate and render
            for agent_name, gif_name in zip(files2load, files2save):
                # make environment
                env = make_atari_env(self.env_name, num_env=1, seed=seed)
                env = VecFrameStack(env, n_stack=n_stack)

                # create agent
                agent = ICMAgent(n_stack, 1, env.action_space.n, attn_target_enum, attn_type_enum)

                self.load_and_eval(agent, env, agent_name, gif_name, steps)

    def load_and_eval(self, agent: ICMAgent, env, agent_path, gif_path, steps=2500):
        # load agent and set to evaluation mode
        agent.load_state_dict(torch.load(join(self.data_dir, agent_path)))
        agent.eval()

        # loop and acquire images
        images = []
        obs = env.reset()
        for _ in range(steps):
            tensor = torch.from_numpy(obs.transpose((0, 3, 1, 2))).float() / 255.
            tensor = tensor.cuda() if torch.cuda.is_available() else tensor
            action, _, _, _, _ = agent.a2c.get_action(tensor)
            _, _, _, _ = env.step(action)
            images.append(env.render(mode="rgb_array"))

        # render
        imageio.mimsave(join(self.render_dir, f"{gif_path}.gif"),
                        [np.array(img) for i, img in enumerate(images) if i % 2 == 0], fps=29)
