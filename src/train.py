from os.path import abspath
from time import gmtime, strftime

import numpy as np
import torch
import torch.nn as nn

from logger import TemporalLogger
from storage import RolloutStorage


class Runner(object):

    def __init__(self, net, env, params, is_cuda=True, seed=42, log_dir=abspath("/data/patrik")):
        super().__init__()

        # constants
        self.timestamp = strftime("%Y-%m-%d %H_%M_%S", gmtime())
        self.seed = seed
        self.is_cuda = torch.cuda.is_available() and is_cuda

        # parameters
        self.params = params

        """Logger"""
        self.logger = TemporalLogger(self.params.env_name, self.timestamp, log_dir, *["rewards", "features"])

        """Environment"""
        self.env = env

        self.storage = RolloutStorage(self.params.rollout_size, self.params.num_envs,
                                      self.env.observation_space.shape[0:-1], self.params.n_stack, is_cuda=self.is_cuda)

        """Network"""
        self.net = net

        if self.is_cuda:
            self.net = self.net.cuda()

    def train(self):

        """Environment reset"""
        obs = self.env.reset()
        self.storage.states[0].copy_(self.storage.obs2tensor(obs))
        best_loss = np.inf

        for num_update in range(self.params.num_updates):

            final_value, entropy = self.episode_rollout()

            self.net.optimizer.zero_grad()

            """ICM prediction """
            # tensors for the curiosity-based loss
            # feature, feature_pred: fwd_loss
            # a_t_pred: inv_loss
            icm_loss = self.net.icm(
                self.params.num_envs,
                self.storage.states.view(-1, self.params.n_stack, *self.storage.frame_shape),
                self.storage.actions.view(-1))

            """Assemble loss"""
            a2c_loss, rewards = self.storage.a2c_loss(final_value, entropy, self.params.value_coeff,
                                                      self.params.entropy_coeff)

            loss = a2c_loss + icm_loss

            loss.backward(retain_graph=False)

            # gradient clipping
            nn.utils.clip_grad_norm_(self.net.parameters(), self.params.max_grad_norm)

            """Log rewards & features"""
            if len(self.storage.episode_rewards) > 1:
                self.logger.log(
                    **{"rewards": np.array(self.storage.episode_rewards),
                       "features": self.storage.features[-1].detach().cpu().numpy()})

            self.net.optimizer.step()

            # it stores a lot of data which let's the graph
            # grow out of memory, so it is crucial to reset
            self.storage.after_update()

            if loss < best_loss:
                best_loss = loss.item()
                print("model saved with best loss: ", best_loss, " at update #", num_update)
                torch.save(self.net.state_dict(), "best_agent")

            if num_update % 1000 == 0:
                print("current loss: ", loss.item(), " at update #", num_update)
                self.storage.print_reward_stats()
                # torch.save(self.net.state_dict(), "a2c_time_log_no_norm")

        self.env.close()

        self.logger.save(*["rewards", "features"])
        self.params.save(self.logger.data_dir, self.timestamp)

    def episode_rollout(self):
        episode_entropy = 0
        for step in range(self.params.rollout_size):
            """Interact with the environments """
            # call A2C
            a_t, log_p_a_t, entropy, value, a2c_features = self.net.a2c.get_action(self.storage.get_state(step))
            # accumulate episode entropy
            episode_entropy += entropy

            # interact
            obs, rewards, dones, infos = self.env.step(a_t.cpu().numpy())

            # save episode reward
            self.storage.log_episode_rewards(infos)

            self.storage.insert(step, rewards, obs, a_t, log_p_a_t, value, dones, a2c_features)
            self.net.a2c.reset_recurrent_buffers(reset_indices=dones)

        # Note:
        # get the estimate of the final reward
        # that's why we have the CRITIC --> estimate final reward
        # detach, as the final value will only be used as a
        with torch.no_grad():
            _, _, _, final_value, final_features = self.net.a2c.get_action(self.storage.get_state(step + 1))

        self.storage.features[step + 1].copy_(final_features)

        return final_value, episode_entropy
