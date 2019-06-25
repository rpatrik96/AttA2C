import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from storage import RolloutStorage
from utils import HyperparamScheduler, TemporalLogger


class Runner(object):

    def __init__(self, net, env, params, tensorboard_log=False, log_path="./log", is_cuda=True, seed=42):
        super().__init__()

        # constants
        self.seed = seed
        self.is_cuda = torch.cuda.is_available() and is_cuda

        # parameters
        self.params = params
        self.curiosity_coeff = HyperparamScheduler(params.curiosity_coeff, 0.0)

        # objects
        self.logger = TemporalLogger()
        """Tensorboard logger"""
        self.writer = SummaryWriter(comment="statistics",
                                    log_dir=log_path) if tensorboard_log else None

        """Environment"""
        self.env = env

        self.storage = RolloutStorage(self.rollout_size, self.num_envs, self.env.observation_space.shape[0:-1],
                                      self.n_stack, is_cuda=self.is_cuda, value_coeff=params.value_coeff,
                                      entropy_coeff=params.entropy_coeff, writer=self.writer)

        """Network"""
        self.net = net
        self.net.a2c.writer = self.writer

        if self.is_cuda:
            self.net = self.net.cuda()

        # self.writer.add_graph(self.net, input_to_model=(self.storage.states[0],)) --> not working for LSTMCEll

    def train(self):

        """Environment reset"""
        obs = self.env.reset()
        self.storage.states[0].copy_(self.storage.obs2tensor(obs))
        best_loss = np.inf

        for num_update in range(self.num_updates):

            final_value, entropy = self.episode_rollout()

            self.net.optimizer.zero_grad()

            """ICM prediction """
            # tensors for the curiosity-based loss
            # feature, feature_pred: fwd_loss
            # a_t_pred: inv_loss
            feature, feature_pred, a_t_pred = self.net.icm(
                self.num_envs,
                self.storage.states.view(-1, self.n_stack, *self.storage.frame_shape),
                self.storage.actions.view(-1))

            """Curiosity loss"""
            # how bad it can predict the next state
            curiosity_loss = (self.storage.features[1:, :, :]
                              .view(-1, self.storage.feature_size) - feature_pred.detach()).pow(2).mean()

            if self.writer is not None:
                self.writer.add_scalar("curiosity_loss", curiosity_loss.item())

            """Assemble loss"""
            a2c_loss, rewards = self.storage.a2c_loss(final_value, entropy)
            loss =  a2c_loss \
                   + self.icm_loss(feature, feature_pred, a_t_pred, self.storage.actions) \
                   - self.curiosity_coeff.param * curiosity_loss

            loss.backward(retain_graph=False)

            # gradient clipping
            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)

            if self.writer is not None:
                self.writer.add_scalar("loss", loss.item())

            """Log rewards & features"""
            self.logger.log(rewards, feature.detach().cpu().numpy())

            # code for logging gradients
            # params = list(self.net.parameters())
            # for i, param in enumerate(params):
            #     self.writer.add_histogram("param_" + str(i) + "_" + str(list(param.grad.shape)), param.grad.detach())

            self.net.optimizer.step()

            # it stores a lot of data which let's the graph
            # grow out of memory, so it is crucial to reset
            self.storage.after_update()

            # update curiosity loss, it should be decreased, otherwise, the feature distribution will not be normal
            self.curiosity_coeff.step()

            if loss < best_loss:
                best_loss = loss.item()
                print("model saved with best loss: ", best_loss, " at update #", num_update)
                torch.save(self.net.state_dict(), "a2c_best_loss_no_norm")

            elif num_update % 10 == 0:
                print("current loss: ", loss.item(), " at update #", num_update)
                self.storage.print_reward_stats()

            elif num_update % 100 == 0:
                torch.save(self.net.state_dict(), "a2c_time_log_no_norm")

            if self.writer is not None and len(self.storage.episode_rewards) > 1:
                self.writer.add_histogram("episode_rewards", torch.tensor(self.storage.episode_rewards))

        self.env.close()

        self.logger.save()

    def episode_rollout(self):
        episode_entropy = 0
        for step in range(self.rollout_size):
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

    def icm_loss(self, features, feature_preds, action_preds, actions):

        # forward loss
        # measure of how good features can be predicted
        loss_fwd = F.mse_loss(feature_preds, features)

        if self.writer is not None:
            pass
            # self.writer.add_histogram("icm_features", features.detach())
            # self.writer.add_histogram("icm_feature_preds", feature_preds.detach())

        # inverse loss
        # how good is the action estimate between states
        loss_inv = F.cross_entropy(action_preds.view(-1, self.net.num_actions), actions.long().view(-1))

        if self.writer is not None:
            pass
            # self.writer.add_scalar("loss_fwd", loss_fwd.item())
            # self.writer.add_scalar("loss_inv", loss_inv.item())

        return loss_fwd + loss_inv
