import numpy as np
import torch
import torch.nn as nn
from storage import RolloutStorage
from torch.utils.tensorboard import SummaryWriter

# todo: model save

class Runner(object):

    def __init__(self, net, env, optimizer, num_envs, rollout_size=5, num_updates=3000000, is_cuda=True):
        super().__init__()

        # constants
        self.num_envs = num_envs
        self.rollout_size = rollout_size
        self.num_updates = num_updates

        self.max_grad_norm = 0.5

        # loss scaling coefficients
        self.entropy_coeff = 0.02
        self.value_coeff = 0.5
        torch.autograd.set_detect_anomaly(True)

        self.is_cuda = torch.cuda.is_available() and is_cuda

        # objects
        self.writer = SummaryWriter(comment="grads", log_dir="../../log/own")
        self.net = net
        self.net.writer = self.writer
        self.env = env
        self.optimizer = optimizer
        self.storage = RolloutStorage(self.rollout_size, self.num_envs, (84, 84), 4, self.is_cuda)
        # set_trace()

        if self.is_cuda:
            self.net = self.net.cuda()

        # self.writer.add_graph(self.net, input_to_model=(self.storage.states[0],))

    def train(self):

        """Environment reset"""
        obs = self.env.reset()
        self.storage.states[0].copy_(self.storage.obs2tensor(obs))
        best_loss = np.inf

        for num_update in range(self.num_updates):

            final_value, entropy = self.episode_rollout()

            self.optimizer.zero_grad()
            loss = -self.entropy_coeff * entropy + self.a2c_loss(self.storage.compute_reward(final_value),
                                                                 self.storage.log_probs,
                                                                 self.storage.values)
            loss.backward(retain_graph=False)
            nn.utils.clip_grad_norm_(self.net.parameters(),
                                     self.max_grad_norm)

            # set_trace()
            # params = list(self.net.parameters())
            # for i, param in enumerate(params):
            #     self.writer.add_histogram("param_" + str(i) + "_" + str(list(param.grad.shape)), param.grad.detach())

            self.optimizer.step()

            # it stores a lot of data which let's the graph
            # grow out of memory, so it is crucial to reset
            self.storage.after_update()

            if loss < best_loss:
                best_loss = loss.item()
                print("model saved with best loss: ", best_loss, " at update #", num_update)
                torch.save(self.net.state_dict(), "a2c_best_loss_no_norm")

            elif num_update % 10 == 0:
                print("current loss: ", loss.item(), " at update #", num_update)
                self.storage.print_reward_stats()

            elif num_update % 100 == 0:
                torch.save(self.net.state_dict(), "a2c_time_log_no_norm")

            if len(self.storage.episode_rewards) > 1:
                self.writer.add_histogram("episode_rewards", torch.tensor(self.storage.episode_rewards))

        self.env.close()

    def episode_rollout(self):
        episode_entropy = 0
        for step in range(self.rollout_size):

            """Interact with the environments """
            # call A2C
            a_t, log_p_a_t, entropy, value = self.net.get_action(self.storage.get_state(step))

            episode_entropy += entropy
            # interact
            obs, rewards, dones, infos = self.env.step(a_t.cpu().numpy())
            # self.env.render()

            # save episode reward
            self.storage.log_episode_rewards(infos)

            self.storage.insert(step, rewards, obs, a_t, log_p_a_t, value, dones)
            self.net.reset_recurrent_buffers(reset_indices=dones)

        # Note:
        # get the estimate of the final reward
        # that's why we have the CRITIC --> estimate final reward
        # detach, as the final value will only be used as a
        with torch.no_grad():
            _, _, _, final_value = self.net.get_action(self.storage.get_state(step + 1))
        return final_value, episode_entropy

    def a2c_loss(self, rewards, log_prob, values):
        # calculate advantage
        # i.e. how good was the estimate of the value of the current state
        advantage = rewards - values

        # weight the deviation of the predicted value (of the state) from the
        # actual reward (=advantage) with the negative log probability of the action
        # taken (- needed as log(p) p in [0;1] < 0)
        policy_loss = (-log_prob * advantage.detach()).mean()

        # the value loss weights the squared difference between the actual
        # and predicted rewards
        value_loss = advantage.pow(2).mean()

        # return the a2c loss
        # which is the sum of the actor (policy) and critic (advantage) losses
        # due to the fact that batches can be shorter (e.g. if an env is finished already)
        # MEAN is used instead of SUM
        loss = policy_loss + self.value_coeff * value_loss

        self.writer.add_scalar("loss", loss.item())
        self.writer.add_scalar("policy_loss", policy_loss.item())
        self.writer.add_scalar("value_loss", value_loss.item())
        self.writer.add_histogram("advantage", advantage.detach())
        self.writer.add_histogram("rewards", rewards.detach())
        self.writer.add_histogram("action_prob", log_prob.detach())

        return loss
