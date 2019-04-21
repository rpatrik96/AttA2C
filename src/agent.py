import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.model import A2CNet, ICMNet

class ICMAgent(nn.Module) :
    def __init__(self, num_actions, in_size=288) :
        super().__init__()

        # constants
        self.in_size = in_size
        self.num_actions = num_actions
        self.is_cuda = torch.cuda.is_available()

        # networks
        self.icm = ICMNet(self.num_actions, self.in_size)
        self.a2c = A2CNet(self.num_actions, self.in_size)

        if self.is_cuda :
            self.icm.cuda()
            # self.a2c.cuda()

        # optimizer
        # self.optimizer = optim.Adam(list(self.icm.parameters()) + list(self.a2c.parameters()))
        self.optimizer = optim.Adam(self.a2c.parameters())

    def train(self, env_name, num_epoch, num_steps) :
        """
            s_t : current state
            s_t1: next state

            phi_t: current encoded state
            phi_t1: next encoded state

            a_t: current action
        """
        pass

        env = gym.make(env_name)

        """for i in epoch
        
        calculate reduced extrinsic + (phi_t1_hat-phi_t1)^2 + categorical(a_t, a_t_hat)
        maintain running statistics all of them

        sample action space
        
        """

        for epoch in range(num_epoch) :
            s_t = env.reset()

            for step in range(num_steps) :
                s_t = self.pix2tensor(s_t)
                a_t, policy, value = self.get_action(s_t)  # select action from the policy

                # interact with the environment
                s_t1, r, done, info = env.step(a_t)
                r_cum = self.cumulate_reward(r)
                s_t1 = self.pix2tensor(s_t1)

                # call the ICM model
                a_t = torch.FloatTensor(a_t)
                if self.is_cuda :
                    a_t = a_t.cuda()

                # set_trace()
                phi_t1, phi_t1_pred, a_t_pred = self.icm(s_t, s_t1, a_t)

                # calculate losses
                loss_int = F.mse_loss(phi_t1_pred, phi_t1)
                loss_inv = F.cross_entropy(a_t_pred, a_t)

                self.optimizer.zero_grad()
                # compose losses
                loss = loss_int + loss_inv + r_cum

                print("Epoch: {}, step: {}, loss {}".format(epoch, step, loss))

                loss.backward()
                self.optimizer.step()

                s_t = s_t1  # the current next state will be the new current state
