import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from model import A3CNet, ICMNet

class ICMAgent(nn.Module):
    def __init__(self, num_actions, in_size=288):
        super().__init__()

        # constants
        self.in_size = in_size
        self.num_actions = num_actions

        # networks
        self.icm = ICMNet(self.num_actions, self.in_size)
        self.a3c = A3CNet(self.num_actions, self.in_size)

        # optimizer
        self.optimizer = optim.Adam( list(self.icm.parameters()) + list(self.a3c.parameters()) )

    def get_action(self, s_t):
        s_t = torch.Tensor(s_t).to(self.device).float()  # copy state to device as float
        #s_t = s_t.float()
        s_t = self.pix2tensor(s_t)
        policy, value = self.a3c(s_t) # use A3C to get policy and value
        action_prob = F.softmax(policy, dim=-1).data.cpu().numpy()

        a_t = self.sel_rnd_idx(action_prob) # detach for action?

        return a_t, value.data.cpu().numpy().squeeze(), policy.detach()

    @staticmethod
    def sel_rnd_idx(p, axis=1):
        r = np.expand_dims(np.random.rand(p.shape[1 - axis]), axis=axis) # insert a new dim with a random value
        return (p.cumsum(axis=axis) > r).argmax(axis=axis)

    def cumulate_reward(self, r):
        return r

    # functions
    def pix2tensor(pix):
        im2tensor = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Grayscale(1),
                                        transforms.Resize((42,42)),
                                        transforms.ToTensor()])

        return im2tensor(pix)

    def train(self, env_name, num_epoch, num_steps):
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

        for epoch in range(num_epoch):
            s_t  = env.reset()

            for step in num_steps:
                a_t, policy, value = self.get_action(s_t) # select action from the policy

                # interact with the environment
                s_t1, r, done, info = env.step(a_t)
                r_cum = self.cumulate_reward(r)

                # call the ICM model
                phi_t1, phi_t1_pred, a_t_pred = self.icm(s_t, s_t1, a_t)


                # calculate losses
                loss_int = F.mse_loss(phi_t1_pred, phi_t1)
                loss_inv = F.cross_entropy(a_t_pred, a_t)

                self.optimizer.zero_grad()
                # compose losses
                loss = loss_int + loss_inv + r_cum

                print("Epoch: {}, step: {}, loss {}".format(epoch, step, loss) )

                loss.backward()
                self.optimizer.step()



                s_t = s_t1 # the current next state will be the new current state 