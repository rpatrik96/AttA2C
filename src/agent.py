import torch
import torch.nn as nn
import torch.optim as optim

from model import A2CNet, ICMNet

class ICMAgent(nn.Module):
    def __init__(self, n_stack, num_envs, num_actions, in_size=288, feat_size=256, lr=1e-4):
        super().__init__()

        # constants
        self.n_stack = n_stack
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.in_size = in_size
        self.feat_size = feat_size
        self.is_cuda = torch.cuda.is_available()

        # networks
        self.icm = ICMNet(self.n_stack, self.num_actions, self.in_size, self.feat_size)
        self.a2c = A2CNet(self.n_stack, self.num_envs, self.num_actions, self.in_size)

        if self.is_cuda:
            self.icm.cuda()
            self.a2c.cuda()

        # optimizer
        self.lr = lr
        self.optimizer = optim.Adam(list(self.icm.parameters()) + list(self.a2c.parameters()), self.lr)