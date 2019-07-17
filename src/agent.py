import torch
import torch.nn as nn
import torch.optim as optim

from model import A2CNet, ICMNet


class ICMAgent(nn.Module):
    def __init__(self, n_stack, num_envs, num_actions, attn_target, attn_type, in_size=288, feat_size=256, lr=1e-4):
        """
        Container class of an A2C and an ICM network, the baseline for experimenting with other curiosity-based
        methods.

        :param attn_target:
        :param attn_type:
        :param n_stack: number of frames stacked
        :param num_envs: number of parallel environments
        :param num_actions: size of the action space of the environment
        :param in_size: dimensionality of the input tensor
        :param feat_size: number of the features
        :param lr: learning rate
        """
        super().__init__()

        # constants
        self.n_stack = n_stack
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.in_size = in_size
        self.feat_size = feat_size
        self.is_cuda = torch.cuda.is_available()

        # networks
        self.icm = ICMNet(self.n_stack, self.num_actions, attn_type, attn_type, self.in_size, self.feat_size)
        self.a2c = A2CNet(self.n_stack, self.num_actions, attn_target, self.in_size)
        self.icm.attn_target = self.a2c.attn_target = attn_target
        self.icm.attn_type = self.a2c.attn_type = attn_type

        if self.is_cuda:
            self.icm.cuda()
            self.a2c.cuda()

        # init LSTM buffers with the number of the environments
        self.a2c.set_recurrent_buffers(num_envs)

        # optimizer
        self.lr = lr
        self.optimizer = optim.Adam(list(self.icm.parameters()) + list(self.a2c.parameters()), self.lr)
