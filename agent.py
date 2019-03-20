import torch.nn as nn
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

