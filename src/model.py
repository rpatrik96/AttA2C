import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ConvBlock(nn.Module) :
    """ 4 Conv2d + LeakyReLU """

    def __init__(self, ch_in=4) :
        super(ConvBlock, self).__init__()

        # constants
        self.num_filter = 32
        self.size = 3
        self.stride = 2
        self.pad = self.size // 2

        # layers
        self.conv1 = nn.Conv2d(ch_in, self.num_filter, self.size, self.stride, self.pad)
        self.conv2 = nn.Conv2d(self.num_filter, self.num_filter, self.size, self.stride, self.pad)
        self.conv3 = nn.Conv2d(self.num_filter, self.num_filter, self.size, self.stride, self.pad)
        self.conv4 = nn.Conv2d(self.num_filter, self.num_filter, self.size, self.stride, self.pad)

    def forward(self, x) :
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))


        x = nn.AvgPool2d(2)(x) # needed as the input image is 84x84, not 42x42
        # return torch.flatten(x)
        return  x.view(x.shape[0], -1) # retain batch size

class FeatureEncoderNet(nn.Module) :
    """ Network for feature encoding

        In: [s_t]
            Current state (i.e. pixels) -> 1 channel image is needed

        Out: phi(s_t)
            Current state transformed into feature space

    """

    def __init__(self, in_size, n_stack, is_lstm=True) :
        super().__init__()
        # constants
        self.in_size = in_size
        self.h1 = 256
        self.is_lstm = is_lstm  # indicates whether the LSTM is needed

        # layers
        self.conv = ConvBlock(ch_in=n_stack)
        if self.is_lstm :
            self.lstm = nn.LSTMCell(input_size=self.in_size, hidden_size=self.h1)

    def reset_lstm(self, x) :
        if self.is_lstm :
            with torch.no_grad() :
                self.h_t1 = self.c_t1 = torch.zeros(x, self.h1).cuda() if torch.cuda.is_available() else torch.zeros(x,
                                                                                                                     self.h1)

    def forward(self, x) :
        x = self.conv(x)

        if self.is_lstm :
            x = x.view(-1, self.in_size)
            self.h_t1, self.c_t1 = self.lstm(x, (self.h_t1, self.c_t1))  # h_t1 is the output
            return self.h_t1  # [:, -1, :]#.reshape(-1)

        else :
            return x.view(-1, self.in_size)

class InverseNet(nn.Module) :
    """ Network for the inverse dynamics

        In: torch.cat((phi(s_t), phi(s_{t+1}), 1)
            Current and next states transformed into the feature space, 
            denoted by phi().

        Out: \hat{a}_t
            Predicted action

    """

    def __init__(self, num_actions, feat_size=288) :
        super().__init__()

        # constants
        self.feat_size = feat_size
        self.fc_hidden = 256
        self.num_actions = num_actions

        # layers
        self.fc1 = nn.Linear(self.feat_size * 2, self.fc_hidden)
        self.fc2 = nn.Linear(self.fc_hidden, self.num_actions)

    def forward(self, x) :
        return self.fc2(self.fc1(x))

class ForwardNet(nn.Module) :
    """ Network for the forward dynamics

    In: torch.cat((phi(s_t), a_t), 1)
        Current state transformed into the feature space, 
        denoted by phi() and current action

    Out: \hat{phi(s_{t+1})}
        Predicted next state (in feature space)

    """

    def __init__(self, in_size) :
        super().__init__()

        # constants
        self.in_size = in_size
        self.fc_hidden = 256
        self.out_size = 288

        # layers
        self.fc1 = nn.Linear(self.in_size, self.fc_hidden)
        self.fc2 = nn.Linear(self.fc_hidden, self.out_size)

    def forward(self, x) :
        return self.fc2(self.fc1(x))

class AdversarialHead(nn.Module) :
    def __init__(self, feat_size, num_actions) :
        super().__init__()

        # constants
        self.feat_size = feat_size
        self.num_actions = num_actions

        # networks
        self.fwd_net = ForwardNet(self.feat_size + self.num_actions)
        self.inv_net = InverseNet(self.num_actions, self.feat_size)

    def forward(self, phi_t, phi_t1, a_t) :
        """
            phi_t: current encoded state
            phi_t1: next encoded state

            a_t: current action
        """

        # forward dynamics
        # predict next encoded state
        fwd_in = torch.cat((phi_t, a_t), 1)  # concatenate next to each other
        phi_t1_hat = self.fwd_net(fwd_in)

        # inverse dynamics
        # predict the action between s_t and s_t1
        inv_in = torch.cat((phi_t, phi_t1), 1)
        a_t_hat = self.inv_net(inv_in)

        return phi_t1_hat, a_t_hat

class ICMNet(nn.Module) :
    def __init__(self, num_actions, in_size=288, feat_size=256) :
        super().__init__()

        # constants
        self.in_size = in_size  # pixels i.e. state
        self.feat_size = feat_size
        self.num_actions = num_actions

        # networks
        self.feat_enc_net = FeatureEncoderNet(self.in_size, is_lstm=False)
        self.pred_net = AdversarialHead(self.in_size, self.num_actions)  # goal: minimize prediction error
        self.policy_net = AdversarialHead(self.in_size, self.num_actions)  # goal: maximize prediction error
        # (i.e. predict states which can contain new information)

    def forward(self, s_t, s_t1, a_t) :
        """
            s_t : current state
            s_t1: next state

            phi_t: current encoded state
            phi_t1: next encoded state

            a_t: current action
        """

        # encode the states

        phi_t = self.feat_enc_net(s_t)
        phi_t1 = self.feat_enc_net(s_t1)

        # HERE COMES THE NEW THING (currently commented out)
        phi_t1_pred, a_t_pred = self.pred_net(phi_t, phi_t1, a_t)
        # phi_t1_policy, a_t_policy = self.policy_net_net(phi_t, phi_t1, a_t)

        return phi_t1, phi_t1_pred, a_t_pred  # (phi_t1_pred, a_t_pred), (phi_t1_policy, a_t_policy)

class A2CNet(nn.Module) :
    def __init__(self, num_actions, n_stack, in_size=288) :
        super().__init__()

        # constants
        self.in_size = in_size  # in_size
        self.num_actions = num_actions

        # networks
        self.feat_enc_net = FeatureEncoderNet(self.in_size, n_stack=n_stack)
        self.actor = nn.Linear(self.feat_enc_net.h1, self.num_actions)  # estimates what to do
        self.critic = nn.Linear(self.feat_enc_net.h1,
                                1)  # estimates how good the value function (how good the current state is)

    def forward(self, s_t) :
        """
            s_t : current state

            phi_t: current encoded state
        """
        phi_t = self.feat_enc_net(s_t)

        policy = self.actor(phi_t)
        value = self.critic(phi_t)

        return policy, torch.squeeze(value)

    def get_action(self, s_t) :
        """

        :param s_t: current state
        :param num_env: number of parallel environments
        :return: tuple of (a_t, log_prob_a_t, value)
        """

        """Evaluate the A2C"""
        policy, value = self(s_t)  # use A3C to get policy and value

        """Calculate action"""
        # 1. convert policy outputs into probabilities
        # 2. sample the categorical  distribution represented by these probabilities
        action_prob = F.softmax(policy, dim=-1)
        cat = Categorical(action_prob)
        a_t = cat.sample()

        # from   pdb import set_trace
        # set_trace()
        return (a_t, cat.log_prob(a_t), value)

        """# 1. convert policy outputs into probabilities
        # 2. sample the multinomial distribution represented by these probabilities
        action_prob = F.softmax(policy, dim=-1)
        # action_prob = action_prob[0,:] # only one row is needed
        a_t = action_prob.multinomial(num_env, replacement=True)

        # index into the probability tensor
        # to get the corresponding probability for each action
        from pdb import set_trace
        set_trace()
        return (a_t, torch.log(action_prob[a_t]), value)"""
