import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from utils import AttentionType, AttentionTarget


def init(module, weight_init, bias_init, gain=1):
    """

    :param module: module to initialize
    :param weight_init: initialization scheme
    :param bias_init: bias initialization scheme
    :param gain: gain for weight initialization
    :return: initialized module
    """
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class ConvBlock(nn.Module):

    def __init__(self, ch_in=4):
        """
        A basic block of convolutional layers,
        consisting: - 4 Conv2d
                    - LeakyReLU (after each Conv2d)
                    - currently also an AvgPool2d (I know, a place for me is reserved in hell for that)

        :param ch_in: number of input channels, which is equivalent to the number
                      of frames stacked together
        """
        super().__init__()

        # constants
        self.num_filter = 32
        self.size = 3
        self.stride = 2
        self.pad = self.size // 2

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('leaky_relu'))
        # layers
        self.conv1 = init_(nn.Conv2d(ch_in, self.num_filter, self.size, self.stride, self.pad))
        self.conv2 = init_(nn.Conv2d(self.num_filter, self.num_filter, self.size, self.stride, self.pad))
        self.conv3 = init_(nn.Conv2d(self.num_filter, self.num_filter, self.size, self.stride, self.pad))
        self.conv4 = init_(nn.Conv2d(self.num_filter, self.num_filter, self.size, self.stride, self.pad))

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))

        x = nn.AvgPool2d(2)(x)  # needed as the input image is 84x84, not 42x42
        # return torch.flatten(x)
        return x.view(x.shape[0], -1)  # retain batch size


class AttentionNet(nn.Module):

    def __init__(self, attention_size):
        super().__init__()

        self.attention_size = attention_size

        self.attention = nn.Linear(self.attention_size, self.attention_size)

    def forward(self, x):
        return x * F.softmax(self.attention(x), dim=-1)


class FeatureEncoderNet(nn.Module):
    def __init__(self, n_stack, in_size, is_lstm=True):
        """
        Network for feature encoding

        :param n_stack: number of frames stacked beside each other (passed to the CNN)
        :param in_size: input size of the LSTMCell if is_lstm==True else it's the output size
        :param is_lstm: flag to indicate wheter an LSTMCell is included after the CNN
        """
        super().__init__()
        # constants
        self.in_size = in_size
        self.h1 = 288  # todo: changed to 288 from 256
        self.is_lstm = is_lstm  # indicates whether the LSTM is needed

        # layers
        self.conv = ConvBlock(ch_in=n_stack)
        if self.is_lstm:
            self.lstm = nn.LSTMCell(input_size=self.in_size, hidden_size=self.h1)

    def reset_lstm(self, buf_size=None, reset_indices=None):
        """
        Resets the inner state of the LSTMCell

        :param reset_indices: boolean list of the indices to reset (if True then that column will be zeroed)
        :param buf_size: buffer size (needed to generate the correct hidden state size)
        :return:
        """
        if self.is_lstm:
            with torch.no_grad():
                if reset_indices is None:
                    # set device to that of the underlying network
                    # (it does not matter, the device of which layer is queried)
                    self.h_t1 = self.c_t1 = torch.zeros(buf_size, self.h1, device=self.lstm.weight_ih.device)
                else:
                    # set device to that of the underlying network
                    # (it does not matter, the device of which layer is queried)
                    resetTensor = torch.as_tensor(reset_indices.astype(np.uint8), device=self.lstm.weight_ih.device)

                    if resetTensor.sum():
                        self.h_t1 = (1 - resetTensor.view(-1, 1)).float() * self.h_t1
                        self.c_t1 = (1 - resetTensor.view(-1, 1)).float() * self.c_t1

    def forward(self, x):
        """
        In: [s_t]
            Current state (i.e. pixels) -> 1 channel image is needed

        Out: phi(s_t)
            Current state transformed into feature space

        :param x: input data representing the current state
        :return:
        """
        x = self.conv(x)

        # return self.lin(x)

        if self.is_lstm:
            x = x.view(-1, self.in_size)
            self.h_t1, self.c_t1 = self.lstm(x, (self.h_t1, self.c_t1))  # h_t1 is the output
            return self.h_t1  # [:, -1, :]#.reshape(-1)

        else:
            return x.view(-1, self.in_size)


class InverseNet(nn.Module):
    def __init__(self, num_actions, feat_size=288):
        """
        Network for the inverse dynamics

        :param num_actions: number of actions, pass env.action_space.n
        :param feat_size: dimensionality of the feature space (scalar)
        """
        super().__init__()

        # constants
        self.feat_size = feat_size
        self.fc_hidden = 256
        self.num_actions = num_actions

        # layers
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.fc1 = init_(nn.Linear(self.feat_size * 2, self.fc_hidden))
        self.fc2 = init_(nn.Linear(self.fc_hidden, self.num_actions))

    def forward(self, x):
        """
        In: torch.cat((phi(s_t), phi(s_{t+1}), 1)
            Current and next states transformed into the feature space,
            denoted by phi().

        Out: \hat{a}_t
            Predicted action

        :param x: input data containing the concatenated current and next states, pass
                  torch.cat((phi(s_t), phi(s_{t+1}), 1)
        :return:
        """
        return self.fc2(self.fc1(x))


class ForwardNet(nn.Module):

    def __init__(self, in_size):
        """
        Network for the forward dynamics

        :param in_size: size(feature_space) + size(action_space)
        """
        super().__init__()

        # constants
        self.in_size = in_size
        self.fc_hidden = 256
        self.out_size = 288

        # layers
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.fc1 = init_(nn.Linear(self.in_size, self.fc_hidden))
        self.fc2 = init_(nn.Linear(self.fc_hidden, self.out_size))

    def forward(self, x):
        """
        In: torch.cat((phi(s_t), a_t), 1)
            Current state transformed into the feature space,
            denoted by phi() and current action

        Out: \hat{phi(s_{t+1})}
            Predicted next state (in feature space)

        :param x: input data containing the concatenated current state in feature space
                  and the current action, pass torch.cat((phi(s_t), a_t), 1)
        :return:
        """
        return self.fc2(self.fc1(x))


class AdversarialHead(nn.Module):
    def __init__(self, feat_size, num_actions, attn_target, attention_type):
        """
        Network for exploiting the forward and inverse dynamics

        :param attn_target:
        :param attention_type:
        :param feat_size: size of the feature space
        :param num_actions: size of the action space, pass env.action_space.n
        """
        super().__init__()

        # constants
        self.feat_size = feat_size
        self.num_actions = num_actions

        # networks
        self.fwd_net = ForwardNet(self.feat_size + self.num_actions)
        self.inv_net = InverseNet(self.num_actions, self.feat_size)

        # attention
        self.attention_type = attention_type
        self.attn_target = attn_target

        if self.attn_target is AttentionTarget.ICM:
            if self.attention_type == AttentionType.SINGLE_ATTENTION:
                self.fwd_att = AttentionNet(self.feat_size + self.num_actions)
                self.inv_att = AttentionNet(2 * self.feat_size)
            elif self.attention_type == AttentionType.DOUBLE_ATTENTION:
                self.fwd_feat_att = AttentionNet(self.feat_size)
                self.fwd_action_att = AttentionNet(self.num_actions)
                self.inv_cur_feat_att = AttentionNet(self.feat_size)
                self.inv_next_feat_att = AttentionNet(self.feat_size)

    def forward(self, current_feature, next_feature, action):
        """

        :param current_feature: current encoded state
        :param next_feature: next encoded state
        :param action: current action
        :return: next_feature_pred (estimate of the next state in feature space),
                 action_pred (estimate of the current action)
        """

        """Forward dynamics"""
        # predict next encoded state

        # encode the current action into a one-hot vector
        # set device to that of the underlying network (it does not matter, the device of which layer is queried)
        action_one_hot = torch.zeros(action.shape[0], self.num_actions, device=self.fwd_net.fc1.weight.device) \
            .scatter_(1, action.long().view(-1, 1), 1)

        if self.attn_target is AttentionTarget.ICM:
            if self.attention_type == AttentionType.SINGLE_ATTENTION:
                fwd_in = self.fwd_att(torch.cat((current_feature, action_one_hot), 1))
            elif self.attention_type == AttentionType.DOUBLE_ATTENTION:
                fwd_in = torch.cat((self.fwd_feat_att(current_feature), self.fwd_action_att(action_one_hot)), 1)
        else:
            fwd_in = torch.cat((current_feature, action_one_hot), 1)

        next_feature_pred = self.fwd_net(fwd_in)

        """Inverse dynamics"""
        # predict the action between s_t and s_t1
        if self.attn_target is AttentionTarget.ICM:
            if self.attention_type == AttentionType.SINGLE_ATTENTION:
                inv_in = self.inv_att(torch.cat((current_feature, next_feature), 1))
            elif self.attention_type == AttentionType.DOUBLE_ATTENTION:
                inv_in = torch.cat((self.inv_cur_feat_att(current_feature), self.inv_next_feat_att(next_feature)), 1)
        else:
            inv_in = torch.cat((current_feature, next_feature), 1)

        action_pred = self.inv_net(inv_in)

        return next_feature_pred, action_pred


class ICMNet(nn.Module):
    def __init__(self, n_stack, num_actions, attn_target, attn_type, in_size=288, feat_size=256):
        """
        Network implementing the Intrinsic Curiosity Module (ICM) of https://arxiv.org/abs/1705.05363

        :param n_stack: number of frames stacked
        :param num_actions: dimensionality of the action space, pass env.action_space.n
        :param attn_target:
        :param attn_type:
        :param in_size: input size of the AdversarialHeads
        :param feat_size: size of the feature space
        """
        super().__init__()

        # constants
        self.in_size = in_size  # pixels i.e. state
        self.feat_size = feat_size
        self.num_actions = num_actions

        # networks
        self.feat_enc_net = FeatureEncoderNet(n_stack, self.in_size, is_lstm=False)
        self.pred_net = AdversarialHead(self.in_size, self.num_actions, attn_target,
                                        attn_type)  # goal: minimize prediction error
        # self.policy_net = AdversarialHead(self.in_size, self.num_actions)  # goal: maximize prediction error
        # (i.e. predict states which can contain new information)

    def forward(self, num_envs, states, action):
        """

        feature: current encoded state
        next_feature: next encoded state

        :param num_envs: number of the environments
        :param states: tensor of the states
        :param action: current action
        :return:
        """

        """Encode the states"""
        features = self.feat_enc_net(states)

        # slice features
        # this way, we can spare one forward pass
        feature = features[0:-num_envs]
        next_feature = features[num_envs:]

        """ HERE COMES THE NEW THING (currently commented out)"""
        next_feature_pred, action_pred = self.pred_net(feature, next_feature, action)
        # phi_t1_policy, a_t_policy = self.policy_net(feature, next_feature, a_t)

        return next_feature, next_feature_pred, action_pred  # (next_feature_pred, action_pred), (phi_t1_policy, a_t_policy)


class A2CNet(nn.Module):
    def __init__(self, n_stack, num_actions, attn_type, attn_target, in_size=288):
        """
        Implementation of the Advantage Actor-Critic (A2C) network

        :param attn_target:
        :param n_stack: number of frames stacked
        :param num_actions: size of the action space, pass env.action_space.n
        :param attn_type:
        :param in_size: input size of the LSTMCell of the FeatureEncoderNet
        """
        super().__init__()

        # constants
        self.in_size = in_size  # in_size
        self.num_actions = num_actions

        # attention
        self.attn_type = attn_type
        self.attn_target = attn_target
        self.attention = self.attn_target is AttentionTarget.A2C and self.attn_type is AttentionType.SINGLE_ATTENTION

        # networks
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.feat_enc_net = FeatureEncoderNet(n_stack, self.in_size)
        self.actor = init_(nn.Linear(self.feat_enc_net.h1, self.num_actions))  # estimates what to do
        self.critic = init_(nn.Linear(self.feat_enc_net.h1,
                                      1))  # estimates how good the value function (how good the current state is)

        if self.attention:
            self.actor_att = AttentionNet(self.feat_enc_net.h1)
            self.critic_att = AttentionNet(self.feat_enc_net.h1)

    def set_recurrent_buffers(self, buf_size):
        """
        Initializes LSTM buffers with the proper size,
        should be called after instatiation of the network.

        :param buf_size: size of the recurrent buffer
        :return:
        """
        self.feat_enc_net.reset_lstm(buf_size=buf_size)

    def reset_recurrent_buffers(self, reset_indices):
        """

        :param reset_indices: boolean numpy array containing True at the indices which
                              should be reset
        :return:
        """
        self.feat_enc_net.reset_lstm(reset_indices=reset_indices)

    def forward(self, state):
        """

        feature: current encoded state

        :param state: current state
        :return:
        """

        # encode the state
        feature = self.feat_enc_net(state)

        # calculate policy and value function
        if self.attention:
            policy = self.actor(self.actor_att(feature))
            value = self.critic(self.critic_att(feature))
        else:
            policy = self.actor(feature)
            value = self.critic(feature)

        return policy, torch.squeeze(value), feature

    def get_action(self, state):
        """
        Method for selecting the next action

        :param state: current state
        :return: tuple of (action, log_prob_a_t, value)
        """

        """Evaluate the A2C"""
        policy, value, feature = self(state)  # use A3C to get policy and value

        """Calculate action"""
        # 1. convert policy outputs into probabilities
        # 2. sample the categorical  distribution represented by these probabilities
        action_prob = F.softmax(policy, dim=-1)
        cat = Categorical(action_prob)
        action = cat.sample()

        return (action, cat.log_prob(action), cat.entropy().mean(), value,
                feature)  # ide is jön egy feature bypass a self(state-ből)
