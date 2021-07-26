import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math

import numpy as np

from time import time

class Policy(nn.Module):
    def __init__(self, num_inputs, action_n, lr=0.0003, cnn_d=10, shift=False, win_size=31):
        super(Policy, self).__init__()
        self.num_inputs = num_inputs
        self.window_size = win_size
        self.num_action = action_n
        self.shift =shift

        D1 = cnn_d
        D2 = 15
        self.conv1 = nn.Conv2d(self.num_inputs, D1, kernel_size=(1, 2))
        self.conv2 = nn.Conv2d(D1, D2, kernel_size=(1, self.window_size-1))
        self.conv2_5 = nn.Conv2d(1, self.num_action-1 , kernel_size=(self.num_action-1, 1))
        self.conv3 = nn.Conv2d(D2+1, self.num_action, kernel_size=(self.num_action-1, 1))
        self.leakyrelu = nn.LeakyReLU()

        self.optimizer = optim.AdamW([
            dict(params=[*self.conv1.parameters()]),
            dict(params=self.conv2.parameters(),
                 weight_decay=5e-9), # L2 reg
            dict(params=self.conv3.parameters(),
                 weight_decay=5e-8), # L2 reg
            dict(params=self.conv2_5.parameters(),
                 weight_decay=5e-9),
        ], lr=lr)

    def forward(self, x, last_action):

        B = x.shape[0] # x.shape == (B, F, C, W)
        if self.shift:
            x = x-1

        x = self.conv1(x)  # (B, 3, C, W-1)
        x = self.leakyrelu(x)

        x = self.conv2(x)  # (B, D1, C, 1)
        x2 = x.transpose(1,3)
        x2 = self.conv2_5(x2)
        x2 = x2.permute(0,3,1,2)
        x = x + x2
        x = self.leakyrelu(x)
    
        prev_w = last_action.view(B, 1, self.num_action-1, 1)  # (B, 1, C, 1)
        x = torch.cat([x, prev_w], 1)  # (B, 12, C, 1
        x = self.conv3(x)  # (B, 1, C, 1)

        x = x[:, :, 0, 0]
        x[x != x] = 0
        x = torch.softmax(x, -1)  # (B, 1+C)

        return x


    def train_net(self,x, y, last_w, args=None):
        prob = self(x, last_w)
        
        if args.cash_reward != 'default':
            # use market awareness cash reward
            y[:, 0] = 1 + (args.cr_shift - torch.mean(y[:, 1:], 1)) * args.cr_scale

        # reward
        L1 = -torch.mean((y-1)*prob)

        # baseline reward
        baseline = torch.mean((y-1)*(1/self.num_action))

        # max weight regularization
        if args.rm_money_reg:
            th = torch.exp((torch.max(prob[:, 1:], 1)[0] * 10) - 4)
        else:
            th = torch.exp((torch.max(prob, 1)[0] * 10) - 4)
        reg_loss = args.reg_w * th.mean()

        # weight changing penalty
        if args.rm_smooth_cost:
            smooth_loss = 0
        else:
            cost = torch.abs(prob[:, 1:]- last_w)
            smooth_loss = 1e-3 * cost.mean()
        
        # total loss
        loss = L1 + baseline + reg_loss + smooth_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 0.1)

        return L1.detach().cpu().numpy()
      
class PVNet(nn.Module):
    def __init__(self, num_inputs, action_n, window_size):
        super(PVNet, self).__init__()
        self.num_inputs = num_inputs
        self.window_size = window_size
        self.num_action = action_n
        D1 = 10
        D2 = 15
        self.model = nn.Sequential(
            nn.Conv2d(self.num_inputs, D1, kernel_size=(1, 2)),# (B, 3, C, W-1)
            nn.LeakyReLU(),
            nn.Conv2d(D1, D2, kernel_size=(1, self.window_size-1)),# (B, D1, C, 1)
            nn.LeakyReLU(),
            nn.Conv2d(D2, self.num_action, kernel_size=(self.num_action, 1))# (B, D1, 1, 1)
        )

    def forward(self, x):
        x = self.model(x)
        return x




