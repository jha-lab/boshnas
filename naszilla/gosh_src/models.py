import torch
import torch.nn as nn
from .constants import *
from .npn import *

class student(nn.Module):
    def __init__(self, feats):
        super(student, self).__init__()
        self.name = "student"
        self.run = nn.Sequential(
            nn.Linear(feats, 64),
            nn.Softplus(),
            nn.Linear(64, 32),
            nn.Softplus(),
            nn.Linear(32, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.run(x)
        return x

class teacher(nn.Module):
    def __init__(self, feats):
        super(teacher, self).__init__()
        self.name = "teacher"
        self.run = nn.Sequential(
            nn.Linear(feats, 64),
            nn.Softplus(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.Softplus(),
            nn.Dropout(0.5),
            nn.Linear(32, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = x.flatten()
        x = self.run(x)
        return x

class npn(nn.Module):
    def __init__(self, feats):
        super(npn, self).__init__()
        self.name = "npn"
        self.run = nn.Sequential(
            NPNLinear(feats, feats // 2, False),
            NPNRelu(),
            NPNLinear(feats // 2, feats // 4),
            NPNRelu(),
            NPNLinear(feats // 4, 1),
            NPNSigmoid())

    def forward(self, x):
        x = x.reshape(1, -1)
        x, s = self.run(x)
        return x, s
