import torch.nn as nn
import torch


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.f1 = nn.Linear(3, 64)
        self.f2 = nn.Linear(64, 64)
        self.f3 = nn.Dropout(p=0.1)
        self.f4 = nn.Linear(64, 1)
        self.sigmoid = torch.sigmoid
        self.leakyReLU = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, t):

        t = self.leakyReLU(self.f3(self.f1(t)))
        t = self.leakyReLU(self.f3(self.f2(t)))
        t = self.leakyReLU(self.f3(self.f2(t)))
        t = self.leakyReLU(self.f3(self.f2(t)))
        t = self.leakyReLU(self.f3(self.f2(t)))
        t = self.leakyReLU(self.f3(self.f2(t)))
        t = self.leakyReLU(self.f3(self.f2(t)))
        t = self.tanh(self.f4(t))

        return t

