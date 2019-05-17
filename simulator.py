import numpy as np
import torch
import config
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class Simulator():
    def __init__(self):
        self.time = np.arange(0, config.TRAIN_SIZE, config.DELTA_T)
        self.ca = np.zeros(self.time.size)
        self.cb = np.zeros(self.time.size)
        self.targets = np.zeros(self.time.size)
        self.target = 0.1
        self.K = config.K
        self.e_t = 0.0
        self.control = np.zeros(self.time.size)
        self.update()
        plt.plot(self.time, self.cb)
        plt.plot(self.time, self.targets)
        plt.show()

    def update(self):

        for i in range(1, self.time.size):
            if i % config.STEP_T == 0:
                self.target = config.CB_MAX * np.random.random_sample()
            if i % config.STEP_K == 0:
                self.K = config.K * (1 - config.K_VAR/2) + config.K * config.K_VAR * np.random.random_sample()

            self.ca[i] = self.ca[i - 1] + (config.F / config.V * (self.control[i - 1] - self.ca[i - 1]) - self.K * self.ca[i - 1]) * config.DELTA_T
            self.cb[i] = self.cb[i - 1] + (-config.F / config.V * self.cb[i - 1] + self.K / 2 * self.ca[i - 1]) * config.DELTA_T
            self.e_t += config.DELTA_T * (self.target - self.cb[i] + self.target - self.cb[i - 1]) / 2
            self.control[i] = config.K1 * (self.target - self.cb[i]) + config.K2 * self.e_t + config.K3 * (self.cb[i - 1] - self.cb[i]) / config.DELTA_T
            self.targets[i] = self.target

    def get(self, ind):
        if ind == 0:
            ind = 1

        return torch.tensor([self.cb[ind], self.targets[ind], self.cb[ind] - self.targets[ind]], dtype=torch.float32), torch.tensor([(self.control[ind] - self.control[ind-1])/config.CA_MAX], dtype=torch.float32)


class Data(Dataset):

    def __init__(self, simulator):
        self.simulator = simulator

    def __len__(self):
        return config.TRAIN_SIZE

    def __getitem__(self, ind):
        return self.simulator.get(ind)
