from network import Network
import torch.optim as optim
import config
import torch.nn as nn
import torch
import argparse
from simulator import Data
from simulator import Simulator
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader


def run_trainer():

    enable_cuda = False
    if torch.cuda.is_available():
        print("Using cuda")
        enable_cuda = True

    criterion = nn.MSELoss()

    if enable_cuda:
        criterion = criterion.cuda()

    net = Network()

    for p in net.parameters():  # Zero out gradients in the parameters
        p.grad = torch.zeros(p.data.size(), requires_grad=True)
        p.grad.data.zero_()

    if enable_cuda:
        net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    simulator = Simulator()
    data = Data(simulator)
    loader = DataLoader(data, batch_size=config.BATCH_SIZE, num_workers=8, shuffle=True)

    for epoch in range(0, config.EPOCHS):
        total_loss = 0.0
        num_data = 0
        for data in loader:

            measurements, target = data

            if enable_cuda:
                measurements, target = measurements.cuda(), target.cuda()

            optimizer.zero_grad()

            out = net(measurements)  # Get control action

            if enable_cuda:
                out = out.cuda()
            loss = criterion(out, target)

            loss.backward()

            optimizer.step()
            num_data += 1
            total_loss += loss.item()

        print(f'epoch {epoch} last batch loss: {total_loss/num_data:.10f}')

    torch.save(net.state_dict(), 'nn5.pt')


def run_test():

        np.random.seed(config.RAND_SEED + 1932)

        net = Network()
        net.load_state_dict(torch.load('nn5.pt'))

        time = np.arange(0, config.TEST_SIZE, config.DELTA_T)
        ca1 = np.zeros(time.size)
        cb1 = np.zeros(time.size)
        ca2 = np.zeros(time.size)
        cb2 = np.zeros(time.size)
        control = np.zeros(time.size)
        targets = np.zeros(time.size)

        e_int = 0
        mse_error1 = 0.0
        mse_error2 = 0.0
        target = 5
        kinetic_constant = config.K

        control_prev = 0

        for i in range(1, time.size):
            if time[i] % config.STEP_T == 0:
                target = config.CB_MAX * np.random.random_sample()
            if time[i] % config.STEP_K == 0:
                kinetic_constant = config.K * (1 - config.K_VAR / 2) + config.K * config.K_VAR * np.random.random_sample()

            # PID

            ca1[i] = ca1[i - 1] + (config.F / config.V * (control[i - 1] - ca1[i - 1]) - kinetic_constant * ca1[i - 1]) * config.DELTA_T
            cb1[i] = cb1[i - 1] + (-config.F / config.V * cb1[i - 1] + kinetic_constant / 2 * ca1[i - 1]) * config.DELTA_T
            e_int += config.DELTA_T * (target - cb1[i] + target - cb1[i - 1]) / 2
            control[i] = config.K1 * (target - cb1[i]) + config.K2 * e_int + config.K3 * (cb1[i - 1] - cb1[i]) / config.DELTA_T
            mse_error1 += (target - cb1[i]) ** 2
            targets[i] = target

            # NN

            inputs = torch.tensor([cb2[i-1], target, cb2[i-1] - target], dtype=torch.float32)
            action = net(inputs)
            control_prev += action.item() * config.CA_MAX
            if control_prev < 0:
                control_prev = 0
            if control_prev > config.CA_MAX:
                control_prev = config.CA_MAX
            ca2[i] = ca2[i - 1] + (config.F / config.V * (control_prev - ca2[i - 1]) - kinetic_constant * ca2[i - 1]) * config.DELTA_T
            cb2[i] = cb2[i - 1] + (-config.F / config.V * cb2[i - 1] + kinetic_constant / 2 * ca2[i - 1]) * config.DELTA_T
            mse_error2 += (target - cb2[i]) ** 2

        print(f'PID total error: {mse_error1}')
        print(f'NN total error: {mse_error2}')
        plt.figure(1)
        # a, = plt.plot(time, cb1)
        a, = plt.plot(time, cb2)
        b, = plt.plot(time, targets)
        plt.xlabel("time (s)")
        plt.ylabel("Concentration of B (mol/m^3)")
        plt.legend([a, b], ["Output", "Target"])
        plt.show()






def main():

    parser = argparse.ArgumentParser(description='Project')
    parser.add_argument('--train',
                        dest='run_trainer',
                        action='store_true',
                        default=False,
                        help='Run trainer with parameters in config')
    parser.add_argument('--test',
                        dest='run_test',
                        action='store_true',
                        default=False,
                        help='Run test against PID')

    args = parser.parse_args()

    if args.run_trainer:
        run_trainer()

    if args.run_test:
        run_test()


if __name__ == "__main__":
    main()
