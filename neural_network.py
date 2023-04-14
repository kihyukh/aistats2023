import torch
import numpy as np
from torch import nn
from data import BanditDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MultiplyConstant(torch.nn.Module):
    def __init__(self, c):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()

        self.c = c

    def forward(self, x):
        return self.c * x

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'y = {self.a} * x'

class NN:
    def __init__(self, d: int, m: int, L: int):
        """Constructor

        Args:
            d: input dimension
            m: network width
            L: network depth
        """

        layers = [
            nn.Linear(d, m, bias=False),
            nn.ReLU(),
        ]
        for l in range(L):
            layers.append(nn.Linear(m, m, bias=False))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(m, 1, bias=False))
        self.network = nn.Sequential(*layers).to(device)
        self.m = m

        # Initialization
        for layer in self.network.children():
            if isinstance(layer, nn.Linear):
                output_dim, input_dim = layer.weight.size()
                nn.init.normal_(layer.weight, 0, np.sqrt(2 / self.m))

        self.theta0 = [
            params.data.clone() for params in self.network.parameters()
        ]
        self.p = len(torch.nn.utils.parameters_to_vector(self.theta0))

    def initialize(self):
        for i, params in enumerate(self.network.parameters()):
            params.data = self.theta0[i].clone()
        self.network.zero_grad()


    def feature_map(self, arms):
        """Compute dynamic feature map of each arm.

        Args:
            arms: n x d numpy array where n is the number of arms
                  and d is the dimension
        Return:
            n x n tensor containing dynamic feature maps of arms
        """
        arms = torch.from_numpy(arms).to(device).float()
        G = torch.zeros(len(arms), self.p)
        for i, a in enumerate(arms):
            self.network.zero_grad()
            y = self.network(a)
            y.backward()
            G[i, :] = torch.nn.utils.parameters_to_vector(
                [p.grad for p in self.network.parameters()]
            )
            self.network.zero_grad()
        K = G @ G.T
        return torch.linalg.cholesky(K)

    def train(self, A, history, config):
        C = config
        if C['J'] == 0:
            return
        dataset = BanditDataset(A, history)

        if C['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=C['eta'],
            )
        elif C['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=C['eta'],
            )
        else:
            raise Exception(f'Unsupported optimizer: {optimizer}')
        for j in range(int(C['J'])):
            train_data = DataLoader(
                dataset, batch_size=C['batch_size'], shuffle=True)
            for X, y, n in train_data:
                X, y, n = X.to(device), y.to(device), n.to(device)

                pred = np.sqrt(self.m) * self.network(X)
                loss = (n * ((pred - y) ** 2)).sum() / 2
                if C.get('simple_data'):
                    loss = ((pred - y) ** 2).sum() / 2
                if j == 0 or j == int(C['J']) - 1:
                    print(f'j = {j}; loss = {loss}')
                for i, params in enumerate(self.network.parameters()):
                    loss += C['lambda'] * C['m'] * ((self.theta0[i] - params) ** 2).sum() / 2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

