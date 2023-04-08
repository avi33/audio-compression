import torch
import torch.nn as nn

def WNConv1d(*args, **kwargs):
    return torch.nn.utils.weight_norm(torch.nn.Conv1d(*args, **kwargs))

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = WNConv1d(1, 1, 1, 1)
    def forward(self, x):
        x = self.block(x)
        return x

def create_net(args):
    net = Net()
    return net