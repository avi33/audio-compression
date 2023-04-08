import torch.nn as nn

class ResBlock1d(nn.Module):
    def __init__(self, dim, dilation=1, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad1d(dilation * (kernel_size//2)),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=1, bias=False, dilation=dilation, groups=dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=True, groups=dim),
        )        
        
    def forward(self, x):
        return x + self.block(x)
    