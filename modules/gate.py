import torch
import torch.nn as nn
from modules.res_block import ResBlock1d

class Gate(nn.Module):
    def __init__(self, dim_in, n_experts, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        block = [nn.Conv1d(dim_in, dim_in, kernel_size=5, padding=2, stride=1, padding_mode="reflect"), 
                 nn.BatchNorm1d(dim_in), 
                 nn.LeakyReLU(0.1, True)]
        for k in range(3):
            block += [ResBlock1d(dim_in, dilation=3)]
        
        block += [nn.Conv1d(dim_in, n_experts, kernel_size=3, padding=1, stride=1, padding_mode="reflect"),
                  nn.BatchNorm1d(n_experts), 
                  nn.LeakyReLU(0.1, True)]
        
        for k in range(3):
            block += [ResBlock1d(n_experts, dilation=3)]
        self.block = nn.Sequential(*block)
    
    def forward(self, x):
        y = self.block(x)
        return y