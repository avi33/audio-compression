import torch
import torch.nn as nn
import numpy as np

class ContentEncoder(nn.Module):
    def __init__(self, dim_input, dim_latent, win_len, hop_len, n_fft):
        super(ContentEncoder, self).__init__()
        self.block = nn.Conv1d(dim_input, dim_latent, 1, 1)        
        self.fft_params = {"win_len": win_len,
                           "hop_len": hop_len,
                           "n_fft": n_fft}
    
        c = (n_fft//2+1)*2
        nf = 256
        block = [nn.Conv1d(c, nf, kernel_size=5, stride=1, padding=2, groups=2, bias=False),
                 nn.BatchNorm1d(nf), 
                 nn.LeakyReLU(0.1, True)]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        X = torch.stft(x.squeeze(1), win_length=self.fft_params["win_len"], hop_length=self.fft_params["hop_len"], n_fft=self.fft_params["n_fft"], return_complex=True)
        X = torch.cat((X.real, X.imag), dim=1)
        # b, c, n = X.shape                
        z = self.block(X)
        return z

if __name__ == "__main__":
    b = 2
    c = 1
    n = 128
    x = torch.randn(b, n)
    E = ContentEncoder(dim_input=1, dim_latent=16, win_len=16, hop_len=4, n_fft=16)
    y = E(x)        
    print(y)    
    print(x.shape, y.shape)