from typing import Any
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class YullieWalker:
    def __init__(self):
        pass

    @staticmethod
    def build_system_of_equantions(x, win_len, step=1):
        y = x[win_len::step]
        X = x[:-step].unfold(dimension=0, size=win_len, step=step).flip(dims=[-1, ])    
        return X, y    
    
    @staticmethod
    def solve_yw(X, y):
        A = torch.matmul(X.T, X)
        b = torch.matmul(X.T, y)
        theta = torch.linalg.lstsq(A, b).solution        
        return theta   
    
    @staticmethod
    def est_ar_params(x, win_len, step):
        R, r = YullieWalker.build_system_of_equantions(x=x, win_len=win_len, step=step)
        theta = YullieWalker.solve_yw(R, r)
        y_pred = torch.matmul(R, theta)
        y_pred = torch.cat((x[:win_len], y_pred))
        return y_pred, theta
        

if __name__ == "__main__":
    fs = 200
    f = 12
    t = torch.arange(0, 1, 1/fs).float()
    x = torch.sin(2*np.pi*f*t)
    std_noise = 0.1
    y = x + torch.randn_like(x)*std_noise
    y_pred, theta = YullieWalker.est_ar_params(y, win_len=3, step=1)
    # X, y = build_system_of_equantions(x, step=1, win_len=3)    
    # y_pred, p = yw(X, y, p=3)    
    err = y_pred-y
    err_rel = 20*torch.log10(err.abs().mean()/y.abs().mean())
    err_abs = 20*torch.log10(err.abs().mean())
    print(20*np.log10(std_noise))
    plt.plot(y, '.-')
    plt.plot(y_pred, 'ro-')        
    plt.title("rel err:{:.2f} abs err:{:.2f}".format(err_rel, err_abs))
    plt.grid(True)
    plt.show()    