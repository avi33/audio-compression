import torch
import matplotlib.pyplot as plt

def build_system_of_equantions(x, win_len, step=1):
    y = x[win_len::step]
    X = x[:-step].unfold(dimension=0, size=win_len, step=step)    
    return X, y

def yw(X, y, p=1):
    win_len = p    
    X = X.flip(dims=[-1, ])
    p = torch.linalg.solve(torch.matmul(X.T, X), torch.matmul(X.T, y))
    y_pred = torch.matmul(X, p)
    return y_pred, p

if __name__ == "__main__":
    win_len = 128
    x = torch.randn(100).float()
    X, y = build_system_of_equantions(x, step=1, win_len=4)    
    y_pred, p = yw(X, y, p=win_len)
    err_rel = 20*torch.log10((y_pred-y).abs().mean()/y.abs().mean())
    err_abs = (y_pred-y).abs()
    plt.plot(y)
    plt.plot(y_pred, 'r-')
    plt.title("rel err:{:.2f}".format(err_rel))
    plt.grid(True)
    plt.show()