import torch
import torch.nn as nn

@torch.jit.script
def update_ema_jit(ema_v: torch.Tensor, model_v: torch.Tensor, decay_per_step: float, model_factor: float):
    ema_v.mul_(decay_per_step).add_(model_factor * model_v.float())

class KMeans(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.K = params['num_centers']
        self.ema_decay = params['ema_decay']
        self.tol = params['tol']
        self.eps = params['eps']
        self.max_iters = params['max_iters']        
        self.register_buffer("ema_centers", torch.empty(0))
        self.is_first = True
        self.handle_dead_centers = True
    
    def _init_centers(self, x):
        centers = x[torch.randperm(x.shape[0])[:self.K], :]
        self.ema_centers = centers.clone()
        self.is_first = False
        self.dead_center_reinint = 2

    def forward(self, x):
        b, c, t = x.shape
        x = x.transpose(2, 1).contiguous().view(-1, c)
        N = b*t
        if self.training:
            self._init_centers(x)
        centers = torch.zeros_like(self.ema_centers)
        counts = torch.zeros(self.K, dtype=torch.float32, device=x.device)
        one = torch.ones(N, device=x.device)
        for i in range(self.max_iters):
            dists = torch.cdist(x, self.ema_centers)
            labels = dists.argmin(dim=1)
            #zero 
            centers.fill_(0)
            counts.fill_(0)
            centers.scatter_add_(0, labels.unsqueeze(-1).expand(-1, c), x)
            counts.scatter_add_(0, labels, one)
            centers /= counts.unsqueeze(-1) + self.eps

            if self.handle_dead_centers:
                dead_centers = counts==0
                if torch.any(dead_centers):
                    centers[dead_centers, :] = self.ema_centers[dead_centers, :]
                    self.dead_center_reinint = -1
                    if self.dead_center_reinint == 0:
                        centers = self.ema_centers.clone()
                        dead_centers = torch.zeros(self.K, dtype=torch.bool, device=x.device)
                        self.dead_center_reinint = 2
            update_ema_jit(self.ema_centers, centers, self.ema_decay, 1-self.ema_decay)
            err = torch.norm(self.ema_centers - centers)
            if err < self.tol:
                break
        else:
            dists = torch.cdist(x, self.ema_centers)
            labels = dists.argmin(dim=1)
        # labels = labels.view(b, t, c).transpose(2, 1).contiguous()
        print(self.ema_centers)
        return labels
    

if __name__ == "__main__":
    params = {
        "ema_decay": 0.7,
        "num_centers": 2,
        "tol": 1e-3,
        "eps": 1e-8,
        "max_iters": 100
    }
    model = KMeans(params)
    x = torch.zeros(100, 2)
    x[:50, :] = torch.randn(50, 2) + 5
    x[50:, :] = torch.randn(50, 2) - 5
    labels = model(x.unsqueeze(-1))
    print(labels)