import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def update_ema_jit(ema_v: torch.Tensor, model_v: torch.Tensor, decay_per_step: float, model_factor: float):
    ema_v.mul_(decay_per_step).add_(model_factor * model_v.float())

class Predictor(nn.Module):
    def __init__(self, emb_dim, latent_dim):
        super(Predictor, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(emb_dim, latent_dim, 3, 1, 1, bias=False),
            nn.BatchNorm1d(latent_dim),
            nn.LeakyReLU(0.1, True),
            nn.Conv1d(latent_dim, emb_dim, 3, 1, 1, bias=False),
            nn.BatchNorm1d(emb_dim),
            nn.LeakyReLU(0.1, True),
        )

    def forward(self, x):        
        return self.block(x)

class BYOL(nn.Module):
    def __init__(self, online_network, target_network, predictor_network, ema_dacay=0.995):
        super(BYOL, self).__init__()
        self.remove_jit = False
        # initialize online and target networks
        self.online_network = online_network
        self.target_network = target_network
        # ensure same initialization
        self.target_network.load_state_dict(self.online_network.state_dict())
        
        # initialize predictor network
        self.predictor_network = predictor_network
                
        self.target_network.eval()            
        self.ema_dacay = ema_dacay
        
    def update_target_network(self):
        # Update target network using momentum-based update rule
        with torch.no_grad():
            for online_param, target_param in zip(self.online_network.parameters(), self.target_network.parameters()):
                if self.remove_jit:
                    target_param.data = self.tau * target_param.data + (1 - self.tau) * online_param.data
                else:
                    update_ema_jit(target_param.data, online_param.data, self.ema_dacay, 1-self.ema_dacay)
    
    def forward(self, x1, x2):
        # Compute online network outputs
        z1 = self.online_network(x1)
        
        # Compute target network outputs
        with torch.no_grad():            
            z2 = self.target_network(x2).detach()
        
        # Compute online and target network predictions
        p1 = self.predictor_network(z1)
        p2 = self.predictor_network(z2)        

        return z1, z2, p1, p2
    
    @staticmethod
    def compute_loss(z1, z2, p1, p2):
        loss = 0.5*(F.mse_loss(p1, z2.detach(), reduction="sum") + F.mse_loss(p2, z1.detach(), reduction="sum"))
        return loss