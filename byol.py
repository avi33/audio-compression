import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def update_ema_jit(ema_v: torch.Tensor, model_v: torch.Tensor, decay_per_step: float, model_factor: float):
    ema_v.mul_(decay_per_step).add_(model_factor * model_v.float())

class BYOL(nn.Module):
    def __init__(self, online_network, target_network, predictor_network, tau=0.996):
        super(BYOL, self).__init__()
        self.remove_jit = False
        # Initialize online and target networks
        self.online_network = online_network
        self.target_network = target_network
        #ensure same initialization
        self.target_network.load_state_dict(self.online_network.state_dict())
        
        # Initialize predictor network
        self.predictor_network = predictor_network
        
        # Set target network to eval mode
        self.target_network.eval()
        
        # Set optimizer
        self.optimizer = optim.SGD(self.online_network.parameters(), lr=0.2, momentum=0.9, weight_decay=1.5e-6)
        
        # Set temperature parameter for loss function
        self.tau = tau
        
    def update_target_network(self):
        # Update target network using momentum-based update rule
        with torch.no_grad():
            for online_param, target_param in zip(self.online_network.parameters(), self.target_network.parameters()):
                if self.remove_jit:
                    target_param.data = self.tau * target_param.data + (1 - self.tau) * online_param.data
                else:
                    update_ema_jit(target_param.data, online_param.data, self.tau, 1-self.tau)
    
    def forward(self, x1, x2):
        # Compute online network outputs
        z1 = self.online_network(x1)
        
        # Compute target network outputs
        with torch.no_grad():            
            z2 = self.target_network(x2)
        
        # Compute online and target network predictions
        p1 = self.predictor_network(z1)
        p2 = self.predictor_network(z2)
        
        self.update_target_network()

        return z1, z2, p1, p2
    
    def compute_loss(self, z1, z2, p1, p2):
        # Compute BYOL loss
        loss = F.mse_loss(p1, z2.detach()) + F.mse_loss(p2, z1.detach())
        return loss        

    def train_step(self, x1, x2):
        # Compute loss and perform backward pass
        loss = self.forward(x1, x2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_target_network()
        
        return loss