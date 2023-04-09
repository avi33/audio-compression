import torch
import torch.nn as nn

class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, expert_num, output_dim):
        super().__init__()
        
        # Initialize the experts and the gating network
        self.experts = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(expert_num)])
        self.gating_network = nn.Linear(input_dim, expert_num)
        
        # Initialize the output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Compute the expert activations
        expert_activations = torch.stack([expert(x) for expert in self.experts], dim=2)
        
        # Compute the gate activations
        gate_activations = torch.relu(self.gating_network(x))
        
        # Compute the mixture of expert activations
        expert_mixture = torch.sum(expert_activations * gate_activations.unsqueeze(1), dim=2)
        
        # Compute the output
        output = self.output_layer(expert_mixture)
        
        return output


if __name__ == "__main__":
    from utils.helper_funcs import count_parameters
    net = MoE(input_dim=258, hidden_dim=256, expert_num=30, output_dim=258)
    x = torch.randn(2, 258, 62)
    y = net(x)
    print(y.shape)
    print('num of parameters:{:.2f}'.format(count_parameters(y)/1e6))