import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistanceGumbelSoftmaxVQ(nn.Module):
    def __init__(self, codebook_size, embedding_dim, commitment_cost, decay=0.99):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.temperature = 1
        
        self.codebook = nn.Parameter(torch.randn(codebook_size, embedding_dim))
        self.codebook.requiresGrad = True
        self.ema = EMA(decay)
    
    def update_ema(self):
        self.ema.update(self.codebook)

    def forward(self, inputs):
        #inputs - b, n, c --> b*n, c
        inputs = inputs.view(-1, self.embedding_dim).contiguous()
        #b*n, d
        distances = torch.cdist(inputs, self.codebook, p=2)
        # Calculate the Gumbel-Softmax probabilities        
        codes = F.gumbel_softmax(-distances / self.temperature, tau=1.0, dim=-1, hard=True)
        # Quantize the embeddings
        # compute quantization loss
        quantization_loss = F.mse_loss(codes @ self.codebook, inputs)
        # compute commitment loss
        commitment_loss = self.commitment_cost * F.mse_loss(inputs, codes.detach() @ self.codebook.detach())

        #update codebook
        self.codebook.data = self.decay * self.codebook.data + (1 - self.decay) * codes.detach().transpose(0, 1) @ inputs  
        
        self.update_ema()
        
        return codes, quantization_loss + commitment_loss
            
class EMA(nn.Module):
    def __init__(self, decay):
        super().__init__()
        self.decay = decay
    
    def forward(self, x, y):
        with torch.no_grad():
            x *= self.decay
            x += (1 - self.decay) * y
    
    def update(self, x):
        with torch.no_grad():
            self.x = x.clone()
    
    def get(self):
        return self.x

if __name__ == "__main__":
    speech_vq = DistanceGumbelSoftmaxVQ(codebook_size=128, embedding_dim=32, commitment_cost=0.25)
    speech_features = torch.randn(2, 8, 32)
    quantized, loss = speech_vq(speech_features)