import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class AdaptiveObjectCounting(nn.Module):
    def __init__(self, input_len, input_dim, hidden_dim, num_classes, num_layer=2):
        super().__init__()
        self.input_len = input_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.proj = MLP(input_dim, hidden_dim, num_classes, num_layer)


    def forward(self, x, mask=None, aggregate=False):
        """
        x: [B Q C]   or [l B Q C] if aggregate
        return: [B num_classes]
        """
        x = self.proj(x)
        if aggregate:
            x = x.mean(dim=0)
        x1 = x.sum(dim=1)
        return x1, x


    
