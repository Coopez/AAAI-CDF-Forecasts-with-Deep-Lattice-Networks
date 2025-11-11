import torch
import torch.nn as nn

class Neural_Net_with_Quantile(nn.Module):
    def __init__(self, input_size,output_size, num_layers=1, activation="relu"):
        super(Neural_Net_with_Quantile,self).__init__()
        """
        attempt to give linear out some more complexity because quantiles are not linear.
        
        """

        self._layers = nn.ModuleList()
        self._activation = nn.ReLU() if activation == "relu" else nn.Tanh()
        for i in range(num_layers):
            self._layers.append(nn.Linear(input_size, input_size))
        self._layers.append(nn.Linear(input_size, output_size))
    def forward(self, x):
        for i, layer in enumerate(self._layers):
            x = layer(x)
            if i < len(self._layers) - 1:
                x = self._activation(x)
        return x
