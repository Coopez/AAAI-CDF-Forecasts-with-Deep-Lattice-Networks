import torch
import torch.nn as nn  


class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,window_size, output_size = 1, dtype = torch.float32):
  
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.window_size = window_size
        self.output_size = output_size # non factor as we do not have an output layer
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.dtype = dtype
        self.lstm = nn.ModuleList()
        self.lstm.append(nn.LSTM(self.input_size, self.hidden_size[0],batch_first=True,dtype=dtype))
        in_size = self.hidden_size[0]
        for layer in self.hidden_size[1:]:
            self.lstm.append(nn.LSTM(in_size, layer,batch_first=True,dtype=dtype))
            in_size = layer


    def forward(self, x:torch.tensor,cs = None):
        # first layer
        out, (hn, cn) = self.lstm[0](x)
        for idx in range(1,len(self.lstm)):
            out, (hn, cn) = self.lstm[idx](out,(hn,cn))
        return out[:,-1,:]