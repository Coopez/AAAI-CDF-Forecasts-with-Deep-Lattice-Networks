import pandas as pd
import numpy as np
import torch


class CalibratedDataset(torch.utils.data.Dataset):
    """A class for loading a dataset for a calibrated model."""

    def __init__(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        cs: np.ndarray = None,
        idx: np.ndarray = None,
        device: str = "cpu",
        params: str = dict(),
    ):
        Xc = X.copy()
        self.length = len(Xc)
        self.updated_horizon_size = params['horizon_size']
        self.target_summary = params['target_summary']
        self.horizon_size =   int(self.updated_horizon_size * self.target_summary)
        self.window_size = params['window_size']   

         

        """Initializes an instance of `Dataset`."""
        self.device = device

        # find all quantiles in Xc and pop them into quantiles, stacking them at dim -1
        quantile_columns = [col for col in Xc.columns if "quantile" in col]
        if quantile_columns:
            self.quantiles = torch.stack([torch.from_numpy(Xc.pop(col).values) for col in quantile_columns], dim=-1).to(self.device)
        else:
            self.quantiles = None
        #self.idx_start = 
        self.idx = torch.tensor(idx).to(device) 
        self.data = torch.from_numpy(Xc.values).to(device) 
        self.targets = torch.from_numpy(y.copy())[:, None].to(device)
        if cs is not None:
            self.cs = torch.from_numpy(cs.copy())[:, None].to(device)
        else:
            self.cs = None

    def __len__(self):
        return self.length - self.window_size - self.horizon_size + 1

    def __getitem__(self, idx):

        x = self.data[idx : idx + self.window_size]
        y = self.targets[idx + self.window_size : idx + self.window_size + self.horizon_size]

        if self.cs is not None:
            cs = self.cs[idx + self.window_size : idx + self.window_size + self.horizon_size]
        out_idx = self.idx[idx : idx + self.window_size + self.horizon_size]

        if self.target_summary != 1:
            y = y.view(-1, self.target_summary).mean(dim=1, keepdim=True)
            new_idx = torch.cat((idx[:self.window_size], idx[self.window_size:self.window_size + self.horizon_size:self.target_summary]))
            if cs is not None:
                cs = cs.view(-1, self.target_summary).mean(dim=1, keepdim=True)
            return [x, y, cs, new_idx]
        if self.cs is not None:

            return [x,y,cs,out_idx] 
        else:
            return [x,y,0,out_idx]
    def return_quantile(self, batchsize,quantile_dim=1, constant = False):
        """ returns a quantile tensor of shape (batchsize,window_size,quantile_dim) and a quantile range tensor of shape (quantile_dim)"""
        if quantile_dim <= 2:
            if constant:
                quantile = torch.rand(1, device = self.device)
                quantiles = quantile.repeat(batchsize,1)
            else:
                quantiles = torch.rand(batchsize, device = self.device).unsqueeze(-1)
                #quantiles = quantile.repeat(1,1)
        if quantile_dim == 2: # put inverted quantiles in last dim
            quantiles = quantiles.unsqueeze(-1)
            quantiles = torch.cat([quantiles, 1-quantiles], dim=-1)
        if quantile_dim > 2: # make arbitrary many quantiles if required
            exponent = np.ceil(np.log10(quantile_dim))
            range_setting = 2.5 # originally 1.0, 2.5 sets us to 0.95 CI. needs to be 0.5 for 0.99 CI
            min_range = range_setting / 10**exponent 
            max_range = 1.0 - min_range
            quantile_range = torch.linspace(min_range, max_range, quantile_dim, device=self.device)
            quantiles = quantile_range.repeat(batchsize,1,1)    
            return quantiles, quantile_range
        return quantiles