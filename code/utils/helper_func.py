import numpy as np
import pandas as pd
from pytorch_lattice.models.features import NumericalFeature
from pytorch_lattice.enums import Monotonicity
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from res.data import data_import


def generate_surrogate_quantiles(length: int, params: dict):
    quantiles = np.random.uniform(0,1,length)
    return quantiles

def return_features(quantiles:np.ndarray,params:dict,data:np.ndarray = None):

    if data is None:
        amount = params['lstm_hidden_size'][-1]
    features = []
    data_features = params['lstm_input_size'] # this is the expected number of features in the data
    quantiles = np.expand_dims(quantiles, axis=-1) if len(quantiles.shape) == 1 else quantiles
    if data is not None:
        data = np.expand_dims(data, axis=-1) if len(data.shape) == 1 else data
    
    if params['input_model'] == 'lstm':
        amount = params['lstm_hidden_size'][-1]
    elif params['input_model'] == 'dnn':
        amount = params['dnn_hidden_size'][-1]
    else:
        amount = data_features
    data = np.random.uniform(0, 1, (quantiles.shape[0], amount)) if data is None else data #.astype('f')
    
    for i in range(amount):
        features.append(NumericalFeature(f"feature_{i}", data[...,i], num_keypoints=params['lattice_calibration_num_keypoints']))
    features.append(NumericalFeature(f"quantiles_0", quantiles[...,0], num_keypoints=params['lattice_calibration_num_keypoints_quantile'], monotonicity=Monotonicity.INCREASING))
    return features



def return_Dataframe_with_q(quantiles,data):
    data = np.expand_dims(data, axis=-1) if len(data.shape) == 1 else data
    quantiles = np.expand_dims(quantiles, axis=-1) if len(quantiles.shape) == 1 else quantiles
    dset = {}
    for i in range(data.shape[-1]):
        dset[f"feature_{i}"] = data[...,i]
    for i in range(quantiles.shape[-1]):
        dset[f"quantiles_{i}"] = quantiles[...,i]
    df = pd.DataFrame(dset)
    return df

def return_Dataframe(data):
    data = np.expand_dims(data, axis=-1) if len(data.shape) == 1 else data
    dset = {}
    for i in range(data.shape[-1]):
        dset[f"feature_{i}"] = data[...,i]
    df = pd.DataFrame(dset)
    return df

        

def rank_batches_by_variance(dataloader: torch.utils.data.DataLoader):
    batch_variances = []
    
    for i, batch in enumerate(dataloader):
        training_data, target,cs, idx = batch

        variance = torch.var(target)
        batch_variances.append((i, variance.item()))
    
    ranked_batches = sorted(batch_variances, key=lambda x: x[1], reverse=True)
    
    return ranked_batches


def return_cs():
    """
    Returns cs values with split. Assumes station 11 as target.
    """
    train, _, valid ,_, test, _ = data_import()

    cs_valid = valid[:,20*2 + 11]
    cs_test = test[:,20*2 + 11]
    cs_train = train[:,20*2 + 11]

    return cs_valid, cs_test, cs_train,{"valid":[np.mean(cs_valid),np.std(cs_valid)], "test":[np.mean(cs_test),np.std(cs_test)], "train":[np.mean(cs_train),np.std(cs_train)]}


