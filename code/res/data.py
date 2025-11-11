import torch
import numpy as np
import os
import sys



def data_import(base_path = None,dtype = "float32"):
    # Get the parent directory of the current file (which is .../code)
    # Find the 'code' directory in the current path, even if we're in a subdirectory
    code_dir = os.path.dirname(os.path.abspath(__file__))
    parts = code_dir.split(os.sep)
    if 'code' in parts:
        code_index = parts.index('code')
        parent_dir = os.sep.join(parts[:code_index])
        code_dir = os.sep.join(parts[:code_index+1])
    else:
        # fallback: assume current directory is 'code'
        parent_dir = code_dir
    # Set data_dir to the 'data' directory in the same parent as 'code'
    data_dir = os.path.join(parent_dir, 'data')
    
    filename = "unified_data"
    load_path = os.path.join(data_dir, filename + ".npz")
    
    data = np.load(load_path)
    if dtype == "float32":
        data = {k: v.astype(np.float32) if k != "var_names" else v for k, v in data.items()}
    elif dtype == "float64":
        data = {k: v.astype(np.float64) if k != "var_names" else v for k, v in data.items()}
    return data['flat_train'], data['target_train'], data['flat_valid'] ,data['target_valid'], data['flat_test'], data['target_test']




class Data_Normalizer():
    """
    Class normalizing and outputting train, valid, train_target, valid_target, test, test_target
    unless not supplied, then it will output None.
    Always uses min-max normalization.
    """
    def __init__(self,
                 train: np.ndarray,
                 train_target: np.ndarray,
                 valid: np.ndarray = None,
                 valid_target: np.ndarray = None,
                 test: np.ndarray = None,
                 test_target: np.ndarray = None):
        """
        Initialize the Data_Normalizer with training, validation, and test data.

        Args:
            train (np.ndarray): Training data.
            train_target (np.ndarray): Training target data.
            valid (np.ndarray): Validation data.
            valid_target (np.ndarray): Validation target data.
            test (np.ndarray): Test data.
            test_target (np.ndarray): Test target data.
        """
        self.train = train
        self.train_target = train_target
        self.valid = valid
        self.valid_target = valid_target
        self.test = test
        self.test_target = test_target
        
        self.min_train = np.expand_dims(np.min(train, axis=0), axis=0)
        self.max_train = np.expand_dims(np.max(train, axis=0), axis=0)

        self.min_target = np.min(train_target, axis=0)
        self.max_target = np.max(train_target, axis=0)

        self.min_train_expanded = np.expand_dims(self.min_train.copy(), axis=0)
        self.max_train_expanded = np.expand_dims(self.max_train.copy(), axis=0)

        
    ## Helper functions, not supposed to be called directly
    def apply_minmax(self,var:str, data):
        if var == "train":
            return (data - self.min_train) / (self.max_train - self.min_train)
        elif var == "target":
            return (data - self.min_target) / (self.max_target - self.min_target)
        else:
            raise ValueError("Variable not recognized")
        
    
    #######################################
    def transform_all(self):
        """
        Apply min-max normalization to all datasets.

        Returns:
            tuple: Normalized train, train_target, valid, valid_target, test, test_target datasets.
        """
        local = []

        ## need to apply train min-max to all datasets as if they are not observable
        if self.train is not None:
            local.append(self.apply_minmax("train",self.train))
        if self.train_target is not None:
            local.append(self.apply_minmax("target",self.train_target))
        if self.valid is not None:
            local.append(self.apply_minmax("train",self.valid))
        if self.valid_target is not None:
            local.append(self.apply_minmax("target",self.valid_target))
        if self.test is not None:
            local.append(self.apply_minmax("train",self.test))
        if self.test_target is not None:
            local.append(self.apply_minmax("target",self.test_target))

        return local
    
    def inverse_transform(self, data, target: str="train"):
        if target == "train":
            denormed = data*(self.max_train_expanded - self.min_train_expanded) + self.min_train_expanded
            return denormed
        elif target == "target":
            denormed = data*(self.max_target - self.min_target) + self.min_target
            return denormed
        else:
            raise ValueError("Target not recognized")



class Batch_Normalizer():
    """
    Normalizing everything in a batch of data
    """
    def __init__(self,data: torch.Tensor):
        self.mean = torch.mean(data,dim=1).unsqueeze(1)
        self.std = torch.std(data,dim=1).unsqueeze(1)



    def transform(self,data: torch.Tensor):
        tmean = self.mean.repeat_interleave(data.shape[1], dim=1)
        tstd = self.std.repeat_interleave(data.shape[1], dim=1)
        return (data)/tstd
    def inverse_transform(self,data: torch.Tensor, pos: int = 0):
        tmean = self.mean.repeat_interleave(data.shape[1], dim=1)
        tstd = self.std.repeat_interleave(data.shape[1], dim=1)
        return (data*tstd[...,pos].unsqueeze(-1)) 