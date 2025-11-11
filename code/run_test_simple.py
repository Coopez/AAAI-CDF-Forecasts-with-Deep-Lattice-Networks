import torch
device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
##############################
from metrics.metrics import Metrics
from metrics.metric_plots import MetricPlots
from res.data import data_import, Data_Normalizer
from dataloader.calibratedDataset import CalibratedDataset

import numpy as np

from utils.helper_func import generate_surrogate_quantiles, return_features, return_Dataframe
from config import  params, _DATA_DESCRIPTION

from testing.test_loop import test_model
# from models.persistence_model import Persistence_Model
from models.smart_day_persistence import sPersistence_Forecast
import os
import pandas as pd


def test(Seed,iter):
    _LOG_NEPTUNE = False

    # pytorch random seed
    torch.manual_seed(Seed)


    train,train_target,valid,valid_target,test,test_target= data_import() #dtype="float64"

    start_date = "2016-01-01 00:30:00"
    end_date = "2020-12-31 23:30:00"    
    index = pd.date_range(start=start_date, end = end_date, freq = '1h', tz='CET')
    i_series = np.arange(0, len(index), 1)
    test_index = i_series[:len(test_target)]
    overall_time = index.values
        

    # normalize train and valid
    Normalizer = Data_Normalizer(train,train_target,valid,valid_target,test,test_target)
    train,train_target,valid,valid_target,test,test_target = Normalizer.transform_all()
    # Training set
    y = test_target
    X = return_Dataframe(test)


    data = CalibratedDataset(X, y, idx=  test_index, device=device,params=params) 
    dataloader = torch.utils.data.DataLoader(data, batch_size=params['batch_size'], shuffle=False, generator=torch.Generator(device=device))

    # features_lattice = return_features(quantiles,params,data=None)


    model = torch.load(find_test_model(params['test_model_name'],iter,params), map_location=device)
    #build_model(params=params,device=device,features=features_lattice).to(device)


    persistence = sPersistence_Forecast(Normalizer,params)

    metric = Metrics(params,Normalizer,_DATA_DESCRIPTION)
    metric_plots = MetricPlots(params,Normalizer,sample_size=params["valid_plots_sample_size"],log_neptune=_LOG_NEPTUNE)

    
    results,times, array_results = test_model(params = params,
                    model = model,
                    metric = metric,
                    metric_plots = metric_plots,
                    dataloader_test = dataloader,
                    data_test = data,
                    log_neptune=_LOG_NEPTUNE,
                    neptune_run=  None,
                    overall_time = overall_time,
                    persistence = persistence)

    if itr == 1:
        # Print the number of parameters in the model
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters in the model: {num_params}")
    return  results,times, array_results


def find_test_model(name:str,iter,params:dict=None) -> str:
    """
    Find the test model path based on the name.
    """
    
    
    
    _loc_data = os.getcwd()
    if os.path.basename(_loc_data) == "code":
        model_path = os.path.join(_loc_data, "saved_test")
    else:

        model_path = os.path.join(_loc_data, "code","saved_test")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    
    file_name = "{iter}_lstm-{name}_test.pt".format(iter=iter, name=name)
    model_path = os.path.join(model_path, file_name)
    
    return model_path






if __name__ == "__main__":
    results = params['metrics'].copy()
    array_results = params['array_metrics'].copy()
    times = []
    for itr,seed in enumerate(params['random_seed'],start=1):
        current_results,time,current_array_results = test(seed,itr)
        times.append(time)
        for key, value in current_results.items():
            results[key].append(value)
    # Aggregate results: compute mean for each metric

        for key, value in current_array_results.items():
            array_results[key].append(value)
       
    for key,value in array_results.items():
        array_results[key] = np.vstack(value)

    mean_array_results = {key: np.mean(values,axis = 0) for key, values in array_results.items()}
    var_array_results = {key: np.std(values,axis = 0) for key, values in array_results.items()}
    mean_results = {key: np.mean(values) for key, values in results.items()}
    var_results = {key: np.std(values) for key, values in results.items()}
    print("Mean results across seeds:")
    for key, value in mean_results.items():
        print(f"{key}: {value}")
    print("STD of results across seeds:")
    for key, value in var_results.items():
        print(f"{key}: {value}")
    print("Forward pass times on average and variance:")
    print(np.mean(times, axis=0))
    print(np.std(times, axis=0))
    print("#################################")
    print("#Mean array results across seeds:#")
    for key, value in mean_array_results.items():
        # Convert array to string with comma-separated values
        array_str = ", ".join(map(str, value))
        print(f"{key}: [{array_str}]")
    print("#STD of array results across seeds:#")
    for key, value in var_array_results.items():
        array_str = ", ".join(map(str, value))
        print(f"{key}: [{array_str}]")
        # print(f"{key}: {value}")
        