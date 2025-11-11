import torch

### Debug tool################
from debug.plot import Debug_model
##############################
from metrics.metrics import Metrics
from metrics.metric_plots import MetricPlots
from res.data import data_import, Data_Normalizer
from dataloader.calibratedDataset import CalibratedDataset
from utils.helper_func import return_cs
from losses.qr_loss import SQR_loss

from utils.helper_func import generate_surrogate_quantiles, return_features, return_Dataframe
from config import  params, _DATA_DESCRIPTION
from models.builder import build_model, build_optimizer
from training.train_loop import train_model

import optuna
import logging
import sys
import os
import pandas as pd
import numpy as np
from neptune.utils import stringify_unsupported
import neptune.integrations.optuna as optuna_utils
from models.smart_day_persistence import sPersistence_Forecast
from torch.optim.lr_scheduler import ReduceLROnPlateau

YOUR_NEPTUNE_PROJECT = "projectname/your_project"  # Set your Neptune project here
_NEPTUNE_API_TOKEN = "YOUR_NEPTUNE_API" # you need this if you want to log to neptune.

RUN_NAME = "YOUR_RUN_NAME"  # Set your run name here
MAIN_LOSS = "CRPS" # this is the main loss that is used for optimization. Use CRPS or ACE.

assert _NEPTUNE_API_TOKEN != "YOUR_NEPTUNE_API", "Please set your neptune api token in the code."
assert YOUR_NEPTUNE_PROJECT != "projectname/your_project", "Please set your neptune project in the code."

_LOG_NEPTUNE = True # do not change this unless you plan to remove all neptune logging.
def objective(trial):
    print(f"STUDY: Starting trial: {trial.number}")
    """
    Objective function for optuna. We need to define the hyperparameter search space here and in the main config file. Ranges are defined in config.py, but the search space is defined here and in the main starting in line 337. At the moment, the search space is configured for SMNN, but it can be adapted if needed. Refer to examples here or the documentation of optuna https://optuna.readthedocs.io/en/stable/index.html
    """
    ##### HPO Logic #####
    # hpo_hidden_size = trial.suggest_categorical("hidden_size", params["hpo_hidden_size"])
    # hpo_num_layers = trial.suggest_categorical("num_layers", params["hpo_num_layers"])
    # params["learning_rate"] = trial.suggest_categorical("learning_rate", params["hpo_lr"])
    # # params["batch_size"] = trial.suggest_categorical("batch_size", params["hpo_batch_size"])
    # # params["window_size"] = trial.suggest_categorical("window_size", params["hpo_window_size"])

    # if params["input_model"] == 'lstm':
    #     params['lstm_hidden_size'] = [hpo_hidden_size] * hpo_num_layers
    #     params['lstm_num_layers'] = hpo_num_layers
    # if params["input_model"] == 'dnn':
    #     params['dnn_hidden_size'] = [hpo_hidden_size] * hpo_num_layers
    #     params['dnn_num_layers'] = hpo_num_layers

    #smnn stuff

    hpo_smnn_exp_1 = trial.suggest_categorical("smnn_exp_1", params["hpo_smnn_exp_1"])
    hpo_smnn_exp_2 = trial.suggest_categorical("smnn_exp_2", params["hpo_smnn_exp_2"])
    hpo_smnn_relu_1 = trial.suggest_categorical("smnn_relu_1", params["hpo_smnn_relu_1"])
    hpo_smnn_relu_2 = trial.suggest_categorical("smnn_relu_2", params["hpo_smnn_relu_2"])
    hpo_smnn_conf_1 = trial.suggest_categorical("smnn_conf_1", params["hpo_smnn_conf_1"])
    hpo_smnn_conf_2 = trial.suggest_categorical("smnn_conf_2", params["hpo_smnn_conf_2"])

    params["smnn_exp"] = (hpo_smnn_exp_1, hpo_smnn_exp_2)
    params["smnn_relu"] = (hpo_smnn_relu_1, hpo_smnn_relu_2)
    params["smnn_conf"] = (hpo_smnn_conf_1, hpo_smnn_conf_2)



    # disable plotting
    params['valid_plots_every'] = 900000
    params['array_metrics'] = dict()

    params["metrics"] = {"ACE": [],  
            "CRPS": []}
    
    ###############################


    device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'

    # pytorch random seed
    torch.manual_seed(params['random_seed'][0])

    if _LOG_NEPTUNE:
        temp = params["array_metrics"]
        params["array_metrics"] = 0 # neptune only supports float and string
        run['parameters'] = stringify_unsupported(params) # neptune only supports float and string
        params["array_metrics"] = temp
    if _DATA_DESCRIPTION ==  "Station 11 Irradiance Sunpoint":
        train,train_target,valid,valid_target,_,test_target= data_import() #dtype="float64"
        _loc_data = os.getcwd()
        cs_valid, cs_test, cs_train, cs_de_norm = return_cs()

        start_date = "2016-01-01 00:30:00"
        end_date = "2020-12-31 23:30:00"    
        index = pd.date_range(start=start_date, end = end_date, freq = '1h', tz='CET')
        i_series = np.arange(0, len(index), 1)
        train_index = i_series[len(test_target)+len(valid_target):]
        valid_index = i_series[len(test_target):len(test_target)+len(valid_target)]
        overall_time = index.values
    else:
        raise ValueError("Data description not implemented")

    
    # normalize train and valid
    Normalizer = Data_Normalizer(train,train_target,valid,valid_target)
    train,train_target,valid,valid_target = Normalizer.transform_all()
    # Training set
    quantiles = generate_surrogate_quantiles(len(train),params)
    y = train_target
    X = return_Dataframe(train)
    # Validation set
    # quantiles_valid = generate_surrogate_quantiles(len(valid),params)

    Xv = return_Dataframe(valid)
    #features = return_features(quantiles,params,data=train)
    data = CalibratedDataset(X, y,cs_train, train_index, device=params['dataloader_device'],params=params) 
    dataloader = torch.utils.data.DataLoader(data, batch_size=params['batch_size'], shuffle=params['train_shuffle'], generator=torch.Generator(device=device))

    data_valid = CalibratedDataset(Xv, valid_target,cs_valid,valid_index, device=params['dataloader_device'],params=params)
    data_loader_valid = torch.utils.data.DataLoader(data_valid, batch_size=params['batch_size'], shuffle=params['valid_shuffle'], generator=torch.Generator(device=device))

    features_lattice = return_features(quantiles,params,data=None)


    model = build_model(params=params,device=device,features=features_lattice).to(device)

    # Forward pass
    # Define loss function and optimizer
    if _LOG_NEPTUNE:
        run['model_summary'] = str(model)

    criterion = SQR_loss(type=params['loss'], lambda_=params['loss_calibration_lambda'], scale_sharpness=params['loss_calibration_scale_sharpness'])
    metric = Metrics(params,Normalizer,_DATA_DESCRIPTION)
    metric_plots = MetricPlots(params,Normalizer,sample_size=params["valid_plots_sample_size"],log_neptune=_LOG_NEPTUNE,trial_num=trial.number)
    optimizer = build_optimizer(params, model)

    persistence = sPersistence_Forecast(Normalizer,params)
    final_loss = train_model(params = params,
                    model = model,
                    optimizer = optimizer,
                    criterion = criterion,
                    metric = metric,
                    metric_plots = metric_plots,
                    dataloader = dataloader,
                    dataloader_valid = data_loader_valid,
                    data = data,
                    data_valid = data_valid,
                    trial = trial,
                    log_neptune=_LOG_NEPTUNE,
                    verbose=False, 
                    neptune_run=  run,
                    overall_time = overall_time,
                    persistence = persistence
                    )

    return final_loss


def return_loss(metrics):
    for metric, value in metrics.items():
            metrics[metric] = np.mean(value)
    if MAIN_LOSS == "ACE":
        metrics["optuna_loss"] = metrics["ACE"]
    elif MAIN_LOSS == "CRPS":
        metrics["optuna_loss"] = metrics["CRPS"]
    else:
        raise ValueError("MAIN_LOSS must be either 'ACE' or 'CRPS'")
    return metrics 




def train_model(params,
                model,
                optimizer,
                criterion,
                metric,
                metric_plots,
                dataloader,
                dataloader_valid,
                data,
                data_valid,
                trial,
                log_neptune=False,
                verbose=False, 
                neptune_run=None,
                overall_time = [],
                persistence = None
                ):

    run = neptune_run 
    run["status/trial"] = trial.number
    device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'
    last_checkpoint = trial.user_attrs.get("last_checkpoint", False)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=params["scheduler_patience"], factor=params["scheduler_factor"], min_lr=params["scheduler_min_lr"])
    epoch_begin = 0
    if last_checkpoint:
        if os.path.exists(f"./tmp/tmp_model_{last_checkpoint}.pt"):
            checkpoint = torch.load(f"./tmp/tmp_model_{last_checkpoint}.pt")
            epoch = checkpoint["epoch"]
            epoch_begin = epoch + 1

            print(f"RESTORATOR: Loading a checkpoint from trial {last_checkpoint} in epoch {epoch}.")

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            #accuracy = checkpoint["accuracy"]


    epochs = params['epochs']
    plot_ids = sorted(list(set([int(x * (512 / params["batch_size"])) for x in [7,24,14,22,37,8,3]])))
    min_loss = 1000.0
    for epoch in range(epoch_begin,epochs):
        model.train()
        for batch in dataloader:
            training_data, target, cs, time_idx = batch
             # 2 quantiles because sharpness loss is logged
            training_data, target,cs = training_data.to(device), target.to(device), cs.to(device)
            # if params["inject_persistence"]:
            #     pers= persistence.forecast(time_idx[:,params["window_size"]]).unsqueeze(-1)
            # else:
            #     pers = None    

            quantile = data.return_quantile(training_data.shape[0],quantile_dim=2)
            quantile = quantile.to(device) # this line may be unnecessary, but it is not a problem to have it here.



            output = forward_pass(params,model,training_data,quantile,quantile_dim=quantile.shape[-1])
            # Compute loss
            loss = criterion(output, target, quantile)

            #Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # train_losses.append(loss.item())

            # with torch.no_grad():
            #     sharpness = sharpness_loss(output,quantile,scale_sharpness_scale=params['loss_calibration_scale_sharpness'])
            #     sharp_losses.append(sharpness.item())
            # if log_neptune:
            #     run['train/'+params['loss']].log(np.mean(train_losses))
            #     if params['loss'] == 'calibration_sharpness_loss':
            #         run['train/sharpness'].log(np.mean(sharp_losses))
    
        model.eval()
        
        if epoch % params['valid_metrics_every'] == 0:
            metric_dict = params['metrics']
            sample_counter = 1
            with torch.no_grad():
                for b_idx,batch in enumerate(dataloader_valid):
                    training_data, target,cs, time_idx = batch
                    training_data, target,cs = training_data.to(device), target.to(device), cs.to(device)
                    # There is not reason not to already use a quantile range here, as we are not training
                    pers= persistence.forecast(time_idx[:,0]).unsqueeze(-1)
                    pers_denorm = persistence.forecast_raw(time_idx[:,0]).unsqueeze(-1)
                    quantile,q_range = data_valid.return_quantile(training_data.shape[0],quantile_dim=params["metrics_quantile_dim"])
                    quantile, q_range = quantile.to(device), q_range.to(device)
                    output = forward_pass(params,model,training_data,quantile,quantile_dim=quantile.shape[-1])
                    # if params['valid_clamp_output']:
                    #     output = torch.clamp(output,min=0)
                    
                    metric_dict= metric(pred = output, truth = target, quantile = quantile, metric_dict=metric_dict,q_range=q_range,pers=pers_denorm)
                    if epoch == params['epochs'] - 1:
                        # metric_array_dict = metric_plots.accumulate_array_metrics(metric_array_dict,pred = output, truth = target, quantile = q_range) #q_range is just the quantiles in a range arrangement. 
                        if b_idx in plot_ids and sample_counter != params["valid_plots_sample_size"]:
                            metric_plots.generate_result_plots(training_data,output, target, quantile, cs, overall_time[time_idx.detach().cpu().numpy()],sample_num = sample_counter, neptune_run=neptune_run)
                            sample_counter += 1


        report_metrics = return_loss(metric_dict)
        if params["scheduler_enable"]:
            scheduler.step(report_metrics["CRPS"])
        trial.report(report_metrics["optuna_loss"],epoch)
        # run["status/epoch_time"] = epoch_time
        # run["status/epoch"] = epoch

        # min_loss = report_metrics["optuna_loss"] if report_metrics["optuna_loss"] < min_loss else min_loss
        min_loss = report_metrics["optuna_loss"]
        run[f"trials/{trial.number}/best_value"] = report_metrics["optuna_loss"]

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            f"./tmp/tmp_model_{trial.number}.pt",
        )
        trial.set_user_attr("last_checkpoint", trial.number)
        # # Handle pruning based on the intermediate value.
        # if trial.should_prune():
        #     if os.path.exists(f"./tmp/tmp_model_{trial.number}.pt"):
        #         os.remove(f"./tmp/tmp_model_{trial.number}.pt")
        #     return min_loss
        #     raise optuna.exceptions.TrialPruned()
    
    if os.path.exists(f"./tmp/tmp_model_{trial.number}.pt"):
        os.remove(f"./tmp/tmp_model_{trial.number}.pt")
     #del run["trials"] # remove trials from server as that is unnecessary meta data.
    return min_loss #report_metrics["optuna_loss"]



def forward_pass(params:dict,
                 model: torch.nn.Module,
                 batch:torch.Tensor, 
                 quantile: torch.Tensor,
                 quantile_dim:int,
                 persistence = None,
                 device='cuda',):
    """
    Handels forward pass through model and does X amount of passes for different quantiles."""
    
    assert quantile_dim == quantile.shape[-1], 'Quantile dimension must match quantile tensor'
    if params['dataloader_device'] == 'cpu':
        batch = batch.to(device)
        quantile = quantile.to(device)
    output = torch.zeros((batch.size()[0],params['horizon_size'],quantile_dim))
    model.train()
    embedded = model[0](batch)
    
    for i in range(quantile_dim):
        aggregated_input = torch.cat([embedded,quantile[...,i]],dim=-1)
        output[...,i] = model[1](aggregated_input)
        if persistence is not None:
            output[...,i] = model[2](x = output[...,i],c = persistence.squeeze(),tau = quantile[0,0,i])
    return output


    

if __name__ == "__main__":
    
    
    import neptune
    from neptune.utils import stringify_unsupported
    
    run = neptune.init_run(
        project=YOUR_NEPTUNE_PROJECT,
        custom_run_id=RUN_NAME,
        name = "HPO",
        api_token=_NEPTUNE_API_TOKEN,
        tags= params["neptune_tags"]
    )
    run['data/type'] = _DATA_DESCRIPTION

    
    # pytorch random seed
    torch.manual_seed(params['random_seed'][0])

    neptune_callback = optuna_utils.NeptuneCallback(run,plots_update_freq=1,log_all_trials=False)

    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = RUN_NAME  # Unique identifier of the study.

    storage_name = "sqlite:///{}.db".format(study_name)

    
    # search_space = {
    #     "hidden_size": params["hpo_hidden_size"],
    #     "num_layers": params["hpo_num_layers"],
    #     "learning_rate": params["hpo_lr"]
    # }
    
    search_space = {
        "smnn_exp_1": params["hpo_smnn_exp_1"],
        "smnn_exp_2": params["hpo_smnn_exp_2"],
        "smnn_relu_1": params["hpo_smnn_relu_1"],
        "smnn_relu_2": params["hpo_smnn_relu_2"],
        "smnn_conf_1": params["hpo_smnn_conf_1"],
        "smnn_conf_2": params["hpo_smnn_conf_2"]
    }
    sampler = optuna.samplers.GridSampler(search_space)
    

    study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="minimize", sampler=sampler)

    study.optimize(objective, n_trials=5,callbacks=[neptune_callback],show_progress_bar=False)

    run.stop()

