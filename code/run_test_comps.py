import torch
device = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
### Debug tool################
from debug.plot import Debug_model
##############################
from res.data import data_import, Data_Normalizer
from dataloader.calibratedDataset import CalibratedDataset
import numpy as np
from losses.qr_loss import SQR_loss

from utils.helper_func import generate_surrogate_quantiles, return_Dataframe
from config import  params, _DATA_DESCRIPTION
from testing.test_loop import test_model
from models.smart_day_persistence import sPersistence_Forecast
from utils.helper_func import return_cs
import os
import pandas as pd
import numpy as np
import time as time_pack

import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn import MSELoss, L1Loss
from metrics.metrics import PICP, PINAW , PICP_quantile, ACE, RSE, MAPE, MSPE, CORR, approx_crps, MSE

def test(Seed,itr):
    _LOG_NEPTUNE = False

    # pytorch random seed
    torch.manual_seed(Seed)
    if params["comp_model"] == "pp" or params["comp_model"] == "sp":
        params["metrics"] = {"MAE": [], "RMSE": [], "SS": []} 
        params["array_metrics"] = {}
    elif params["comp_model"] == "qr":
        params["metrics"] = {"ACE": [], "MAE": [], "RMSE": [], "CRPS": [], "SS": []}
        params["array_metrics"] = {"PICP": [], "Cali_PICP": []}
    else:
        raise ValueError("Unknown comparison model type")

    train,train_target,valid,valid_target,test,test_target= data_import() 
    _loc_data = os.getcwd()
    cs_valid, cs_test, cs_train, cs_de_norm = return_cs()

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
    quantiles = generate_surrogate_quantiles(len(test),params)
    y = test_target
    X = return_Dataframe(test)


    data = CalibratedDataset(X, y,cs_test, test_index, device=device,params=params) 
    dataloader = torch.utils.data.DataLoader(data, batch_size=params['batch_size'], shuffle=False, generator=torch.Generator(device=device))


    if params['comp_model'] == "sp":
        model = None
    else:
        model = torch.load(find_test_model(params['comp_model'],itr,params), map_location=device)
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
                    verbose=False, 
                    neptune_run=  None,
                    overall_time = overall_time,
                    persistence = persistence,
                    base_path=params["data_save_path"])
    itr = 0 if params["comp_model"] == "sp" else itr
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
    
    file_name = "{iter}_{name}-linear.pt".format(iter=iter, name=name)
    model_path = os.path.join(model_path, file_name)
    
    return model_path


class Metrics():
    @torch.no_grad()
    def __init__(self,params,normalizer,data_source):
        self.metrics = params["metrics"]
        self.params = params
        self.lambda_ = params["loss_calibration_lambda"] 
        self.batchsize = params["batch_size"]
        self.horizon = params["horizon_size"]
        if params["input_model"] == "dnn":
            self.input_size = params["dnn_input_size"]
        elif params["input_model"] == "lstm":
            self.input_size = params["lstm_input_size"]
        else:
            raise ValueError("Metrics: Unknown input model type")  
        self.normalizer = normalizer 
        self.quantile_dim = params['metrics_quantile_dim'] 

        self.cs_multiplier = True if self.params["target"] == "CSI" else False
    @torch.no_grad()
    def __call__(self, pred, truth,quantile,cs, metric_dict,q_range,pers=None):
        if not metric_dict:
            return {}
        else:
            results = metric_dict.copy()
        pred_denorm = self.normalizer.inverse_transform(pred,"target")
        truth_denorm = self.normalizer.inverse_transform(truth,"target")
        
        if self.params["comp_model"] == "pp" or self.params["comp_model"] == "sp":
            median = pred_denorm
        else:

            median = pred_denorm[...,int(self.quantile_dim/2)].unsqueeze(-1)
        
        if self.params["valid_clamp_output"]: # Clamping the output to be >= 0.0
            pred_denorm = torch.clamp(pred_denorm, min=0.0)
            median = torch.clamp(median, min=0.0)
        
        for metric in self.metrics:
            
            if metric == 'CS_L':
                sqr = SQR_loss(type='calibration_sharpness_loss', lambda_=self.lambda_)
                results['CS_L'].append(sqr(pred_denorm, truth_denorm, quantile).item()) 
            ### - Deterministic metrics
            elif metric == 'MAE':
                mae = L1Loss()
                results['MAE'].append(mae(median,truth_denorm).item())
            elif metric == 'MSE':
                results['MSE'].append(MSE(median, truth_denorm).item())
            elif metric == 'RMSE':
                rmse = MSELoss()
                results['RMSE'].append(torch.sqrt(rmse(median,truth_denorm)).item())
                #RMSE(median, truth)
            elif metric == 'MAPE':
                results['MAPE'].append(MAPE(median, truth_denorm).item())
            elif metric == 'MSPE':
                results['MSPE'].append(MSPE(median, truth_denorm).item()) 
            elif metric == 'RSE':
                results['RSE'].append(RSE(median, truth_denorm).item())
            elif metric == 'CORR':
                results['CORR'].append(CORR(median, truth_denorm).item()) 
            elif metric == 'SS':
                # we calculate the skill score later in summarize_metrics
                mse = MSELoss()
                results['SS'].append(torch.sqrt(mse(pers,truth_denorm)).item())
            ###
            ### - Probabilistic metrics
            elif metric == 'ACE':
                picp = PICP(pred, truth,quantiles=q_range)
                results['ACE'].append((ACE(picp)).item())    #/(self.batchsize*self.horizon)
            elif metric == 'CRPS':
                results['CRPS'].append(approx_crps(pred_denorm, truth_denorm,quantiles=quantile).item())
            elif metric == 'COV':
                pass
            else:
                raise ValueError(f"Unknown metric: {metric}")
        return results
    
    def summarize_metrics(self, results,verbose=True,neptune=False,neptune_run=None):
        scheduler_metrics = dict()
        for metric, value in results.items():           
            if isinstance(value, (list, tuple, np.ndarray)):
                value_str = np.mean(np.array(value))
            else:
                value_str = value
            if metric == "SS": # calculate skill score for the averages
                value_str = 1 - (scheduler_metrics['RMSE']**2 / value_str**2)
            scheduler_metrics[metric] = value_str
            if metric == "Time":
                if verbose:
                    print(f"{value_str:.1f}s".ljust(8)+"-|", end=' ')
            elif metric == "Epoch":
                if verbose:
                    print("Epoch:" + value_str, end=' ')
            elif metric == "PICP" or metric == "PINAW":
                pass # we don't want to print this
            else:
                if verbose:
                    print((f"{metric}: {value_str:.4f}").ljust(15), end=' ')            
                if neptune:
                    neptune_run[f'valid/{metric}'].log(value_str)
        print(f" ")
        return scheduler_metrics


class MetricPlots:
    def __init__(self, params, normalizer, sample_size=1, log_neptune=False,trial_num = 1,fox=False):
        self.params = params
        self.sample_size = sample_size
        self.log_neptune = log_neptune
        self.save_path = params['valid_plots_save_path']
        self.metrics = params['array_metrics']
        self.normalizer = normalizer
        self.range_dict = {"PICP": None, "PINAW": None, "Cali_PICP": None}
        self.trial_num = trial_num
        self.FOX = fox
        seaborn_style = "whitegrid"
        sns.set_theme(style=seaborn_style, palette="colorblind")
    def accumulate_array_metrics(self,metrics,pred,truth,quantile,pers):
        if not metrics:
            return {}
        midpoint = pred.shape[-1] // 2
        picp, picp_interval = PICP(pred,truth,quantiles=quantile, return_counts=False,return_array=True)
        pinaw, pinaw_interval =PINAW(pred,truth,quantiles=quantile, return_counts=False,return_array=True)
        picp_c, picp_c_quantiles = PICP_quantile(pred,truth,quantiles=quantile, return_counts=False,return_array=True)   
        corrs = error_uncertainty(pred,truth)
        if self.range_dict["PICP"] is None:
            self.range_dict["PICP"] = picp_interval
            self.range_dict["PINAW"] = pinaw_interval
            self.range_dict["Cali_PICP"] = picp_c_quantiles
            self.range_dict["Correlation"] = range(corrs.shape[0])
        if metrics.keys().__contains__("PICP"):
            metrics["PICP"].append(picp)
        if metrics.keys().__contains__("PINAW"):
            metrics["PINAW"].append(pinaw)
        if metrics.keys().__contains__("Cali_PICP"):
            metrics["Cali_PICP"].append(picp_c)

        if metrics.keys().__contains__("RMSE_Horizon"):
            metrics['RMSE_Horizon'].append(torch.mean(torch.sqrt((pred[...,midpoint] - truth.squeeze(-1))**2),axis=0).detach().cpu().numpy())
        if metrics.keys().__contains__("ACE_Horizon"):
            ace = []
            for i in range(self.params["horizon_size"]):
                picp = PICP(pred[:, i, :].unsqueeze(1),truth[:, i, :].unsqueeze(1),quantiles=quantile)
                ace.append(ACE(picp).item())
            metrics['ACE_Horizon'].append(np.array(ace))
        return metrics
    """
    rewrite the code to just take the very last batch of an epoch and calculate the metrics on that. 

    """

    def generate_metric_plots(self,metrics,neptune_run=None, dataloader_length=None):
        plot_data = self._summarize_array_metrics(metrics)
        for name, x_values in self.range_dict.items():
            if name in plot_data.keys():
                self._plot_metric(name,plot_data[name],x_values,neptune_run=neptune_run)
            else:
                pass

    def _summarize_array_metrics(self,metrics):
        summary = self.metrics.copy()
        for element in summary.keys():
            if element not in ["Correlation"]:
                summary[element] = np.mean(np.array(metrics[element]),axis = 0)
            else:
                summary[element] = np.mean(np.array(metrics[element]),axis = 0)
                summary[element]= np.stack((summary[element],np.std(np.array(metrics[element]),axis = 0)))
        return summary

    
    def _plot_metric(self,name, value, ideal = None, neptune_run=None):
        colors = sns.color_palette("colorblind")
        if name == "Correlation":
            stds = value[1]
            value = value[0]
            
        if ideal is not None:
            sorted_indices = np.argsort(ideal)
            value = np.array(value)[sorted_indices]
            ideal = np.array(ideal)[sorted_indices]
            # value = np.insert(value, 0, 0)
            # ideal = np.insert(ideal, 0, 0)
        
        x = np.linspace(ideal[0], ideal[-1], len(value))

        x_label = self.name_assign(name)
        # x_label = "Quantiles" if name == "Cali_PICP" else  "Time steps"  if name=="Correlation" else "Intervals" 
        plt.ioff()  # Turn off interactive mode
        plt.figure(figsize=(4, 3))
        plt.plot(x, value, label=name, linewidth=3,color=colors[0])
        if name == "Cali_PICP" or name == "PICP":         
            plt.plot(x, ideal, label="Ideal", linewidth=3, color = colors[1])
            plt.yticks(np.linspace(0, 1, 5))
        if name == "Correlation":
            plt.fill_between(x, value - stds, value + stds, alpha=0.2, color=colors[0])
        

        plt.xlabel(x_label)  # should be label quantiles for calibration, intervals for picp and pinaw
        plt.ylabel(name)
        plt.legend()
        plt.grid(True)
        
        if name == "Correlation" :
            pass
        else:
            plt.xticks(np.linspace(0,1,5))
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        if not self.FOX:
            wd = os.getcwd()
            if os.path.basename(wd) == "code":
                plt.savefig(f"{self.save_path}/{name}_plot.png")
            else:
                plt.savefig(f"code/{self.save_path}/{name}_plot.png")
            
        plt_fig = plt.gcf()  # Get the current figure
        
        
        if neptune_run is not None:
            neptune_run[f"valid/distribution_{name}"].append(plt_fig)
            neptune_run[f"valid/{name}"].extend(value.tolist())
            if name == "Correlation":
                neptune_run[f"valid/{name}_std"].extend(stds.tolist())
        plt.close()
    def name_assign(self,name):
        if name == "PICP" or name == "PINAW":
            return "Intervals"
        elif name == "Cali_PICP":
            return "Quantiles"
        elif name == "Correlation":
            return "Time steps"
        else:
            return "bieb"
    def generate_result_plots(self,data,pred,truth,quantile,cs,time,sample_num,neptune_run=None):
        """
        Plotting the prediction performance of the model.
        Saves the plots to save_path and logs them to neptune if needed.
        """


            
        sample_idx = range(1)#np.arange(sample_start, sample_start + self.sample_size)
        data = data.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        truth = truth.detach().cpu().numpy()
        if self.params["comp_model"] == "pp" or self.params["comp_model"] == "sp":
            quantile = [0]
        else:
            quantile = quantile.detach().cpu().numpy()

        data_denorm = self.normalizer.inverse_transform(data,"train")
        pred_denorm = self.normalizer.inverse_transform(pred,"target")
        truth_denorm = self.normalizer.inverse_transform(truth,"target")

        if cs is not None and self.params["target"] == "CSI":
            # cs = cs[:self.sample_size].detach().cpu().numpy()
            pred_denorm = pred_denorm * cs.detach().cpu().numpy()
            truth_denorm = truth_denorm * cs.detach().cpu().numpy()

        if self.params["valid_clamp_output"]:
            pred_denorm = np.clip(pred_denorm,a_min = 0,a_max = None)


        # Plotting
        target_max = self.normalizer.max_target
         
        for i in sample_idx:
            self._plot_results(data_denorm[i],pred_denorm[i],truth_denorm[i],quantile[i],time[i],target_max=target_max,sample_num=sample_num,neptune_run=neptune_run)

    
    def _plot_results(self,data,pred,truth,quantile,time,target_max,sample_num,neptune_run=None):
        if self.params["comp_model"] == "pp" or self.params["comp_model"] == "sp":
            x_idx = np.arange(0,len(data),1)
            y_idx = np.arange(len(data),len(data)+len(pred),1)
            # pred_idx = int(quantile.shape[-1] / 2)
            
            plt.ioff()
            plt.figure(figsize=(10, 4))
            colors = sns.color_palette("colorblind")
            plt.plot(time[x_idx], data[:,11], label='Input Data', color=colors[0])
            plt.plot(time[y_idx], truth[:,0], label='Ground Truth', color=colors[2])
            plt.plot(time[y_idx], pred[:,0], label='Prediction', linestyle='--', color=colors[1])
            # plt.fill_between(time[y_idx], pred[:, 0], pred[:, -1], alpha=0.2, label='Prediction Interval', color=colors[1])
            plt.xlabel('Time (DD HH:MM)')
            plt.ylabel('GHI (W/m^2)')
            plt.legend()
            plt.grid(True)
            plt.yticks(np.linspace(0, target_max, 10))
            # plt.xticks(time)
            plt.tight_layout()
            if not self.FOX:
                wd = os.getcwd()
                if os.path.basename(wd) == "code":
                    plt.savefig(f"{self.save_path}/timeseries_plot_{sample_num}.png")
                else:
                    plt.savefig(f"code/{self.save_path}/timeseries_plot_{sample_num}.png")
            plt_fig = plt.gcf()  # Get the current figure

            if neptune_run is not None:
                neptune_run[f"valid/distribution_trial{self.trial_num}_{sample_num}"].append(plt_fig)
                
            plt.close()
        else:
            x_idx = np.arange(0,len(data),1)
            y_idx = np.arange(len(data),len(data)+len(pred),1)
            pred_idx = int(pred.shape[-1] / 2)
            
            plt.ioff()
            plt.figure(figsize=(10, 4))
            colors = sns.color_palette("colorblind")
            plt.plot(time[x_idx], data[:,11], label='Input Data', color=colors[0])
            plt.plot(time[y_idx], truth[:,0], label='Ground Truth', color=colors[2])
            plt.plot(time[y_idx], pred[:,pred_idx], label='Prediction', linestyle='--', color=colors[1])
            plt.fill_between(time[y_idx], pred[:, 0], pred[:, -1], alpha=0.1, label='Prediction Interval', color=colors[1])
            plt.fill_between(time[y_idx], pred[:, 1], pred[:, -2], alpha=0.1, color=colors[1])
            plt.fill_between(time[y_idx], pred[:, 2], pred[:, -3], alpha=0.1, color=colors[1])
            plt.fill_between(time[y_idx], pred[:, 4], pred[:, -5], alpha=0.1, color=colors[1])
            plt.fill_between(time[y_idx], pred[:, 3], pred[:, -4], alpha=0.1, color=colors[1])
            plt.xlabel('Time (DD HH:MM)')
            plt.ylabel('GHI (W/m^2)')
            plt.legend()
            plt.grid(True)
            plt.yticks(np.linspace(0, target_max, 10))
            # plt.xticks(time)
            plt.tight_layout()
            if not self.FOX:
                wd = os.getcwd()
                if os.path.basename(wd) == "code":
                    plt.savefig(f"{self.save_path}/timeseries_plot_{sample_num}.png")
                else:
                    plt.savefig(f"code/{self.save_path}/timeseries_plot_{sample_num}.png")

            plt_fig = plt.gcf()  # Get the current figure

            if neptune_run is not None:
                neptune_run[f"valid/distribution_trial{self.trial_num}_{sample_num}"].append(plt_fig)
                
            plt.close()



def error_uncertainty(y_pred,y):
    
    middle_idx = y_pred.shape[-1] // 2
    mae = torch.abs(y_pred[...,middle_idx] - y.squeeze(-1))
    qwidth = y_pred[..., -1] - y_pred[..., 0]
    batch_size, time_series_length = mae.shape
    correlation_scores = torch.empty(time_series_length)

    mae_mean = torch.mean(mae,dim = 0)
    qwidth_mean = torch.mean(qwidth,dim = 0)
    
    mae_std = torch.std(mae,dim=0, unbiased=True)  # Use unbiased estimator
    qwidth_std = torch.std(qwidth, dim=0, unbiased=True)
    
    normalized_mae = (mae - mae_mean) / (mae_std + 1e-10)
    normalized_qwidth = (qwidth - qwidth_mean) / (qwidth_std + 1e-10)

    correlation_scores = torch.sum(normalized_mae * normalized_qwidth,dim=0) / (batch_size - 1)

    return correlation_scores.detach().cpu().numpy()


class Comp_loss():
    def __init__(self,params):
        super().__init__()
        self.params = params
    def __call__(self, x, y , quantiles=None, cs=None):
        """
        x: model output
        y: target
        quantile: quantile tensor
        cs: calibration set
        """
        if self.params['comp_model'] == "pp" or self.params['comp_model'] == "sp":
            return torch.nn.functional.l1_loss(x,y)
        elif self.params['comp_model'] == "qr":
            return self._qr_loss(x, y, quantiles, cs)
        else:
            raise ValueError("Comparison model not implemented")
    def _qr_loss(self, y_pred, y_true, quantiles, cs=None):
        if quantiles is None:
            # estimating quantiles from model output shape
            quantiles = torch.linspace(0.025, 0.975, steps=y_pred.shape[-1], device=y_pred.device)
        all_loss = list()
        quantiles_loss = list()
        for idx,quantile in enumerate(quantiles):
            value = y_pred[...,idx].unsqueeze(-1) 
            quantiles_loss.append(torch.mean(torch.max(torch.mul(quantile,(y_true-value)),torch.mul((quantile-1),(y_true-value)))))
        return torch.mean(torch.stack(quantiles_loss, dim=0))

def build_model(params):
    from models.DNN_out_model import Neural_Net_with_Quantile
    from models.LSTM import LSTM
    output_size = 1 if params["comp_model"] == "pp" or params["comp_model"] == "sp" else 11
    if params["input_model"] == "lstm":
        input_model = LSTM(input_size= params["lstm_input_size"],
                           hidden_size= params["lstm_hidden_size"],
                           num_layers= params["lstm_num_layers"],
                           window_size= params["window_size"],
                           output_size= 1
                        #    dtype = torch.float64
                           )
        data_output_size = params["lstm_hidden_size"][-1]

    else:
        raise ValueError("Input_Model not implemented")
    
    #options = "lattice", "linear", "constrained_linear", "linear_lattice", "lattice_linear"
    if params["output_model"] == "linear":
        output_model = torch.nn.Linear(in_features= data_output_size,
                                       out_features= params["horizon_size"]* output_size)
    elif params["output_model"] == "dnn":
        output_model = Neural_Net_with_Quantile(input_size= data_output_size,
                                       output_size= params["horizon_size"]* output_size)
    
    else:
        raise ValueError("Output_Model not implemented")
 
    
    model = torch.nn.ModuleList( 
        [input_model,
        output_model
        ]
    )
    return model 

def test_model(params,
                model,   
                metric,
                metric_plots,
                dataloader_test,
                data_test,
                log_neptune=False,
                verbose=False, 
                neptune_run=None,
                overall_time = [],
                persistence = None,
                base_path = None
                ):
    if log_neptune:
        run = neptune_run 
        run_name = run["sys/id"].fetch()
    else:
        run_name = 'local_test_run'

    plot_ids = sorted(list(set([int(x * (512 / params["batch_size"])) for x in [7,24,14,22,37,8,3]])))
    #26,27,7,24,14,22,37,8,3,4,38 with 512
    #old 122, 128, 136, 131, 184, 124, 278 with 64 

    
    if not params["comp_model"] == "sp":
        model.eval()
    
    
    metric_dict = {label: [] for label in params['metrics'].keys()}
    metric_array_dict = {label: [] for label in params['array_metrics'].keys()}
    # metric_dict = params['metrics'].copy()
    # metric_array_dict = params['array_metrics'].copy()
    sample_counter = 1
    # pers_losses = []

    dict_data = {
        "idx": [],
        "target": [],
        "output": []
    }
    with torch.no_grad():
        forward_times = []
        for b_idx,batch in enumerate(dataloader_test):
            training_data, target,cs, time_idx = batch
            pers= persistence.forecast(time_idx[:,0]).unsqueeze(-1)
            pers_denorm = persistence.forecast_raw(time_idx[:,0]).unsqueeze(-1)
            start_time = time_pack.time()
            if params["comp_model"] == "sp":
                output = pers
            else:
                output = forward_pass(params,model,training_data)
            if params["comp_model"] == "pp" or params["comp_model"] == "sp":
                quantile = None
                q_range = None
            elif params["comp_model"] == "qr":
                quantile,q_range = data_test.return_quantile(training_data.shape[0],quantile_dim=params["metrics_quantile_dim"])
                quantile = torch.unique(quantile).sort()[0]
            
            end_time = time_pack.time()
            forward_times.append(end_time - start_time)
            dict_data = {
                "idx": np.append(dict_data["idx"], b_idx),
                "target": np.append(dict_data["target"], target.detach().cpu().numpy()),
                "output": np.append(dict_data["output"], output.detach().cpu().numpy())
            }


            metric_dict= metric(pred = output, truth = target, quantile = quantile, cs = cs, metric_dict=metric_dict,q_range=q_range,pers=pers_denorm)

            
            metric_array_dict = metric_plots.accumulate_array_metrics(metric_array_dict,pred = output, truth = target, quantile = q_range,pers = pers_denorm) #q_range is just the quantiles in a range arrangement. 
            if b_idx in plot_ids and sample_counter != params["valid_plots_sample_size"]+1:
                metric_plots.generate_result_plots(training_data,output, target, quantile, cs, overall_time[time_idx.detach().cpu().numpy()],sample_num = sample_counter, neptune_run=neptune_run)
                sample_counter += 1

    
    metric_plots.generate_metric_plots(metric_array_dict,neptune_run=neptune_run, dataloader_length=len(dataloader_test))
    
    results = metric.summarize_metrics(metric_dict,verbose = False, neptune=False)

    array_results = metric_plots._summarize_array_metrics(metric_array_dict)    
    # data_save = pd.DataFrame(dict_data)
    # if base_path is not None:
    #     data_save.to_csv(f'{base_path}/test_data_{run_name}.csv',index=False)
    # else:
    #     data_save.to_csv(f'test_data_{run_name}.csv',index=False)

    return results,np.mean(forward_times), array_results


def forward_pass(params:dict,
                 model: torch.nn.Module,
                 batch:torch.Tensor, 
                 persistence=None,
                 device='cuda',):
    if params['dataloader_device'] == 'cpu':
        batch = batch.to(device)

    model.train()
    embedded = model[0](batch)
    
    if persistence is not None:
        output = model[1](embedded)
        persistence_output = model[2](x = output,c = persistence.squeeze())
        output = persistence_output
    else:
        output = model[1](embedded)
    
    if params['comp_model'] == "pp":
        output = output.reshape(output.shape[0], params['horizon_size'], 1)
    elif params['comp_model'] == "qr":
        output = output.reshape(output.shape[0], params['horizon_size'], params["metrics_quantile_dim"])
    else:
        raise ValueError("Comparison model not implemented")

    return output




if __name__ == "__main__":
    if params["comp_model"] == "sp":
        
        current_results,time,_ = test(1,0)
        print(current_results)
        print(time)
        
    else:
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

        print("Mean array results across seeds:")
        for key, value in mean_array_results.items():
            print(f"{key}: {value}")
        print("STD of array results across seeds:")
        for key, value in var_array_results.items():
            print(f"{key}: {value}")
            