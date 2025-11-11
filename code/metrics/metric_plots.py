import os
import torch
from .metrics import PICP, PINAW, PICP_quantile, ACE

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd




class MetricPlots:
    def __init__(self, params, normalizer, sample_size=1, log_neptune=False,trial_num = 1,fox=False,persistence=None):
        self.params = params
        self.sample_size = sample_size
        self.log_neptune = log_neptune
        self.save_path = params['valid_plots_save_path']
        self.metrics = params['array_metrics'].copy()
        self.normalizer = normalizer
        self.range_dict = {"PICP": None, "PINAW": None, "Cali_PICP": None}
        self.trial_num = trial_num
        self.FOX = fox
        self.persistence = persistence
        seaborn_style = "whitegrid"
        sns.set_theme(style=seaborn_style, palette="colorblind")
    def accumulate_array_metrics(self,metrics,pred,truth,quantile,pers):
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
        if metrics.keys().__contains__("Correlation"):
            metrics["Correlation"].append(corrs.tolist())

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
        
        if name == "Correlation":
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
        elif name == "Correlation" :
            return "Time steps"
        else:
            return "name_assgin_failed"
    def generate_result_plots(self,data,pred,truth,quantile,time,sample_num,neptune_run=None):
        """
        Plotting the prediction performance of the model.
        Saves the plots to save_path and logs them to neptune if needed.
        """
        sample_idx = range(1)#np.arange(sample_start, sample_start + self.sample_size)
        data = data.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        truth = truth.detach().cpu().numpy()
        quantile = quantile.detach().cpu().numpy()

        data_denorm = self.normalizer.inverse_transform(data,"train")
        pred_denorm = self.normalizer.inverse_transform(pred,"target")
        truth_denorm = self.normalizer.inverse_transform(truth,"target")


        if self.params["valid_clamp_output"]:
            pred_denorm = np.clip(pred_denorm,a_min = 0,a_max = None)


        # Plotting
        target_max = self.normalizer.max_target
         
        for i in sample_idx:
            self._plot_results(data_denorm[i],pred_denorm[i],truth_denorm[i],quantile[i],time[i],target_max=target_max,sample_num=sample_num,neptune_run=neptune_run)

    
    def _plot_results(self,data,pred,truth,quantile,time,target_max,sample_num,neptune_run=None):
        x_idx = np.arange(0,len(data),1)
        y_idx = np.arange(len(data),len(data)+len(pred),1)
        pred_idx = int(quantile.shape[-1] / 2)
        
        plt.ioff()
        plt.figure(figsize=(10, 4))
        colors = sns.color_palette("colorblind")
        plt.plot(time[x_idx], data[:,11], label='Input Data', color=colors[0])
        plt.plot(time[y_idx], truth[:,0], label='Ground Truth', color=colors[2])
        plt.plot(time[y_idx], pred[:,pred_idx], label='Prediction', linestyle='--', color=colors[1])
        plt.fill_between(time[y_idx], pred[:, 0], pred[:, -1], alpha=0.1, label='Prediction Interval', color=colors[1])
        for i in range(1,pred_idx):
            plt.fill_between(time[y_idx], pred[:, i], pred[:, -1-i], alpha=0.1, color=colors[1])
        plt.xlabel('Time (DD HH:MM)')
        plt.ylabel('GHI (W/m^2)')
        plt.legend()
        plt.grid(True)
        plt.yticks(np.linspace(0, target_max, 10))

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

    def generate_test_plots(self,pred,truth,quantile,time,sample_num):
        sns.set_theme(style="white", palette="colorblind")
        import matplotlib as mpl
        mpl.rcParams['pdf.fonttype'] = 42
        sample_idx = [30,32,34]
        pred = pred.detach().cpu().numpy()
        truth = truth.detach().cpu().numpy()
        quantile = quantile.detach().cpu().numpy()

        pred_denorm = self.normalizer.inverse_transform(pred,"target")
        truth_denorm = self.normalizer.inverse_transform(truth,"target")

        if self.params["valid_clamp_output"]:
            pred_denorm = np.clip(pred_denorm,a_min = 0,a_max = None)


        # Plotting
        target_max = self.normalizer.max_target
         
        for i in sample_idx:
            self._plot_test_results(pred_denorm[i],truth_denorm[i],quantile[i],time[i],target_max=target_max,sample_num=sample_num,idx = i)

    def _plot_test_results(self,pred,truth,quantile,time,target_max,sample_num,idx):
        y_idx = np.arange(len(time)-len(pred),len(time),1)
        pred_idx = int(quantile.shape[-1] / 2)
        
        plt.ioff()
        plt.figure(figsize=(3, 2))
        colors = sns.color_palette("colorblind")
        if self.params["test_model_name"] == "lattice_linear":
            spec_red = colors[0]
        elif self.params["test_model_name"] == "smnn":
            spec_red = colors[2]
        elif self.params["test_model_name"] == "dnn":
            spec_red = colors[1]
        else:
            spec_red = colors[3]
        #spec_red = colors[0] #'#f03b20'
        plt.plot(time[y_idx], truth[:,0], linestyle='--', label='Ground Truth', color="black")
        plt.plot(time[y_idx], pred[:,pred_idx], label='Prediction',  color=spec_red)
        plt.fill_between(time[y_idx], pred[:, 0], pred[:, -1], alpha=0.1, label='Prediction Interval', color=spec_red)
        for i in range(1,pred_idx):
            plt.fill_between(time[y_idx], pred[:, i], pred[:, -1-i], alpha=0.1, color=spec_red)
        plt.xlabel('')
        plt.ylabel('')
        # plt.legend()  # Legend disabled
        plt.grid(False)
        #plt.yticks(np.linspace(0, target_max, 3))
        plt.yticks(np.linspace(0, 1000, 3))
        ticky = time[y_idx]
        plt.xticks([ticky[0], ticky[-1]],[pd.to_datetime(ticky[0]).strftime('%H:%M'), pd.to_datetime(ticky[-1]).strftime('%H:%M')])
        # plt.xticks(time)
        plt.tight_layout()
        model_name = self.params["test_model_name"]
        plt.savefig(f"{self.save_path}/test_ts_{model_name}_{sample_num}-{idx}.png")

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
