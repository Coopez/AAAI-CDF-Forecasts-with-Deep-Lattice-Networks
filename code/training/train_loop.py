# Training loop

import torch
import numpy as np
from debug.model import print_model_parameters
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR
import os

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
                log_neptune=False,
                neptune_run=None,
                overall_time = [],
                persistence = None
                ):
    if params['debug']:
        print_model_parameters(model)

    run_name = 'local_test_run'

    epochs = params['epochs']
    if params['deterministic_optimization']:
        epochs = 1
    plateau_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=params["scheduler_patience"], factor=params["scheduler_factor"], min_lr=params["scheduler_min_lr"])
    step_scheduler = MultiStepLR(optimizer, milestones=params["step_scheduler_milestones"], gamma=params["step_scheduler_gamma"])
    plot_ids = sorted(list(set([int(x * (512 / params["batch_size"])) for x in [7,24,14,22,37,8,3]])))
    epoch_path = params['save_path_model_epoch']
    early_stopping = ModelCheckpointer(path=epoch_path, tolerance=params['early_stopping_tolerance'], patience=params['early_stopping_patience'])
    for epoch in range(epochs):
        start_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        end_time = torch.cuda.Event(enable_timing=True)
        train_losses = []
        sharp_losses = []
        model.train()
        for batch in dataloader:
            training_data, target, _, time_idx = batch

            quantile = data.return_quantile(training_data.shape[0],quantile_dim=2,constant=params['constant_quantile'])
                
            output = forward_pass(params,model,training_data,quantile,quantile_dim=quantile.shape[-1])
            # Compute loss
            loss = criterion(output, target, quantile) 
            #Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if params["output_model"] == "lattice_linear" or params["output_model"] == "constrained_linear":
                model[1].apply_constraints()
            train_losses.append(loss.item())

    
        
        save_model_per_epoch(run_name, model, epoch_path, epoch, save_all=params['save_all_epochs'])

        model.eval()
        
        if epoch % params['valid_metrics_every'] == 0:

            metric_dict = {label: [] for label in params['metrics'].keys()}
            metric_array_dict = {label: [] for label in params['array_metrics'].keys()}

            sample_counter = 1

            with torch.no_grad():
                # batch_var = []
                for b_idx,batch in enumerate(dataloader_valid):
                    training_data, target,_, time_idx = batch

                    pers_denorm = persistence.forecast_raw(time_idx[:,0]).unsqueeze(-1)
     
                    quantile,q_range = data_valid.return_quantile(training_data.shape[0],quantile_dim=params["metrics_quantile_dim"])

                    output = forward_pass(params,model,training_data,quantile,quantile_dim=quantile.shape[-1])


                    metric_dict= metric(pred = output, truth = target, quantile = quantile, metric_dict=metric_dict,q_range=q_range,pers=pers_denorm)

                    if epoch % params['valid_plots_every'] == 0:
                        metric_array_dict = metric_plots.accumulate_array_metrics(metric_array_dict,pred = output, truth = target, quantile = q_range,pers = pers_denorm) #q_range is just the quantiles in a range arrangement. 
                        if b_idx in plot_ids and sample_counter != params["valid_plots_sample_size"]+1:
                            metric_plots.generate_result_plots(training_data,output, target, quantile, overall_time[time_idx.detach().cpu().numpy()],sample_num = sample_counter, neptune_run=neptune_run)
                            sample_counter += 1
            # print(np.mean(pers_losses))
            if epoch % params['valid_plots_every'] == 0:
                metric_plots.generate_metric_plots(metric_array_dict,neptune_run=neptune_run, dataloader_length=len(dataloader_valid))

            
        
        end_time.record()
        torch.cuda.synchronize() 
        epoch_time = start_time.elapsed_time(end_time)/ 1000 # is in ms. Need to convert to seconds
        step_meta = {"Epoch": f"{epoch+1:02d}/{epochs}", "Time": epoch_time, "Train_Loss": np.mean(train_losses)}
        scheduler_metrics = metric.summarize_metrics({**step_meta, **metric_dict},neptune = log_neptune,neptune_run=neptune_run)
        if params["scheduler_enable"]:
            plateau_scheduler.step(scheduler_metrics["CRPS"])
        if params["step_scheduler_enable"]:
            step_scheduler.step()
 
        break_condition, model = early_stopping(model, scheduler_metrics["CRPS"])
        if break_condition is False:
            break
    return model


def forward_pass(params:dict,
                 model: torch.nn.Module,
                 batch:torch.Tensor, 
                 quantile: torch.Tensor,
                 quantile_dim:int,
                 persistence=None,
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
        if persistence is not None:
            output_i = model[1](aggregated_input)
            persistence_output = model[2](x = output_i,c = persistence.squeeze(),tau = quantile[0,0,i], x_input =aggregated_input)
            output[...,i] = persistence_output
        else:
            output[...,i] = model[1](aggregated_input)
    return output


def save_model_per_epoch(run_name, model: torch.nn.Module, path:str, epoch:int, save_all:bool=False):

    parent_dir = os.path.dirname(os.path.abspath(__file__))

    parts = parent_dir.split(os.sep)
    if 'code' in parts:
        code_index = parts.index('code')
        parent_dir = os.sep.join(parts[:code_index])
        code_dir = os.sep.join(parts[:code_index+1])
    else:
        # fallback: assume current directory is 'code'
        parent_dir = code_dir

    if save_all:
        save_path = os.path.join(parent_dir, f"{run_name}_epoch_{epoch}.pt")
        torch.save(model.state_dict(), save_path)
    else:
        # Save to the parent directory of the given path
        
        save_path = os.path.join(parent_dir, f"{run_name}.pt")
        torch.save(model.state_dict(), save_path)
    
class ModelCheckpointer():
    def __init__(self, path:str,tolerance:float=0.0001, patience:int=5):
        self.path = path
        self.last_metric = 9999.0
        self.counter = 0
        self.tolerance = tolerance
        self.patience = patience
        self.wd = os.getcwd()
        if os.path.basename(self.wd) == "code":
            self.path = os.path.join(self.wd, self.path)
        else:
            self.path = os.path.join(self.wd, "code", self.path)
        
    def __call__(self,  model: torch.nn.Module, metric):
        """
        Checks if the metric has improved and saves the model if it has.
        If the metric has not improved for a certain number of epochs, it stops training.
        """
        if metric < (self.last_metric - self.tolerance):
            self._save(model)
            self.last_metric = metric
            self.counter = 0
            return True, model
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Stopping training after {self.counter} epochs without improvement.")
                model = self._load(model)
                return False, model
        return True, model
    def _save(self, model: torch.nn.Module):
        torch.save(model.state_dict(), self.path+"checkpoint_model.pt")
    def _load(self,model: torch.nn.Module):
        model.load_state_dict(torch.load(self.path+"checkpoint_model.pt"))
        return model

