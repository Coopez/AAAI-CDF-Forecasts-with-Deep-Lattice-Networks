# Training loop

import torch
import numpy as np
import time
def test_model(params,
                model,   
                metric,
                metric_plots,
                dataloader_test,
                data_test,
                log_neptune=False,
                neptune_run=None,
                overall_time = [],
                persistence = None
                ):

    plot_ids = sorted(list(set([int(x * (512 / params["batch_size"])) for x in [7,24,14,22,37,8,3]])))
       
    model.eval()
    
    
    metric_dict = {label: [] for label in params['metrics'].keys()}
    metric_array_dict = {label: [] for label in params['array_metrics'].keys()}

    sample_counter = 1

    dict_data = {
        "idx": [],
        "target": [],
        "output": []
    }
    with torch.no_grad():
        forward_times = []
        start_time = time.time()
        for b_idx,batch in enumerate(dataloader_test):
            training_data, target,cs, time_idx = batch

            # pers= persistence.forecast(time_idx[:,0]).unsqueeze(-1)
            pers_denorm = persistence.forecast_raw(time_idx[:,0]).unsqueeze(-1)

            quantile,q_range = data_test.return_quantile(training_data.shape[0],quantile_dim=params["metrics_quantile_dim"])
            start_time = time.time()
            output = forward_pass(params,model,training_data,quantile,quantile_dim=quantile.shape[-1])
            end_time = time.time()
            forward_times.append(end_time - start_time)
            dict_data = {
                "idx": np.append(dict_data["idx"], b_idx),
                "target": np.append(dict_data["target"], target.detach().cpu().numpy()),
                "output": np.append(dict_data["output"], output.detach().cpu().numpy())
            }


            metric_dict= metric(pred = output, truth = target, quantile = quantile, metric_dict=metric_dict,q_range=q_range,pers=pers_denorm)

            
            metric_array_dict = metric_plots.accumulate_array_metrics(metric_array_dict,pred = output, truth = target, quantile = q_range,pers = pers_denorm) #q_range is just the quantiles in a range arrangement. 
            if b_idx in plot_ids and sample_counter != params["valid_plots_sample_size"]+1:
                metric_plots.generate_result_plots(training_data,output, target, quantile, overall_time[time_idx.detach().cpu().numpy()],sample_num = sample_counter, neptune_run=neptune_run)
                sample_counter += 1

    
    metric_plots.generate_metric_plots(metric_array_dict,neptune_run=neptune_run, dataloader_length=len(dataloader_test))
    
    results = metric.summarize_metrics(metric_dict,verbose = False, neptune=False)
    
    array_results = metric_plots._summarize_array_metrics(metric_array_dict)

    return results,np.mean(forward_times),array_results


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

