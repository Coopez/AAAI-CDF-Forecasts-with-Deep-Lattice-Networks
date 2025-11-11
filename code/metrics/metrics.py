
import numpy as np
import torch

from torch.nn import MSELoss, L1Loss

def RSE(pred, true): 
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

@torch.no_grad()
def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

@torch.no_grad()
def MAE(pred, true, return_mean=True, data_scaler=None, return_logits=False):
    if data_scaler is not None: 
        assert len(true.shape) == 2 and true.shape[-1] == 1
        assert len(pred.shape) == 2 and pred.shape[-1] == 1
        true = data_scaler(true)
        pred = data_scaler(pred)
    if torch.is_tensor(pred):
        _logits = (pred - true).abs()
    else:
        _logits = np.abs(pred - true)
    if return_logits:
        return [_logits]
    else:
        if return_mean:
            return torch.mean(_logits)
        else:
            if torch.is_tensor(_logits):
                return np.array([_logits.sum(), _logits.numel()])
            else: 
                return np.array([_logits.sum(), _logits.size])        

@torch.no_grad()
def MSE(pred, true, return_mean=True, data_scaler=None, return_logits=False):
    if data_scaler is not None: 
        assert len(true.shape) == 2 and true.shape[-1] == 1
        assert len(pred.shape) == 2 and pred.shape[-1] == 1
        true = data_scaler(true)
        pred = data_scaler(pred)
    _logits = (pred - true) ** 2
    if return_logits:
        return [_logits]
    else:
        if return_mean:
            return torch.mean(_logits)
        else:
            if torch.is_tensor(_logits):
                return np.array([_logits.sum(), _logits.numel()])
            else: 
                return np.array([_logits.sum(), _logits.size])

@torch.no_grad()
def RMSE(pred, true, return_mean=True):
    return torch.sqrt(MSE(pred, true, return_mean=return_mean))

@torch.no_grad()
def MAPE(pred, true, eps=1e-07, return_mean=True):
    _logits = torch.abs((pred - true) / (true + eps))
    if return_mean:
        return torch.mean(_logits)
    else:
        return _logits

@torch.no_grad()
def MSPE(pred, true, eps=1e-07, return_mean=True):
    _logits = torch.square((pred - true) / (true + eps))
    if return_mean:
        return torch.mean(_logits)
    else:
        return _logits


@torch.no_grad()
def PINAW(pred, truth, intervals=[0.2, 0.5, 0.9], quantiles=None, return_counts=True, return_logits=False,return_array=False):
    if len(truth.shape) == 3: 
        truth = truth.view(-1, truth.shape[-1])
        pred = pred.view(-1, pred.shape[-1])


    assert len(quantiles) % 2 == 1
    quantiles = torch.sort(quantiles)[0]
    intervals = [quantiles[-(i + 1)] - quantiles[i] for i in range(int((len(quantiles) - 1)/2))]
    _scores = {}
    _arrary_scores = []
    _items = []


    for i, interval_i in enumerate(intervals): 
        # if quantiles is not None:
            # quantile prediction. Assumes that the quantiles are in 
            # ascending order and correspond to the intervals
        ci_l = pred[..., i][..., None]
        ci_u = pred[..., -(i+1)][..., None]


        if return_logits:                
            _scores[np.round(interval_i, 5)] =  np.abs(ci_u - ci_l)
        else:        
            if torch.is_tensor(ci_l):
                avg_w = (ci_u - ci_l).abs().mean() 
            else: 
                avg_w = np.mean(np.abs(ci_u - ci_l))
            if return_counts: 
                _scores[np.round(interval_i.item(), 5)] = torch.stack([avg_w, torch.tensor(1.0)]) # 1 instead of range_samples
            elif return_array:
                _arrary_scores.append(avg_w.item() / 1.0) # instead of range_samples.item())
                _items.append(np.round(interval_i.item(), 5))
            else: 
                _scores[np.round(interval_i.item(), 5)] = avg_w.item() / 1.0 #range_samples.item()
    if return_array:
        return _arrary_scores, _items
    return _scores

@torch.no_grad()
def PICP_quantile(pred, truth, intervals=None, quantiles=[0.05,0.125,0.25,0.375,0.45,0.5,0.55,0.625,0.75,0.875,0.95], return_counts=True, loss_type=None, data_scaler=None, return_logits=False, return_array=False):
    if len(truth.shape) == 3: 
        truth = truth.view(-1, truth.shape[-1])
        pred = pred.view(-1, pred.shape[-1])
    
    _scores = {}
    _array_scores = []
    _items = []
    quantiles = torch.sort(quantiles)[0]
    for i, quantile_i in enumerate(quantiles): 
        # if loss_type=="Pinnball":
        ci = pred[..., i][..., None]

        
        if return_logits:                
            _scores[np.round(quantile_i.item(), 5)] = np.concatenate([(truth <= ci)], -1).all(-1).astype(int)[:, None]
            
        else:
            if torch.is_tensor(truth):
                count_correct = torch.cat([ (truth <= ci)], -1).sum() 
            else: 
                count_correct = np.concatenate([(truth <= ci)], -1).all(-1).sum()
            if return_counts: 
                # _scores[np.round(quantile_i.item(), 5)] = np.array([count_correct, truth.shape[0]]) 
                _scores[np.round(quantile_i.item(), 5)] = torch.stack([count_correct, torch.tensor(truth.shape[0]).float()])
            elif return_array:
                _array_scores.append((count_correct.item() / truth.shape[0]))
                _items.append(np.round(quantile_i.item(), 5))
            else: 
                _scores[np.round(quantile_i.item(), 5)] = count_correct.item() / truth.shape[0]
    if return_array:
        return _array_scores, _items
    return _scores

@torch.no_grad()
def PICP(pred, truth, intervals=[0.1,0.25, 0.5, 0.75, 0.9], quantiles=None, return_counts=True, loss_type = None, arbitrary_flag=False, data_scaler=None, return_logits=False, return_array=False):
    if len(truth.shape) == 3: 
        truth = truth.view(-1, truth.shape[-1])
        pred = pred.view(-1, pred.shape[-1])

    if quantiles is not None: 
        assert len(quantiles) % 2 == 1
        quantiles = torch.sort(quantiles)[0]
        intervals = [quantiles[-(i + 1)] - quantiles[i] for i in range(int((len(quantiles) - 1)/2))]
    _scores = {}
    _arrary_scores = []
    _items = []
    for i, interval_i in enumerate(intervals): 
        # if quantiles is not None:
            # quantile prediction. Assumes that the quantiles are in 
            # ascending order and correspond to the intervals
        ci_l = pred[..., i][..., None]
        ci_u = pred[..., -(i+1)][..., None]

        if return_logits:                
            _scores[np.round(interval_i, 5)] = np.concatenate([(truth >= ci_l), (truth <= ci_u)], -1).all(-1).astype(int)[:, None]
            
        else:
            
            count_correct = torch.cat([(truth >= ci_l), (truth <= ci_u)], -1).all(-1).sum().float() 

            if return_counts: 
                _scores[np.round(interval_i.item(), 5)] = torch.stack([count_correct, torch.tensor(truth.shape[0]).float()])
            elif return_array:
                _arrary_scores.append(count_correct.item() / truth.shape[0])
                _items.append(np.round(interval_i.item(), 5))
            else: 
                _scores[np.round(interval_i.item(), 5)] = count_correct.item() / truth.shape[0]
    
    if return_array:
        return _arrary_scores, _items
    return _scores

@torch.no_grad()
def ACE(picp):
    ace = torch.mean(torch.stack([torch.abs(label - (values[0]/values[1])) for label,values in picp.items()]))
    return ace


@torch.no_grad()
def count_CrossOvers(pred,normed=True):
    q_length = pred.shape[-1]
    count = 0
    for q in range(q_length-1): 
       count += torch.sum(torch.where(torch.sign(pred[..., q] - pred[...,q+1])> 0,1.0,0) )
    if normed:
        count = (count / pred.numel()) *100
    return count




import warnings

def pinball_loss(pred, truth, quantiles):
    if not (len(pred.shape) == len(truth.shape) == len(quantiles.shape)):
        warnings.warn('All inputs should have the same number of dimensions')    
    return torch.mean(torch.max((truth - pred) * quantiles, (pred - truth) * (1 - quantiles)))


def approx_crps(pred, truth, quantiles):
    pinball_list = []
    for i in range(quantiles.size()[-1]):
        quantile = quantiles[..., i].unsqueeze(-1)
        pinball_list.append(pinball_loss(pred, truth, quantile).detach().cpu().numpy())
    return np.mean(pinball_list) 



from losses.qr_loss import SQR_loss

class Metrics():
    @torch.no_grad()
    def __init__(self,params,normalizer,data_source):
        self.metrics = params["metrics"].copy()
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
    def __call__(self, pred, truth,quantile, metric_dict,q_range,pers=None):
        results = metric_dict.copy()
        pred_denorm = self.normalizer.inverse_transform(pred,"target")
        truth_denorm = self.normalizer.inverse_transform(truth,"target")


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
            elif metric == "RMSE_Horizon":
                rmse = MSELoss(reduction=None)
                results['RMSE_Horizon'].append(torch.mean(torch.sqrt(rmse(median,truth_denorm)).item()),dim=0)

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
            elif metric == 'ACE_Horizon':
                ace = []
                for i in range(self.horizon):
                    picp = PICP(pred[:, i, :].unsqueeze(1),truth[:, i, :].unsqueeze(1),quantiles=q_range)
                    ace.append(ACE(picp).item())
                results['ACE_Horizon'].append(ace)
            elif metric == 'CRPS':
                results['CRPS'].append(approx_crps(pred_denorm, truth_denorm,quantiles=quantile).item())
            elif metric == 'COV': 
                pass
            elif metric == 'CrossOvers':
                results['CrossOvers'].append(count_CrossOvers(pred_denorm).item())
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
