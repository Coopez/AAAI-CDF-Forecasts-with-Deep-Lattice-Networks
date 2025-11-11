import torch
import warnings


class SQR_loss():
    def __init__(self,type='pinball_loss',lambda_=0.5,scale_sharpness=False,huber_threshold=0.01):
        self.singular_loss = True # if True only one quantile is used for pinball loss calculation 
        self.type = type
        self.lambda_ = lambda_ # lambda for calibration_sharpness_loss, determining weight of sharpness component
        self.scale_sharpness = scale_sharpness # scale sharpness component by quantile
        self.huber_threshold = huber_threshold 
        self.huber_norm = torch.nn.HuberLoss(delta=self.huber_threshold,reduction='none')
    def __call__(self,y_pred,y_true,quantile):
        if quantile.shape[1] > y_pred.shape[1]:
            quantile = quantile[:, :y_pred.shape[1], :]  
        elif quantile.shape[1] < y_pred.shape[1]:
            quantile = quantile.repeat(1, y_pred.shape[1] // quantile.shape[1], 1)
        else:
            pass
        if not (len(y_pred.shape) == len(y_true.shape) == len(quantile.shape)):
            warnings.warn('All inputs should have the same number of dimensions')
        if self.type == 'pinball_loss': 
            if quantile.shape[-1] == 1:
                loss = torch.mean(torch.max(torch.mul(quantile,(y_true-y_pred)),torch.mul((quantile-1),(y_true-y_pred))))
                return loss
            elif quantile.shape[-1] == 2 and self.singular_loss==True:
                quantile = quantile[...,0].unsqueeze(-1)
                y_pred = y_pred[...,0].unsqueeze(-1)
                return torch.mean(torch.max(torch.mul(quantile,(y_true-y_pred)),torch.mul((quantile-1),(y_true-y_pred))))
            else:
                losses = []
                y_true = y_true.squeeze(-1)
                for i in range(0,quantile.shape[-1]):
                    losses.append(torch.mean(torch.max(torch.mul(quantile[...,i],(y_true-y_pred[...,i])),torch.mul((quantile[...,i]-1),(y_true-y_pred[...,i])))))
                return torch.mean(torch.stack(losses))
        elif self.type == 'huber_pinball_loss':
            if quantile.shape[-1] == 1:
                huber = self.huber_norm(input=y_pred, target=y_true)
                error = y_true - y_pred
                loss = torch.mean(
                    torch.where(error > 0, huber *quantile, huber *(1- quantile))
                )
                return loss
            elif quantile.shape[-1] == 2 and self.singular_loss==True:
                quantile = quantile[...,0].unsqueeze(-1)
                y_pred = y_pred[...,0].unsqueeze(-1)
                huber = self.huber_norm(input=y_pred, target=y_true)
                error = y_true - y_pred

                loss = torch.mean(
                    torch.where(error > 0, huber *quantile, huber *(1- quantile))
                )
                return loss
            else:
                losses = []
                y_true = y_true.squeeze(-1)
                for i in range(0,quantile.shape[-1]):
                    huber = self.huber_norm(input=y_pred[...,i], target=y_true)
                    error = y_true - y_pred[...,i]

                    losses.append(torch.mean(
                    torch.where(error > 0, huber *quantile[...,i], huber *(1- quantile[...,i]))
                    ))
        
                return torch.mean(torch.stack(losses))
       
        else:
            raise ValueError('Unknown loss type')
        return None
    
