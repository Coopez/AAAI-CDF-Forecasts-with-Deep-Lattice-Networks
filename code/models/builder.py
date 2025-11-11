import torch
from torch.nn import Sequential, Linear
from .LSTM import LSTM
from .constrained_linear import Constrained_Linear
from .Calibrated_lattice_model import CalibratedLatticeModel
from .DNN_out_model import Neural_Net_with_Quantile
from .SMNN import ScalableMonotonicNeuralNetwork
def build_model(params, device, features=None) -> Sequential:
    
    if params["input_model"] == "lstm":
        input_model = LSTM(input_size= params["lstm_input_size"],
                           hidden_size= params["lstm_hidden_size"],
                           num_layers= params["lstm_num_layers"],
                           window_size= params["window_size"]
                           )
        data_output_size = params["lstm_hidden_size"][-1]

    else:
        raise ValueError("Input_Model not implemented")
    
    #options = "lattice", "linear", "constrained_linear", "linear_lattice", "lattice_linear"
    if params["output_model"] == "linear":
        output_model = Linear(in_features= data_output_size+1,
                                       out_features= params["horizon_size"])
    elif params["output_model"] == "constrained_linear":
        output_model = Constrained_Linear(input_dim= data_output_size+1,
                                       output_dim= params["horizon_size"],
                                       quantile_idx= -1) # assuming quantile is the last feature
    elif params["output_model"] == "dnn":
        output_model = Neural_Net_with_Quantile(input_size= data_output_size+1,
                                       output_size= params["horizon_size"])
    elif params["output_model"] == "lattice_linear":
        assert features is not None, "Features must be provided for lattice model"
        output_model = CalibratedLatticeModel( features= features,
                                        output_min= 0,
                                        output_max= 1,
                                        num_layers= params["lattice_num_layers"],
                                        input_dim_per_lattice= params["lattice_dim_input"],
                                        output_size= params["horizon_size"],
                                        lattice_keypoints= params["lattice_num_keypoints"],
                                        output_calibration_num_keypoints= params["lattice_out_calibration_num_keypoints"],
                                        model_type= params["output_model"],
                                        input_dim= data_output_size,
                                        downsampled_input_dim= params["lattice_donwsampled_dim"],
                                        device= device,
                                        quantile_distribution= params["lattice_quantile"]
        )
        
    elif params["output_model"] == "smnn":
        output_model = ScalableMonotonicNeuralNetwork(
            input_size=data_output_size+1,
            mono_size = 1,
            output_size=params["horizon_size"],
            exp_unit_size = params["smnn_exp"],
            relu_unit_size = params["smnn_relu"],
            conf_unit_size = params["smnn_conf"],
                   )
        
    else:
        raise ValueError("Output_Model not implemented")
 
   
    model = torch.nn.ModuleList( 
            [input_model,
            output_model
            ]
        )
    return model


def build_optimizer(params,model):
    if params['deterministic_optimization']:
        from pytorch_minimize.optim import MinimizeWrapper
        minimizer_args = dict(method='SLSQP', options={'disp':True, 'maxiter':100}) # supports a range of methods
        optimizer = MinimizeWrapper(model.parameters(), minimizer_args)
    else:
        optimizer_class = getattr(torch.optim, params['optimizer'])
        optimizer = optimizer_class(model.parameters(), lr=params['learning_rate'])
    return optimizer

