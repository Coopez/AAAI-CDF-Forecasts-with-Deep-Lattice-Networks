
_DATA_DESCRIPTION = "Station 11 Irradiance Sunpoint" #"IFE Skycam"# Description of the data set 

# Hyperparameters
params = dict(

debug = False, # Determines some debug outputs
batch_size = 64 , # Batchsize 
random_seed = [0], # Random seeds
model_save = False, # Determines if model is saved - needs to be set True if you want to subsequently test the same model

train_shuffle = True, # Determines if data is shuffled
valid_shuffle = False, # Determines if data is shuffled
dataloader_device = 'cpu', # determines if data is loaded on cpu or gpu

learning_rate = 0.01, #0.1, # Learning rate
epochs = 300, # Number of epochs

window_size = 96, # Lookback size
horizon_size = 36, # Horizon size

loss = 'pinball_loss', 
optimizer = 'Adam',


input_model = "lstm",
output_model = "dnn", 
#options = "lattice_linear" (this is DLN), "dnn" (this is NN), "linear", "constrained_linear"
test_model_name = "linear", # Name of the model to be tested
comp_model = "pp",# "pp" , qr for quantile regression, pp for point prediction, sp for smart persistence


scheduler_min_lr = 0.000001, # Minimum learning rate for scheduler
scheduler_patience = 0, # Patience for scheduler
scheduler_factor = 0.1, # Factor for scheduler
scheduler_enable = True, # Determines if scheduler is enabled

step_scheduler_enable = True,
step_scheduler_milestones = [1], # Milestones for step scheduler
step_scheduler_gamma = 0.1, # Gamma for step scheduler

early_stopping_patience = 150, #How many epochs to wait for early stopping
early_stopping_tolerance = 0.01, # At which change in validation loss to stop training

# LSTM Hyperparameters
lstm_input_size = 246,
lstm_hidden_size = [128,128], # LIST of number of nodes in hidden layers will run into error if layers of different sizes. This is because hidden activation
lstm_num_layers = 2, # Number of layers


# Lattice Hyperparameters
lattice_num_layers = 1, # Number of layers
lattice_dim_input = 2, # input dim per lattice
lattice_num_keypoints = 21, # Number of keypoints
lattice_calibration_num_keypoints = 61, # Number of keypoints in calibration layer
lattice_calibration_num_keypoints_quantile = 11, # Number of layers in calibration layer
lattice_donwsampled_dim = 13, # Dimension of downsampled input when using linear_lattice
lattice_quantile = "all", # "all" or "single", determines in how many lattices quantiles are injected
lattice_out_calibration_num_keypoints = 21, # Number of keypoints in output calibration layer

# SMNN Hyperparameters
smnn_exp = (256,512),
smnn_relu = (256,512), 
smnn_conf = (512,256), 


metrics =  {"ACE": [], 
            "MAE": [], 
            "RMSE": [],
            "CRPS": [],
            "SS": [],
            "CrossOvers": [],},
            
array_metrics = {"PICP": [],
            "Cali_PICP": []},  # PICP = Prediction Interval Coverage Probability, cali_PICP are the quantile based PICP values


metrics_quantile_dim = 11, # how many quantiles are used for metrics.

valid_metrics_every = 1, # Determines how often metrics are calculated depending on Epoch number
valid_plots_every = 1, # Determines how often plots are calculated depending on Validation and epoch number
valid_plots_sample_size = 7, # Sample size for plots - will run into error if larger than the batch size adjusted index list.

valid_clamp_output = True, # Determines if output is clamped to 0

save_all_epochs = False, # Determines if all epochs are saved


hpo_lr = [0.00001,0.0001,0.001], # Hyperparameter optimization search space for learning rate
#hpo_batch_size = [64,256,1024], # Hyperparameter optimization search space for batch size
#hpo_window_size = [60,90,120,180], # Hyperparameter optimization search space for window size
hpo_hidden_size = [16,32,64,128], # Hyperparameter optimization search space for hidden size
hpo_num_layers = [2,3,4], # Hyperparameter optimization search space for number of layers

hpo_lattice_dim_input = [1, 2], # Hyperparameter optimization search space for number of input dimensions
hpo_lattice_keypoints = [11,21,51], # Hyperparameter optimization search space for number of keypoints
hpo_calibration_keypoints = [51,61,71], # Hyperparameter optimization search space for number of calibration keypoints
hpo_calibration_keypoints_quantile = [21,31,51], # Hyperparameter optimization search space for number of calibration keypoints


hpo_smnn_exp_1 = [512,1024],
hpo_smnn_exp_2 = [256,1024], # Hyperparameter optimization search space for SMNN exp units
hpo_smnn_relu_1 = [256,1024],
hpo_smnn_relu_2 = [512], # Hyperparameter optimization search space for SMNN relu units
hpo_smnn_conf_1 = [128,256,512],
hpo_smnn_conf_2 = [256,512], # Hyperparameter optimization search space for SMNN confluence units





# pathing options. Ideally, these should not be changed.
save_path_model_epoch = "models_save/", # Path for saving models
valid_plots_save_path = "plots_save/", # Path for saving plots
test_model_path = "models_save/model.pth", # Path for loading model
data_save_path = "data_save", # Path for saving data

# legacy options. This should be ignored.
target = 'GHI', # legacy option
target_summary = 1, # legacy option
neptune_tags = ["IFE Skyimage","HPO"], # List of tags for neptune
# Extra Loss Hyperparameters
loss_calibration_lambda = 0.0, # Lambda for beyond loss
loss_calibration_scale_sharpness = True, # Determines if sharpness is scaled by quantile
dnn_input_size = 246,  # input will be that * window_size
dnn_hidden_size = [32,32], # LIST of number of nodes in hidden layers
dnn_num_layers = 2, # Number of layers
dnn_activation = 'relu', # Activation function
constant_quantile = False,
deterministic_optimization= False, # Determines if optimization is deterministic
inject_persistence = False, # Determines if persistence model is injected
)



