## Run

There are 5 `run_` files with different purposes. `config` is used to configure settings and hyperparameters for `run_` files.
After setting up the environment, any of these files are can be run to replicate our findings. 
The section Run Files explains in more detail how each file works, and what its expected output is.

## Environment Set-up


Required packages can be found in the `env.txt` files. Note that pytorch-lattice only supports pip, even when using conda.

The required packages as installed on a linux-based system are:
Noteworthy here, the latest pytorch-lattice supports at most `python=3.11`. The rest may depend on the version of python.
For pytorch, be sure to follow the official installation instructions for your CUDA if you want GPU support (https://pytorch.org/get-started/locally/).
```plaintext
numpy                             1.25.1
pandas                            2.0.3
matplotlib                        3.10.0
seaborn                           0.13.2
python                            3.11.3
Pytorch                           2.1.2
optuna                            4.2.1
pytorch-lattice                   0.2.0
neptune                           1.13.0
neptune-optuna                    1.4.1
```
## Run-Files
`config.py` is used to configure all run files. Refer to descriptions in config to see what each setting does. 
In addition to metric printouts, training runs will produce plots every epoch which are saved in `plots_save`.
### run_train_simple
This starts a training run for SQR:LSTM-Linear, SQR:LSTM-NN, SQR:LSTM-SMNN, or SQR:LSTM-DLN. In `config.py`, `output_model` is used to determine which model to use.   
### run_test_simple
This starts a testing run for SQR:LSTM-Linear, SQR:LSTM-NN, SQR:LSTM-SMNN, or SQR:LSTM-DLN. In `config.py`, `test_model_name` is used to determine which model to use. All test models are loaded from the same models used to generate the results in the paper. Note that only 1 the 5 different versions are included here and DLNs are not included at all. This is because they are too big to fit into the 50MB limit set by OpenReview. They will be available in the published version. 
### run_train_comp
This starts a training run for LSTM-PP, or LSTM-QR. In `config.py`, `comp_model` is used to determine which model to use.
### run_test_comp
This starts a testing run for Smart Persistence, LSTM-PP, or LSTM-QR. In `config.py`, `comp_model` is used to determine which model to use.
### run_train_hpo
This will run gridsearch on a specified model. 
It is more involved than the other run files, as the packages optuna and neptune have been used to log, plot and save result for easy comparison. 
You need `optuna, neptune, neptune-optuna` installed. 
We highly recommend to become familiar how both packages work before trying to configure and run a hyperparameter optimization. Generally, search space range can be set in `config.py` while the search space itself has to be configured in `objective()` and `main`. 
Neptune needs a user account, which is freely available to students and researchers at https://neptune.ai/. (Others may want delete all neptune from the code, however some optuna configuration will need adjustment in `main`)

