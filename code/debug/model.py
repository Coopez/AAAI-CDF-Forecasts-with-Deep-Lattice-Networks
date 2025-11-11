# print model parameters, their names and shapes and total number of parameters

def print_model_parameters(model):
    print(model)
    print("Model's state_dict:")
    total_params = 0
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        total_params += model.state_dict()[param_tensor].numel()
    print("Total number of parameters: ", total_params)