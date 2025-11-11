from layers.calibrated_lattice_layer import CalibratedLatticeLayer
import torch
import torch.nn as nn

from typing import Optional, Union
from pytorch_lattice.models.features import NumericalFeature, CategoricalFeature
from layers.calibrated_linear import Linear
from typing import Optional, Union

from pytorch_lattice.enums import (
    Interpolation,
    LatticeInit,
)

from pytorch_lattice.utils.models import (
    calibrate_and_stack,
    initialize_monotonicities,
)
from utils.calibrator import initialize_feature_calibrators
# import time

class CalibratedLatticeModel(nn.Module):
    """
    A PyTorch module that implements a calibrated lattice model with multiple layers.
    Args:
        features (list[Union[NumericalFeature, CategoricalFeature]]): List of features used in the model.
        clip_inputs (bool, optional): Whether to clip inputs to the lattice. Defaults to True.
        output_min (Optional[float], optional): Minimum value for the output. Defaults to None.
        output_max (Optional[float], optional): Maximum value for the output. Defaults to None.
        kernel_init (LatticeInit, optional): Initialization method for the lattice kernel. Defaults to LatticeInit.LINEAR.
        interpolation (Interpolation, optional): Interpolation method for the lattice. Defaults to Interpolation.HYPERCUBE.
        output_calibration_num_keypoints (Optional[int], optional): Number of keypoints for output calibration. Defaults to None.
        lattice_type (str, optional): Type of lattice to use. Defaults to 'lattice'.
        num_layers (int, optional): Number of layers in the model. Defaults to 1.
        output_size (int, optional): Size of the output. Defaults to 1.
    Attributes:
        lattice_layers (nn.ModuleList): List of lattice layers in the model.
    Methods:
        forward(x: torch.Tensor) -> torch.Tensor:
            Forward pass through the model.
    """

    def __init__(
        self,
        features: list[Union[NumericalFeature, CategoricalFeature]],
        clip_inputs: bool = True,
        output_min: Optional[float] = 0,
        output_max: Optional[float] = 1,
        kernel_init: LatticeInit = LatticeInit.LINEAR,
        interpolation: Interpolation = Interpolation.HYPERCUBE,
        num_layers: int = 1,
        input_dim_per_lattice: list = [],
        output_size: int = 1,
        lattice_keypoints: int = 2,
        output_calibration_num_keypoints: Optional[int] = None,
        model_type: str = 'lattice_linear',
        input_dim: int = 23,
        downsampled_input_dim: int = 13,
        device=None,
        quantile_distribution: str = "single",
    ) -> None:
        super().__init__()
        self.features = features
        # grabbed from layers/calibrated_lattice_layer.py
        self.monotonicities = initialize_monotonicities(features)
        self.calibrators = initialize_feature_calibrators(
            features=features,
            output_min=0,
            output_max=[feature.lattice_size - 1 for feature in features],
        )
        
        num_lattice_per_layer = []
        input_development = input_dim
        for _ in range(num_layers):
            divide = -(-input_development // input_dim_per_lattice)
            num_lattice_per_layer.append(divide)
            input_development = divide
        
        self.input_dim_per_lattice = input_dim_per_lattice
        self.layer_dims = num_lattice_per_layer
        self.output_size = output_size
        self.parallel_lattices = nn.ModuleList()
        self.lattice_layers = nn.Sequential()
        
        
        if model_type == 'linear_lattice':
            self.lattice_layers.append(
                Linear(
                    input_dim=input_dim,
                    output_dim=downsampled_input_dim,
                    monotonicities=self.monotonicities,
                    use_bias=True,
                    weighted_average=False,
                ))
            self.monotonicities = ['increasing' for _ in range(downsampled_input_dim)]

        # Construct lattice layers
        for i in range(num_layers-1):
            # need to stack layers
            lattice_layer = CalibratedLatticeLayer(
                monotonicities = self.monotonicities,
                clip_inputs=clip_inputs,
                output_min=output_min,
                output_max=output_max,
                kernel_init=kernel_init,
                interpolation=interpolation,
                input_dim_per_lattice = self.input_dim_per_lattice,
                num_lattice= self.layer_dims[i],
                output_calibration_num_keypoints=output_calibration_num_keypoints,
                num_keypoints=lattice_keypoints,
                device=device,
                quantile_distribution=quantile_distribution,
            )
            
            self.lattice_layers.append(lattice_layer)
            self.monotonicities = lattice_layer.lattice.output_monotonicities()
        # Last Lattice Layer
        if model_type == 'lattice' or model_type == 'linear_lattice':
            linear_output_size = None # if this is not none, will create a linear out.
            num_lattice_last = self.output_size
            input_dim_last = -(self.layer_dims[-1]//- 2) # ceiling division
        if model_type == 'lattice_linear':
            linear_output_size = self.output_size
            num_lattice_last = self.layer_dims[-1]
            input_dim_last = self.input_dim_per_lattice
        if model_type == "lattice" or model_type == "lattice_linear":
            self.lattice_layers.append(
                CalibratedLatticeLayer(
                    monotonicities = self.monotonicities,
                    clip_inputs=clip_inputs,
                    output_min=output_min,
                    output_max=output_max,
                    kernel_init=kernel_init,
                    interpolation=interpolation,
                    input_dim_per_lattice = input_dim_last,
                    num_lattice= num_lattice_last,
                    output_calibration_num_keypoints=output_calibration_num_keypoints,
                    num_keypoints=lattice_keypoints,
                    # output specific parameters
                    output_size=linear_output_size,
                    device=device,
                    quantile_distribution=quantile_distribution,
                )
            )
        # self.lattice = nn.Sequential()
        # for layer in self.lattice_layers:
        #     self.lattice.append(layer)
        # #self.lattice_layers = nn.Sequential(self.lattice_layers) #nn.ModuleList(self.lattice_layers)
        #print(self.lattice_layers)
        sum(p.numel() for p in self.lattice_layers.parameters())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the calibrated lattice model.
        Args:
            x (torch.Tensor): Input tensor to the model.
        Returns:
            torch.Tensor: Output tensor after passing through all lattice layers.
        """
        x = calibrate_and_stack(x, self.calibrators)
        # x = x.to(torch.float32)
        x = self.lattice_layers(x) 
        return x
    
    @torch.no_grad()
    def apply_constraints(self) -> None:
        """Constrains the model into desired constraints specified by the config."""
        # start_time = time.time()
        for calibrator in self.calibrators.values():
            calibrator.apply_constraints()
        for lattice in self.lattice_layers:
            lattice.apply_constraints()
        # elapsed_time = time.time() - start_time
        # print(f"apply_constraints took {elapsed_time:.4f} seconds")
        
    @torch.no_grad()
    def assert_constraints(self, eps: float = 1e-6) -> dict[str, list[str]]:
        """Asserts all layers within model satisfied specified constraints.

        Asserts monotonicity pairs and output bounds for categorical calibrators,
        monotonicity and output bounds for numerical calibrators, and monotonicity and
        weights summing to 1 if weighted_average for linear layer.

        Args:
            eps: the margin of error allowed

        Returns:
            A dict where key is feature_name for calibrators and 'linear' for the linear
            layer, and value is the error messages for each layer. Layers with no error
            messages are not present in the dictionary.
        """
        messages = {}

        for name, calibrator in self.calibrators.items():
            calibrator_messages = calibrator.assert_constraints(eps)
            if calibrator_messages:
                messages[f"{name}_calibrator"] = calibrator_messages
        
        for lattice in self.lattice_layers:
            lattice_messages = lattice.assert_constraints(eps)
            if lattice_messages:
                messages["lattice"] = lattice_messages
        # if self.output_calibrator:
        #     output_calibrator_messages = self.output_calibrator.assert_constraints(eps)
        #     if output_calibrator_messages:
        #         messages["output_calibrator"] = output_calibrator_messages

        return messages