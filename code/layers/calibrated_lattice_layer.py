"""Imported and adapted from pytorch-lattice"""
from typing import Optional, Union

import torch

from pytorch_lattice.constrained_module import ConstrainedModule
from pytorch_lattice.enums import (
    Interpolation,
    LatticeInit,
    Monotonicity,
    NumericalCalibratorInit,
)
from pytorch_lattice.layers import Lattice#, NumericalCalibrator
from pytorch_lattice.utils.models import (
    calibrate_and_stack,
    initialize_feature_calibrators,
    initialize_monotonicities,
    initialize_output_calibrator,
)
from pytorch_lattice.models.features import CategoricalFeature, NumericalFeature
from layers.rtl import RTL
from layers.quantile_lattice import Constrained_Quantile_Lattice
#from pytorch_lattice.layers import RTL

import numpy as np
import pandas as pd

from layers.calibrated_linear import Linear

from utils.calibrator import NumericalCalibrator

class CalibratedLatticeLayer(ConstrainedModule):
    """PyTorch Calibrated Lattice Model.

    Creates a `torch.nn.Module` representing a calibrated lattice model, which will be
    constructed using the provided model configuration. Note that the model inputs
    should match the order in which they are defined in the `feature_configs`.

    Attributes:
        All: `__init__` arguments.
        calibrators: A dictionary that maps feature names to their calibrators.
        lattice: The `Lattice` layer of the model.
        output_calibrator: The output `NumericalCalibrator` calibration layer. This
            will be `None` if no output calibration is desired.

    Example:

    ```python
    feature_configs = [...]
    calibrated_model = CalibratedLattice(feature_configs, ...)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(calibrated_model.parameters(recurse=True), lr=1e-1)

    dataset = pyl.utils.data.Dataset(...)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    for epoch in range(100):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = calibrated_model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            calibrated_model.apply_constraints()
    ```
    """

    def __init__(
        self,
        monotonicities: list[Monotonicity],
        clip_inputs: bool = True,
        output_min: Optional[float] = None,
        output_max: Optional[float] = None,
        kernel_init: LatticeInit = LatticeInit.LINEAR,
        interpolation: Interpolation = Interpolation.HYPERCUBE,
        output_calibration_num_keypoints: Optional[int] = None,
        output_size: Optional[int] = None,
        input_dim_per_lattice: int = 1,
        num_lattice: int = 1,
        num_keypoints: int = 2,
        device=None,
        quantile_distribution: str = "single",
    ) -> None:
        """Initializes an instance of `CalibratedLattice`.

        Args:
            features: A list of numerical and/or categorical feature configs.
            clip_inputs: Whether to restrict inputs to the bounds of lattice.
            output_min: The minimum output value for the model. If `None`, the minimum
                output value will be unbounded.
            output_max: The maximum output value for the model. If `None`, the maximum
                output value will be unbounded.
            kernel_init: the method of initializing kernel weights. If otherwise
                unspecified, will default to `LatticeInit.LINEAR`.
            interpolation: the method of interpolation in the lattice's forward pass.
                If otherwise unspecified, will default to `Interpolation.HYPERCUBE`.
            output_calibration_num_keypoints: The number of keypoints to use for the
                output calibrator. If `None`, no output calibration will be used.

        Raises:
            ValueError: If any feature configs are not `NUMERICAL` or `CATEGORICAL`.
        """
        super().__init__()

        self.clip_inputs = clip_inputs
        self.output_min = output_min
        self.output_max = output_max
        self.kernel_init = kernel_init
        self.interpolation = interpolation
        self.output_calibration_num_keypoints = output_calibration_num_keypoints
        self.monotonicities = monotonicities
        self.output_size = output_size
        if quantile_distribution == "all":
            self.lattice = Constrained_Quantile_Lattice(
                num_lattices= num_lattice,
                lattice_rank=input_dim_per_lattice,
                lattice_size=num_keypoints,
                monotonicities=self.monotonicities,
                clip_inputs=self.clip_inputs,
                output_min=self.output_min,
                output_max=self.output_max,
                interpolation=interpolation,
                kernel_init=kernel_init,
                
            )
        elif quantile_distribution == "single":
            self.lattice = RTL(
                num_lattices= num_lattice,
                lattice_rank=input_dim_per_lattice,
                lattice_size=num_keypoints,
                monotonicities=self.monotonicities,
                clip_inputs=self.clip_inputs,
                output_min=self.output_min,
                output_max=self.output_max,
                interpolation=interpolation,
                kernel_init=kernel_init,
                
            )
        else:
            raise ValueError("quantile_distribution must be 'all' or 'single'")
        self.output_monotonicties = self.lattice.output_monotonicities()
        if self.output_calibration_num_keypoints is not None:
            output_calibrators = {}
            for i in range(num_lattice):
                output_calibrators[f"calibrator_{i}"] = NumericalCalibrator(
                    input_keypoints=np.linspace(0.0, 1.0, num=self.output_calibration_num_keypoints),
                    missing_input_value=None,
                    output_min=output_min,
                    output_max=output_max,
                    monotonicity=self.output_monotonicties[i],
                    kernel_init=NumericalCalibratorInit.EQUAL_HEIGHTS,
                )
            self.output_calibrator = torch.nn.ModuleDict(output_calibrators)
        # self.output_calibrator = initialize_output_calibrator(
        #     output_calibration_num_keypoints=output_calibration_num_keypoints,
        #     monotonic=not all(m is None for m in self.monotonicities),
        #     output_min=output_min,
        #     output_max=output_max,
        # )
        if self.output_size is not None:
            if quantile_distribution == "all":
                num_lattice = int(len(self.monotonicities) / (input_dim_per_lattice))
            self.layer_output = Linear(num_lattice, self.output_size, self.output_monotonicties)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs an input through the network to produce a calibrated lattice output.

        Args:
            x: The input tensor of feature values of shape `(batch_size, num_features)`.

        Returns:
            torch.Tensor of shape `(batch_size, 1)` containing the model output result.
        """
        #result = calibrate_and_stack(x, self.calibrators)
        result = self.lattice(x)
        
        if self.output_calibration_num_keypoints is not None:
            result= calibrate_and_stack(result, self.output_calibrator)
            #result = self.output_calibrator(result)
        if self.output_size is not None:
            result = self.layer_output(result)
        return result

    @torch.no_grad()
    def apply_constraints(self) -> None:
        """Constrains the model into desired constraints specified by the config."""
        
        self.lattice.apply_constraints()
        if self.output_calibration_num_keypoints is not None:
            for output_calibrator in self.output_calibrator.values():
                output_calibrator.apply_constraints()
        if self.output_size is not None:
            self.layer_output.apply_constraints()


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

        # for name, calibrator in self.calibrators.items():
        #     calibrator_messages = calibrator.assert_constraints(eps)
        #     if calibrator_messages:
        #         messages[f"{name}_calibrator"] = calibrator_messages
        lattice_messages = self.lattice.assert_constraints(eps)
        if lattice_messages:
            messages["lattice"] = lattice_messages
        # if self.output_calibrator:
        #     output_calibrator_messages = self.output_calibrator.assert_constraints(eps)
        #     if output_calibrator_messages:
        #         messages["output_calibrator"] = output_calibrator_messages
        for name, output_calibrator in self.output_calibrator.items():
            output_calibrator_messages = output_calibrator.assert_constraints(eps)
            if output_calibrator_messages:
                messages[f"{name}_output_calibrator"] = output_calibrator_messages
        if self.output_size is not None:
            layer_output_messages = self.layer_output.assert_constraints(eps)
            if layer_output_messages:
                messages["layer_output"] = layer_output_messages
        return messages
    




    
