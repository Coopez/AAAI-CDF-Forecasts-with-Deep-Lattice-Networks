import torch
from pytorch_lattice.constrained_module import ConstrainedModule
from layers.calibrated_linear import Linear
from typing import Optional
from pytorch_lattice.enums import (
    Monotonicity,
)

class Constrained_Linear(ConstrainedModule):
    def __init__(self, 
                input_dim: int,
                output_dim: int, 
                quantile_idx: int = -1,
                use_bias: bool = True, 
                weighted_average: bool = False):
        super().__init__()

        # need to compute monotonocity here
        monotonicities = [None for _ in range(input_dim)]
        monotonicities[quantile_idx] = Monotonicity.INCREASING
        self.linear = Linear(input_dim, output_dim, monotonicities, use_bias, weighted_average)

    def forward(self, x):
        return self.linear(x)