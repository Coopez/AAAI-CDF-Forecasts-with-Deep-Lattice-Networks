"""Imported and adapted from pytorch-lattice"""

"""A PyTorch module implementing a calibrated modeling layer for Random Tiny Lattices.

This module implements an ensemble of tiny lattices that each operate on a subset of the
inputs. It utilizes the multi-unit functionality of the Lattice module to better
optimize speed performance by putting feature subsets that have the same constraint
structure into the same Lattice module as multiple units.
"""
import logging
from typing import Optional, Union
from pytorch_lattice.constrained_module import ConstrainedModule
import numpy as np
import torch

from pytorch_lattice.enums import Interpolation, LatticeInit, Monotonicity
from .lattice import Lattice


class Constrained_Quantile_Lattice(ConstrainedModule):
    

    def __init__(
        self,
        monotonicities: list[Monotonicity],
        num_lattices: int,
        lattice_rank: int,
        lattice_size: int = 2,
        output_min: Optional[float] = None,
        output_max: Optional[float] = None,
        kernel_init: LatticeInit = LatticeInit.LINEAR,
        clip_inputs: bool = True,
        interpolation: Interpolation = Interpolation.HYPERCUBE,
        average_outputs: bool = False,
        random_seed: int = 42,
    ) -> None:
        
        super().__init__()

        
        self.monotonicities = monotonicities
        self.num_lattices = num_lattices
        self.lattice_rank = lattice_rank + 1
        self.lattice_size = lattice_size
        self.output_min = output_min
        self.output_max = output_max
        self.kernel_init = kernel_init
        self.clip_inputs = clip_inputs
        self.interpolation = interpolation
        self.average_outputs = average_outputs
        self.random_seed = random_seed
        # update lattice rank to exclude quantile at first
        rtl_indices = np.array(
            [i % (len(self.monotonicities)-1) for i in range(int((len(self.monotonicities)) / lattice_rank)*lattice_rank )]
        )
        # added + 1 to length to add more q and -1 * 2 to len of monotonicities to get as many quantiles as features    
        
        rtl_indices = self._insert_quantiles(rtl_indices, lattice_rank, len(self.monotonicities)-1)
        self.num_lattices = int(len(rtl_indices) / self.lattice_rank)
        split_rtl_indices = [list(arr) for arr in np.split(rtl_indices, self.num_lattices)]
        swapped_rtl_indices = self._ensure_unique_sublattices(split_rtl_indices)
        monotonicity_groupings = {}
        for lattice_indices in swapped_rtl_indices:
            monotonic_count = sum(
                1
                for idx in lattice_indices
                if self.monotonicities[idx] == Monotonicity.INCREASING
            )
            if monotonic_count not in monotonicity_groupings:
                monotonicity_groupings[monotonic_count] = [lattice_indices]
            else:
                monotonicity_groupings[monotonic_count].append(lattice_indices)
        for monotonic_count, groups in monotonicity_groupings.items():
            for i, lattice_indices in enumerate(groups):
                sorted_indices = sorted(
                    lattice_indices,
                    key=lambda x: (self.monotonicities[x] is None),
                    reverse=False,
                )
                groups[i] = sorted_indices

        self._lattice_layers = {}
        self._lattice_index_map = {}
        for monotonic_count, groups in monotonicity_groupings.items():
            self._lattice_layers[str(monotonic_count)] = (
                Lattice(
                    lattice_sizes=[self.lattice_size] * self.lattice_rank,
                    output_min=self.output_min,
                    output_max=self.output_max,
                    kernel_init=self.kernel_init,
                    monotonicities=[Monotonicity.INCREASING] * monotonic_count
                    + [None] * (self.lattice_rank - monotonic_count),
                    clip_inputs=self.clip_inputs,
                    interpolation=self.interpolation,
                    units=len(groups),
                ))
            self._lattice_index_map[str(monotonic_count)] = groups    

        self._lattice_layers =  torch.nn.ModuleDict(self._lattice_layers) # make it a module dict, so maybe layers are registered correctly

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method computed by using forward methods of each lattice in ensemble.

        Args:
            x: input tensor of feature values with shape `(batch_size, num_features)`.

        Returns:
            `torch.Tensor` containing the outputs of each lattice within RTL structure.
            If `average_outputs == True`, then all outputs are averaged into a tensor of
            shape `(batch_size, 1)`. If `average_outputs == False`, shape of tensor is
            `(batch_size, num_lattices)`.
        """
        forward_results = []
        
        for idx,lattice in sorted(self._lattice_layers.items()):  
            group = self._lattice_index_map[idx]
            #lattice = self._lattice_layers[str(idx)]
            if len(group) > 1:
                lattice_input = torch.stack([x[:, idx] for idx in group], dim=-2)
            else:
                lattice_input = x[:, group[0]]
            forward_results.append(lattice.forward(lattice_input))
        result = torch.cat(forward_results, dim=-1)
        if not self.average_outputs:
            return result
        result = torch.mean(result, dim=-1, keepdim=True)

        return result

    @torch.no_grad()
    def output_monotonicities(self) -> list[Union[Monotonicity, None]]:
        """Gives the monotonicities of the outputs of RTL.

        Returns:
            List of `Monotonicity` corresponding to each output of the RTL layer, in the
            same order as outputs.
        """
        monotonicities = []
        for monotonic_count, lattice in sorted(self._lattice_layers.items()):
            monotonic_count = int(monotonic_count)
            if monotonic_count:
                monotonicity = Monotonicity.INCREASING
            else:
                monotonicity = None
            for _ in range(lattice.units):
                monotonicities.append(monotonicity)

        return monotonicities

    @torch.no_grad()
    def apply_constraints(self) -> None:
        """Enforces constraints for each lattice in RTL."""
        for lattice in self._lattice_layers.values():
            lattice.apply_constraints()

    @torch.no_grad()
    def assert_constraints(self, eps: float = 1e-6) -> list[list[str]]:
        """Asserts that each Lattice in RTL satisfies all constraints.

        Args:
          eps: allowed constraints violations.

        Returns:
          List of lists, each with constraints violations for an individual Lattice.
        """
        return list(
            lattice.assert_constraints(eps=eps)
            for lattice in self._lattice_layers.values()
        )

    @staticmethod
    def _ensure_unique_sublattices(
        rtl_indices: list[list[int]],
        max_swaps: int = 10000,
    ) -> list[list[int]]:
        """Attempts to ensure every lattice in RTL structure contains unique features.

        Args:
            rtl_indices: list of lists where inner lists are groupings of
                indices of input features to RTL layer.
            max_swaps: maximum number of swaps to perform before giving up.

        Returns:
            List of lists where elements between inner lists have been swapped in
            an attempt to remove any duplicates from every grouping.
        """
        swaps = 0
        num_sublattices = len(rtl_indices)

        def find_swap_candidate(current_index, element):
            """Helper function to find the next sublattice not containing element."""
            for offset in range(1, num_sublattices):
                candidate_index = (current_index + offset) % num_sublattices
                if element not in rtl_indices[candidate_index]:
                    return candidate_index
            return None

        for i, sublattice in enumerate(rtl_indices):
            unique_elements = set()
            for element in sublattice:
                if element in unique_elements:
                    swap_with = find_swap_candidate(i, element)
                    if swap_with is not None:
                        for swap_element in rtl_indices[swap_with]:
                            if swap_element not in sublattice:
                                # Perform the swap
                                idx_to_swap = rtl_indices[swap_with].index(swap_element)
                                idx_duplicate = sublattice.index(element)
                                (
                                    rtl_indices[swap_with][idx_to_swap],
                                    sublattice[idx_duplicate],
                                ) = element, swap_element
                                swaps += 1
                                break
                    else:
                        logging.info(
                            "Some lattices in RTL may use the same feature multiple "
                            "times."
                        )
                        return rtl_indices
                else:
                    unique_elements.add(element)
                if swaps >= max_swaps:
                    logging.info(
                        "Some lattices in RTL may use the same feature multiple times."
                    )
                    return rtl_indices
        return rtl_indices

    @staticmethod
    def _insert_quantiles(
        rtl_indices: list[list[int]],
        lattice_rank: int,
        quantile_index: int
    )   -> list[list[int]]:
        """Inserts quantiles into the RTL structure.

        Args:
            rtl_indices: list of lists where inner lists are groupings of
                indices of input features to RTL layer.
            lattice_rank: number of inputs for each lattice in RTL structure.

        Returns:
            List of lists where elements between inner lists have been swapped in
            an attempt to remove any duplicates from every grouping.
        """
        lattice_rank = lattice_rank
        new_index = []
        for i, index in enumerate(rtl_indices):
            new_index.append(index)
            if (i+1)% lattice_rank == 0 and index != quantile_index:
                new_index.append(quantile_index)
            
            
        return np.array(new_index)   