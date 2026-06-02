import math
import torch
import schnetpack.nn as snn
import schnetpack.transform as trn
import schnetpack.properties as schnet_properties
import torch.utils.checkpoint as checkpoint

from typing import Callable, Dict, Literal, Optional, Union
from torch import nn
from schnetpack.nn import Dense, scatter_add
from schnetpack.nn.activations import shifted_softplus
from dataclasses import dataclass


class SinusoidalPosEmb3D(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[..., None] * emb[None, None, None, None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SinusoidalPosEmb1D(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[..., None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LargeScaleTorchNeighborList(trn.TorchNeighborList):
    def __init__(
        self,
        cutoff: float,
        max_num_neighbors: Union[int, None] = None,
        natom_threshold_for_max_num_neighbors: int = 50000,
    ):
        """
        Args:
            cutoff: Cutoff radius for neighbor search.
        """
        super().__init__(cutoff=cutoff)
        self._max_num_neighbors = max_num_neighbors
        self._natom_threshold_for_max_num_neighbors = (
            natom_threshold_for_max_num_neighbors
        )

    def _build_neighbor_list(self, Z, positions, cell, pbc, cutoff):
        # Check if shifts are needed for periodic boundary conditions
        if torch.all(pbc == 0):
            shifts = torch.zeros(0, 3, device=cell.device, dtype=torch.long)
        else:
            shifts = self._get_shifts(cell, pbc, cutoff)
        idx_i, idx_j, offset = self._get_neighbor_pairs(positions, cell, shifts, cutoff)

        return idx_i, idx_j, offset

    def _get_neighbor_pairs(self, positions, cell, shifts, cutoff):
        """Compute pairs of atoms that are neighbors
        Copyright 2018- Xiang Gao and other ANI developers
        (https://github.com/aiqm/torchani/blob/master/torchani/aev.py)
        Arguments:
            positions (:class:`torch.Tensor`): tensor of shape
                (molecules, atoms, 3) for atom coordinates.
            cell (:class:`torch.Tensor`): tensor of shape (3, 3) of the three vectors
                defining unit cell: tensor([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
            shifts (:class:`torch.Tensor`): tensor of shape (?, 3) storing shifts
        """
        num_atoms = positions.shape[0]
        all_atoms = torch.arange(num_atoms, device=cell.device)

        max_query_atoms = 1024
        query_atoms_splits = torch.split(all_atoms, max_query_atoms)
        all_atom_index_i = []
        all_atom_index_j = []
        for query_atoms in query_atoms_splits:
            # 1) Central cell
            pi_center, pj_center = torch.cartesian_prod(query_atoms, all_atoms).unbind(
                -1
            )

            # ignore self interactions
            mask = pi_center != pj_center
            pi_center = pi_center[mask]
            pj_center = pj_center[mask]
            Rij_all = positions[pi_center] - positions[pj_center]

            # 5) Compute distances, and find all pairs within cutoff
            distances = torch.norm(Rij_all, dim=1)
            in_cutoff = torch.nonzero(distances < cutoff, as_tuple=False).squeeze()

            # filter max num neighbors if needed
            if (
                self._max_num_neighbors is not None
                and num_atoms > self._natom_threshold_for_max_num_neighbors
            ):
                # Extract the distances corresponding to the indices in in_cutoff
                distances_in_cutoff = distances[in_cutoff]

                # Sort the in_cutoff indices based on the corresponding distances
                sorted_indices = torch.argsort(distances_in_cutoff)
                in_cutoff_sorted = in_cutoff[sorted_indices]

                # 6) Reduce tensors to relevant components
                atom_index_i = pi_center[in_cutoff_sorted]
                atom_index_j = pj_center[in_cutoff_sorted]

                # sorted by index i
                sorted_pair_index = torch.sort(atom_index_i, stable=True).indices
                atom_index_i = atom_index_i[sorted_pair_index]
                atom_index_j = atom_index_j[sorted_pair_index]
                atom_index_i_unique = torch.unique(atom_index_i)

                # Iterate over each unique value
                mask = torch.zeros_like(atom_index_i, dtype=torch.bool)
                for value in atom_index_i_unique:
                    # Get the indices where atom_index_i equals the current unique value
                    indices_for_value = torch.nonzero(
                        atom_index_i == value, as_tuple=False
                    ).squeeze()

                    # Select the first N indices (or fewer if less than N are available)
                    mask[indices_for_value[: self._max_num_neighbors]] = True

                atom_index_i = atom_index_i[mask]
                atom_index_j = atom_index_j[mask]
            else:
                atom_index_i = pi_center[in_cutoff]
                atom_index_j = pj_center[in_cutoff]

            all_atom_index_i.append(atom_index_i)
            all_atom_index_j.append(atom_index_j)

        atom_index_i = torch.cat(all_atom_index_i)
        atom_index_j = torch.cat(all_atom_index_j)
        return atom_index_i, atom_index_j, None


class PBGNNInteraction(nn.Module):
    r"""PBGNN interaction block for modeling interactions of atomistic systems."""

    def __init__(
        self,
        n_atom_basis: int,
        n_rbf: int,
        n_filters: int,
        activation: Callable = shifted_softplus,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            n_rbf (int): number of radial basis functions.
            n_filters: number of filters used in continuous-filter convolution.
            activation: if None, no activation function is used.
        """
        super(PBGNNInteraction, self).__init__()
        self.n_atom_basis = n_atom_basis
        self.n_rbf = n_rbf
        self.n_filters = n_filters
        self.activation = activation

        self.in2f = Dense(n_atom_basis, n_filters, bias=False, activation=None)
        self.f2out = nn.Sequential(
            Dense(n_filters, n_atom_basis, activation=activation),
            Dense(n_atom_basis, n_atom_basis, activation=None),
        )
        self.filter_network = nn.Sequential(
            Dense(n_rbf, n_filters, activation=activation), Dense(n_filters, n_filters)
        )

    def forward(
        self,
        x: torch.Tensor,
        f_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rcut_ij: torch.Tensor,
    ):
        """Compute interaction output.

        Args:
            x: input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            atom features after interaction
        """
        x = self.in2f(x)
        Wij = self.filter_network(f_ij)
        Wij = Wij * rcut_ij[:, None]

        # continuous-filter convolution
        x_j = x[idx_j]
        x_ij = x_j * Wij
        x = scatter_add(x_ij, idx_i, dim_size=x.shape[0])

        x = self.f2out(x)
        return x


class LargeScalePBGNNInteraction(PBGNNInteraction):
    def __init__(
        self,
        split_size: Union[int, None] = None,
        fused_interaction_block: bool = False,
        fused_feature_num: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.split_size = split_size
        self.fused_interaction_block = fused_interaction_block

        if self.fused_interaction_block:
            self.fused_network = nn.Sequential(
                Dense(
                    self.n_rbf * fused_feature_num,
                    self.n_filters,
                    activation=self.activation,
                ),
                Dense(self.n_filters, self.n_filters),
            )

    def _linear_filter(
        self,
        x: torch.Tensor,
        f_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rcut_ij: torch.Tensor,
        co_ij: torch.Tensor = None,
    ):
        Wij = self.filter_network(f_ij)
        Wij = Wij * rcut_ij[:, None]

        # fused interaction block
        if self.fused_interaction_block:
            Wij = Wij + self.fused_network(co_ij) * rcut_ij[:, None]

        # continuous-filter convolution
        x_j = x[idx_j]
        x_ij = x_j * Wij
        conv_x = scatter_add(x_ij, idx_i, dim_size=x.shape[0])
        return conv_x

    def forward(
        self,
        x: torch.Tensor,
        f_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rcut_ij: torch.Tensor,
        co_ij: Optional[torch.Tensor] = None,
    ):
        """Compute interaction output.

        Args:
            x: input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            atom features after interaction
        """
        x = self.in2f(x)
        if self.split_size is None:
            conv_x = self._linear_filter(x, f_ij, idx_i, idx_j, rcut_ij, co_ij)
        else:
            list_of_f_ij = torch.split(f_ij, self.split_size)
            list_of_idx_i = torch.split(idx_i, self.split_size)
            list_of_idx_j = torch.split(idx_j, self.split_size)
            list_of_rcut_ij = torch.split(rcut_ij, self.split_size)
            list_of_co_ij = (
                torch.split(co_ij, self.split_size)
                if co_ij is not None
                else [None] * len(list_of_f_ij)
            )
            conv_x = torch.zeros_like(x)

            for sub_f_ij, sub_idx_i, sub_idx_j, sub_rcut_ij, sub_co_ij in zip(
                list_of_f_ij,
                list_of_idx_i,
                list_of_idx_j,
                list_of_rcut_ij,
                list_of_co_ij,
            ):
                conv_x = conv_x + self._linear_filter(
                    x, sub_f_ij, sub_idx_i, sub_idx_j, sub_rcut_ij, sub_co_ij
                )

        x = self.f2out(conv_x)
        return x


class LargeScalePBGNNCrossInteraction(PBGNNInteraction):
    def __init__(
        self,
        split_size: Union[int, None] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.split_size = split_size
        self.in2f_y = Dense(
            self.n_atom_basis, self.n_filters, bias=False, activation=None
        )
        self.f2out_y = nn.Sequential(
            Dense(self.n_filters, self.n_atom_basis, activation=self.activation),
            Dense(self.n_atom_basis, self.n_atom_basis, activation=None),
        )

    def _linear_filter(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        f_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rcut_ij: torch.Tensor,
    ):
        Wij = self.filter_network(f_ij)
        Wij = Wij * rcut_ij[:, None]

        # continuous-filter convolution
        x_j = y[idx_j]
        y_i = x[idx_i]
        x_ij = x_j * Wij
        y_ij = y_i * Wij
        conv_x = scatter_add(x_ij, idx_i, dim_size=x.shape[0])
        conv_y = scatter_add(y_ij, idx_j, dim_size=y.shape[0])
        return conv_x, conv_y

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        f_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rcut_ij: torch.Tensor,
    ):
        """Compute interaction output.

        Args:
            x: input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            atom features after interaction
        """
        x = self.in2f(x)
        y = self.in2f_y(y)
        if self.split_size is None:
            conv_x, conv_y = self._linear_filter(x, y, f_ij, idx_i, idx_j, rcut_ij)
        else:
            list_of_f_ij = torch.split(f_ij, self.split_size)
            list_of_idx_i = torch.split(idx_i, self.split_size)
            list_of_idx_j = torch.split(idx_j, self.split_size)
            list_of_rcut_ij = torch.split(rcut_ij, self.split_size)
            conv_x = torch.zeros_like(x)
            conv_y = torch.zeros_like(y)

            for sub_f_ij, sub_idx_i, sub_idx_j, sub_rcut_ij in zip(
                list_of_f_ij,
                list_of_idx_i,
                list_of_idx_j,
                list_of_rcut_ij,
            ):
                _conv_x, _conv_y = self._linear_filter(
                    x, y, sub_f_ij, sub_idx_i, sub_idx_j, sub_rcut_ij
                )

                conv_x = conv_x + _conv_x
                conv_y = conv_y + _conv_y

        x = self.f2out(conv_x)
        y = self.f2out_y(conv_y)
        return x, y


@dataclass
class PBGNNConfig:
    input_dim: int = 16
    cutoff: float = 5.0
    n_atom_basis: int = 30
    n_interactions: int = 3
    n_filters: Union[int, None] = None
    radial_basis: Literal["GaussianRBF"] = "GaussianRBF"
    shared_interactions: bool = False
    max_z: int = 100
    activation: Literal["shifted_softplus"] = "shifted_softplus"
    n_rbf: int = 20
    neighbor_list_cutoff: float = 5.0
    max_num_neighbors: Union[int, None] = None
    natom_threshold_for_max_num_neighbors: int = 50000
    conv_split_size: Union[int, None] = None
    use_gradient_checkpointing: bool = False
    use_fused_interaction_block: bool = False


class PBGNN(nn.Module):
    """PBGNN architecture for learning representations of atomistic systems"""

    def __init__(
        self,
        config: PBGNNConfig,
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            n_filters: number of filters used in continuous-filter convolution
            shared_interactions: if True, share the weights across
                interaction blocks and filter-generating networks.
            max_z: maximal nuclear charge
            activation: activation function
        """
        super().__init__()
        self.config = config
        self.n_atom_basis = config.n_atom_basis
        self.size = (self.n_atom_basis,)
        self.n_filters = config.n_filters or self.n_atom_basis

        self.cutoff = self.config.cutoff
        self.cutoff_fn = self.set_up_cutoff_fn()
        self.radial_basis = self.set_up_radial_basis()

        self._init_modules()

    def _init_modules(self):
        config = self.config

        # layers
        self.input_fn = nn.Sequential(
            nn.Linear(config.input_dim, self.n_atom_basis),
            nn.GELU(),
            nn.Linear(config.n_atom_basis, self.n_atom_basis),
        )
        self.interactions = snn.replicate_module(
            lambda: LargeScalePBGNNInteraction(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=self.set_up_activation(),
                split_size=self.config.conv_split_size,
                fused_interaction_block=self.config.use_fused_interaction_block,
                fused_feature_num=2,
            ),
            config.n_interactions,
            config.shared_interactions,
        )
        self.output_fn = nn.Sequential(
            nn.Linear(self.n_atom_basis, self.n_atom_basis),
            nn.GELU(),
            nn.Linear(self.n_atom_basis, 1),
        )

        # fused interaction block
        if config.use_fused_interaction_block:
            self.charge_diff = nn.Sequential(
                nn.Linear(config.input_dim, self.config.n_rbf // 2),
                nn.GELU(),
                nn.Linear(self.config.n_rbf // 2, self.config.n_rbf),
            )
            self.position_diff = nn.Sequential(
                nn.Linear(3, self.config.n_rbf // 2),
                nn.GELU(),
                nn.Linear(self.config.n_rbf // 2, self.config.n_rbf),
            )

    def set_up_radial_basis(self):
        if self.config.radial_basis == "GaussianRBF":
            return snn.GaussianRBF(self.config.n_rbf, self.cutoff)
        else:
            raise NotImplementedError(
                f"Radial basis function {self.config.radial_basis} not implemented."
            )

    def set_up_cutoff_fn(self):
        return snn.CosineCutoff(self.cutoff)

    def set_up_activation(self):
        if self.config.activation == "shifted_softplus":
            return snn.shifted_softplus
        else:
            raise NotImplementedError(
                f"Activation function {self.config.activation} not implemented."
            )

    @torch.no_grad()
    def create_neighbor_list(
        self, batch_of_atom_indices: torch.Tensor, atom_xyz: torch.Tensor
    ):
        unique_batch_of_atom_indices, unique_num_of_atoms = torch.unique(
            batch_of_atom_indices, return_counts=True
        )

        transform = LargeScaleTorchNeighborList(
            cutoff=self.config.neighbor_list_cutoff,
            max_num_neighbors=self.config.max_num_neighbors,
            natom_threshold_for_max_num_neighbors=self.config.natom_threshold_for_max_num_neighbors,
        )
        offsets = torch.cumsum(unique_num_of_atoms, dim=0).tolist()
        cell = atom_xyz.new_zeros((3, 3))
        pbc = atom_xyz.new_tensor([0, 0, 0], dtype=torch.bool)
        list_if_atom_idx_i = []
        list_if_atom_idx_j = []
        for i, bi in enumerate(unique_batch_of_atom_indices.tolist()):
            atom_indices = torch.where(batch_of_atom_indices == bi)[0]
            sub_atom_xyz = atom_xyz[atom_indices]
            results = transform(
                {
                    schnet_properties.Z: None,
                    schnet_properties.R: sub_atom_xyz,
                    schnet_properties.cell: cell,
                    schnet_properties.pbc: pbc,
                }
            )
            sub_atom_idx_i = results[schnet_properties.idx_i]
            sub_atom_idx_j = results[schnet_properties.idx_j]

            # add offset
            if i > 0:
                sub_atom_idx_i += offsets[i - 1]
                sub_atom_idx_j += offsets[i - 1]

            list_if_atom_idx_i.append(sub_atom_idx_i)
            list_if_atom_idx_j.append(sub_atom_idx_j)

        atom_idx_i = torch.cat(list_if_atom_idx_i)
        atom_idx_j = torch.cat(list_if_atom_idx_j)
        return atom_idx_i, atom_idx_j

    def forward(self, inputs: Dict[str, torch.Tensor]):
        atom_emb = inputs["atom_emb"]
        r = inputs["atom_xyz"]
        # idx_i, idx_j = self.create_neighbor_list(inputs["batch_of_atom_indices"], r)
        idx_i, idx_j = (
            inputs["batch_dict"]["atom_idx_i"],
            inputs["batch_dict"]["atom_idx_j"],
        )

        # compute atom and pair features
        x = self.input_fn(atom_emb)
        r_ij = r[idx_j] - r[idx_i]
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)

        # compute interaction features
        co_ij = None
        if self.config.use_fused_interaction_block:
            # p_ij = self.position_diff(
            #     r_ij / torch.linalg.vector_norm(r_ij, dim=1, keepdim=True)
            # )
            # co_ij = p_ij

            q_ij = self.charge_diff((atom_emb[idx_j] - atom_emb[idx_i]).detach())
            p_ij = self.position_diff(
                r_ij / torch.linalg.vector_norm(r_ij, dim=1, keepdim=True)
            )
            co_ij = torch.cat((q_ij, p_ij), dim=-1)

        # compute interaction block to update atomic embeddings
        for interaction in self.interactions:
            if self.training and self.config.use_gradient_checkpointing:
                v = checkpoint.checkpoint(
                    interaction, x, f_ij, idx_i, idx_j, rcut_ij, co_ij
                )
            else:
                v = interaction(x, f_ij, idx_i, idx_j, rcut_ij, co_ij)
            x = x + v

        x = self.output_fn(x).flatten()
        return x


class PBGNNwithCrossInteraction(PBGNN):
    """This is the very first trial on leveraging the level set information to improve the PBGNN model. The level set feature is used to compute the interaction between the atom and the level set, and be added to the atom feature before the interaction block."""

    def _init_modules(self):
        config = self.config

        # layers
        self.input_fn = nn.Sequential(
            nn.Linear(config.input_dim, self.n_atom_basis),
            nn.GELU(),
            nn.Linear(config.n_atom_basis, self.n_atom_basis),
        )
        self.lset_input_fn = nn.Sequential(
            nn.Linear(config.input_dim, self.n_atom_basis),
            nn.GELU(),
            nn.Linear(config.n_atom_basis, self.n_atom_basis),
        )

        self.interactions = snn.replicate_module(
            lambda: LargeScalePBGNNInteraction(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=self.set_up_activation(),
                split_size=self.config.conv_split_size,
                fused_interaction_block=self.config.use_fused_interaction_block,
                fused_feature_num=2,
            ),
            config.n_interactions,
            config.shared_interactions,
        )
        self.lset_interactions = snn.replicate_module(
            lambda: LargeScalePBGNNCrossInteraction(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=self.set_up_activation(),
                split_size=self.config.conv_split_size,
            ),
            config.n_interactions,
            config.shared_interactions,
        )

        self.output_fn = nn.Sequential(
            nn.Linear(self.n_atom_basis, self.n_atom_basis),
            nn.GELU(),
            nn.Linear(self.n_atom_basis, 1),
        )

        # fused interaction block
        if config.use_fused_interaction_block:
            self.charge_diff = nn.Sequential(
                nn.Linear(config.input_dim, self.config.n_rbf // 2),
                nn.GELU(),
                nn.Linear(self.config.n_rbf // 2, self.config.n_rbf),
            )
            self.position_diff = nn.Sequential(
                nn.Linear(3, self.config.n_rbf // 2),
                nn.GELU(),
                nn.Linear(self.config.n_rbf // 2, self.config.n_rbf),
            )

    def lset_forward(
        self,
        atom_emb: torch.Tensor,
        lset_emb: torch.Tensor,
        atom_xyz: torch.Tensor,
        lset_xyz: torch.Tensor,
        atom2lset_i: torch.Tensor,
        atom2lset_j: torch.Tensor,
    ):
        # compute atom and pair features
        r_ij = lset_xyz[atom2lset_j] - atom_xyz[atom2lset_i]
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)

        # compute interaction block to update atomic embeddings
        for interaction in self.lset_interactions:
            v, w = interaction(
                atom_emb, lset_emb, f_ij, atom2lset_i, atom2lset_j, rcut_ij
            )
            atom_emb = atom_emb + v
            lset_emb = lset_emb + w

        return atom_emb, lset_emb

    def lset_fusion(
        self,
        atom_emb: torch.Tensor,
        lset_emb: torch.Tensor,
        atom_xyz: torch.Tensor,
        lset_xyz: torch.Tensor,
        atom2lset_i: torch.Tensor,
        atom2lset_j: torch.Tensor,
    ):
        # _x, _ = self.lset_forward(x, x_lset, r, rr, atom2lset_i, atom2lset_j)
        _x, _x_lset = self.lset_forward(
            torch.zeros_like(atom_emb, requires_grad=False),
            lset_emb,
            atom_xyz,
            lset_xyz,
            atom2lset_i,
            atom2lset_j,
        )

        atom_emb = atom_emb + _x
        lset_emb = lset_emb + _x_lset
        return atom_emb, lset_emb

    def forward(self, inputs: Dict[str, torch.Tensor]):
        atom_emb = inputs["atom_emb"]
        lset_emb = inputs["encoding"]["lset_emb"]

        r = inputs["atom_xyz"]
        rr = inputs["batch_dict"]["level_set_xyz"]
        idx_i, idx_j = (
            inputs["batch_dict"]["atom_idx_i"],
            inputs["batch_dict"]["atom_idx_j"],
        )
        atom2lset_i = inputs["batch_dict"]["atom2level_set_i"]
        atom2lset_j = inputs["batch_dict"]["atom2level_set_j"]

        # feature fusion
        x = self.input_fn(atom_emb)
        x_lset = self.lset_input_fn(lset_emb)
        x, _ = self.lset_fusion(x, x_lset, r, rr, atom2lset_i, atom2lset_j)

        # compute atom and pair features
        r_ij = r[idx_j] - r[idx_i]
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)

        # compute interaction features
        co_ij = None
        if self.config.use_fused_interaction_block:
            q_ij = self.charge_diff((atom_emb[idx_j] - atom_emb[idx_i]).detach())
            p_ij = self.position_diff(
                r_ij / torch.linalg.vector_norm(r_ij, dim=1, keepdim=True)
            )
            co_ij = torch.cat((q_ij, p_ij), dim=-1)

        # compute interaction block to update atomic embeddings
        for interaction in self.interactions:
            if self.training and self.config.use_gradient_checkpointing:
                v = checkpoint.checkpoint(
                    interaction, x, f_ij, idx_i, idx_j, rcut_ij, co_ij
                )
            else:
                v = interaction(x, f_ij, idx_i, idx_j, rcut_ij, co_ij)
            x = x + v

        x = self.output_fn(x).flatten()
        return x


class PBGNNwithCrossInteractionV2(PBGNNwithCrossInteraction):
    def forward(self, inputs: Dict[str, torch.Tensor]):
        atom_emb = inputs["atom_emb"]
        lset_emb = inputs["encoding"]["lset_emb"]

        r = inputs["atom_xyz"]
        rr = inputs["batch_dict"]["level_set_xyz"]
        idx_i, idx_j = (
            inputs["batch_dict"]["atom_idx_i"],
            inputs["batch_dict"]["atom_idx_j"],
        )
        atom2lset_i = inputs["batch_dict"]["atom2level_set_i"]
        atom2lset_j = inputs["batch_dict"]["atom2level_set_j"]

        # feature fusion
        x = self.input_fn(atom_emb)
        x_lset = self.lset_input_fn(lset_emb)
        x, x_lset = self.lset_fusion(x, x_lset, r, rr, atom2lset_i, atom2lset_j)

        # compute atom and pair features
        r_ij = r[idx_j] - r[idx_i]
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)

        # compute interaction features
        co_ij = None
        if self.config.use_fused_interaction_block:
            q_ij = self.charge_diff((atom_emb[idx_j] - atom_emb[idx_i]).detach())
            p_ij = self.position_diff(
                r_ij / torch.linalg.vector_norm(r_ij, dim=1, keepdim=True)
            )
            co_ij = torch.cat((q_ij, p_ij), dim=-1)

        # compute interaction block to update atomic embeddings
        for interaction in self.interactions:
            if self.training and self.config.use_gradient_checkpointing:
                v = checkpoint.checkpoint(
                    interaction, x, f_ij, idx_i, idx_j, rcut_ij, co_ij
                )
            else:
                v = interaction(x, f_ij, idx_i, idx_j, rcut_ij, co_ij)
            x = x + v

        x = self.output_fn(x).flatten()
        return x


class PBGNNwithBoundaryCrossInteraction(PBGNNwithCrossInteraction):
    def _init_modules(self):
        config = self.config

        # layers
        self.input_fn = nn.Sequential(
            nn.Linear(config.input_dim, self.n_atom_basis),
            nn.GELU(),
            nn.Linear(config.n_atom_basis, self.n_atom_basis),
        )
        self.lset_input_fn = nn.Sequential(
            nn.Linear(config.input_dim, self.n_atom_basis),
            nn.GELU(),
            nn.Linear(config.n_atom_basis, self.n_atom_basis),
        )

        self.interactions = snn.replicate_module(
            lambda: LargeScalePBGNNInteraction(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=self.set_up_activation(),
                split_size=self.config.conv_split_size,
                fused_interaction_block=self.config.use_fused_interaction_block,
                fused_feature_num=2,
            ),
            config.n_interactions,
            config.shared_interactions,
        )
        self.atom_bundy_interactions = snn.replicate_module(
            lambda: LargeScalePBGNNCrossInteraction(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=self.set_up_activation(),
                split_size=self.config.conv_split_size,
            ),
            config.n_interactions,
            config.shared_interactions,
        )
        self.lset_interactions = snn.replicate_module(
            lambda: LargeScalePBGNNCrossInteraction(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=self.set_up_activation(),
                split_size=self.config.conv_split_size,
            ),
            config.n_interactions,
            config.shared_interactions,
        )

        self.output_fn = nn.Sequential(
            nn.Linear(self.n_atom_basis, self.n_atom_basis),
            nn.GELU(),
            nn.Linear(self.n_atom_basis, 1),
        )
        self.bndy_output_fn = nn.Sequential(
            nn.Linear(self.n_atom_basis, self.n_atom_basis),
            nn.GELU(),
            nn.Linear(self.n_atom_basis, 1),
        )

    def bndy_forward(
        self,
        bndy_emb: torch.Tensor,
        bndy_lset_emb: torch.Tensor,
        bndy_xyz: torch.Tensor,
        bndy_lset_xyz: torch.Tensor,
        bndy2lset_i: torch.Tensor,
        bndy2lset_j: torch.Tensor,
    ):
        x = self.input_fn(bndy_emb)
        x_lset = self.lset_input_fn(bndy_lset_emb)
        bndy_emb, lset_emb = self.lset_fusion(
            x, x_lset, bndy_xyz, bndy_lset_xyz, bndy2lset_i, bndy2lset_j
        )
        return bndy_emb, lset_emb

    def atom_forward(
        self,
        atom_emb: torch.Tensor,
        lset_emb: torch.Tensor,
        atom_xyz: torch.Tensor,
        lset_xyz: torch.Tensor,
        atom2lset_i: torch.Tensor,
        atom2lset_j: torch.Tensor,
    ):
        # feature fusion
        x = self.input_fn(atom_emb)
        x_lset = self.lset_input_fn(lset_emb)
        atom_emb, lset_emb = self.lset_fusion(
            x, x_lset, atom_xyz, lset_xyz, atom2lset_i, atom2lset_j
        )
        return atom_emb, lset_emb

    def atom_bndy_interaction(
        self,
        atom_emb: torch.Tensor,
        bndy_emb: torch.Tensor,
        atom_xyz: torch.Tensor,
        bndy_xyz: torch.Tensor,
        atom_i: torch.Tensor,
        atom_j: torch.Tensor,
        atom2bndy_i: torch.Tensor,
        atom2bndy_j: torch.Tensor,
    ):
        # compute atom and pair features
        r_ij = atom_xyz[atom_j] - atom_xyz[atom_i]
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)

        rr_ij = bndy_xyz[atom2bndy_j] - atom_xyz[atom2bndy_i]
        dd_ij = torch.norm(rr_ij, dim=1)
        ff_ij = self.radial_basis(dd_ij)
        rrcut_ij = self.cutoff_fn(dd_ij)

        # compute interaction block to update atomic embeddings
        co_ij = None
        for interaction, atom_bundy_interaction in zip(
            self.interactions, self.atom_bundy_interactions
        ):
            if self.training and self.config.use_gradient_checkpointing:
                v = checkpoint.checkpoint(
                    interaction, atom_emb, f_ij, atom_i, atom_j, rcut_ij, co_ij
                )
                atom_emb = atom_emb + v
                u, w = checkpoint.checkpoint(
                    atom_bundy_interaction,
                    atom_emb,
                    bndy_emb,
                    ff_ij,
                    atom2bndy_i,
                    atom2bndy_j,
                    rrcut_ij,
                )
                atom_emb = atom_emb + u
                bndy_emb = bndy_emb + w
            else:
                v = interaction(atom_emb, f_ij, atom_i, atom_j, rcut_ij, co_ij)
                atom_emb = atom_emb + v
                u, w = atom_bundy_interaction(
                    atom_emb, bndy_emb, ff_ij, atom2bndy_i, atom2bndy_j, rrcut_ij
                )
                atom_emb = atom_emb + u
                bndy_emb = bndy_emb + w

        # atom and boundary points are jointly used to predict the output
        # y1: torch.Tensor = self.output_fn(atom_emb).flatten()
        # y2: torch.Tensor = self.bndy_output_fn(bndy_emb).flatten()
        # y = y1.scatter_add(0, atom2bndy_i, y2[atom2bndy_j])

        # use fused output (this method seems the best)
        y1: torch.Tensor = self.output_fn(atom_emb).flatten()
        y = y1

        # y2: torch.Tensor = self.bndy_output_fn(bndy_emb).flatten()
        # y1: torch.Tensor = atom_emb.new_zeros((atom_emb.shape[0],), requires_grad=False)
        # y = y1.scatter_add(0, atom2bndy_i, y2[atom2bndy_j])
        return y

    def forward(self, inputs: Dict[str, torch.Tensor]):
        # dispatch inputs
        atom_emb = inputs["atom_emb"]
        lset_emb = inputs["encoding"]["lset_emb"]

        r = inputs["atom_xyz"]
        rr = inputs["batch_dict"]["level_set_xyz"]
        idx_i, idx_j = (
            inputs["batch_dict"]["atom_idx_i"],
            inputs["batch_dict"]["atom_idx_j"],
        )
        atom2lset_i = inputs["batch_dict"]["atom2level_set_i"]
        atom2lset_j = inputs["batch_dict"]["atom2level_set_j"]

        bndy_emb = inputs["encoding"]["boundary_charge_emb"]
        bndy_lset_emb = inputs["encoding"]["boundary_lset_emb"]
        bndy_xyz = inputs["batch_dict"]["boundary_xyz"]
        bndy_lset_xyz = inputs["batch_dict"]["boundary_level_set_xyz"]
        bndy2lset_i = inputs["batch_dict"]["boundary2level_set_i"]
        bndy2lset_j = inputs["batch_dict"]["boundary2level_set_j"]

        atom2bndy_i = inputs["batch_dict"]["atom2boundary_i"]
        atom2bndy_j = inputs["batch_dict"]["atom2boundary_j"]

        # feature fusion
        atom_emb, lset_emb = self.atom_forward(
            atom_emb, lset_emb, r, rr, atom2lset_i, atom2lset_j
        )
        bndy_emb, bndy_lset_emb = self.bndy_forward(
            bndy_emb, bndy_lset_emb, bndy_xyz, bndy_lset_xyz, bndy2lset_i, bndy2lset_j
        )

        # compute atom and pair features
        y = self.atom_bndy_interaction(
            atom_emb, bndy_emb, r, bndy_xyz, idx_i, idx_j, atom2bndy_i, atom2bndy_j
        )
        return y
