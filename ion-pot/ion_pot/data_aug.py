import torch
import numpy as np

from dataclasses import dataclass
from typing import Union
from .utils import round_half_up, initialize_grid_space


class VoxelGridOperationMixin:
    @staticmethod
    def indices2xyz(
        indices: np.ndarray,
        grid_origin: tuple[float],
        grid_space: float,
    ):
        return np.array(grid_origin)[None, :] + indices * grid_space

    @staticmethod
    def xyz2indices(
        xyz: np.ndarray,
        grid_origin: tuple[float],
        grid_space: float,
    ):
        return round_half_up(
            (xyz - np.array(grid_origin)[None, :]) / grid_space
        ).astype(int)

    @staticmethod
    def get_valid_indices_mask(
        indices: np.ndarray,
        grid_dims: tuple[int],
    ):
        mask = np.all(indices >= 0, axis=1) & np.all(
            indices < np.array(grid_dims)[None, :],
            axis=1,
        )
        return mask

    @staticmethod
    def shrink_to_bounding_box(
        atom_indices: np.ndarray,
    ):
        # Find the bounding box of the non-zero elements
        min_z, min_y, min_x = np.min(atom_indices, axis=0)
        max_z, max_y, max_x = np.max(atom_indices, axis=0)

        # new size
        new_size = (
            max_z - min_z + 1,
            max_y - min_y + 1,
            max_x - min_x + 1,
        )

        return new_size, (min_z, min_y, min_x)


class VoxelPatchRotationDataAugmentationMixin:
    @staticmethod
    def dispatch_rotation_matrix(angle: float, axis: str):
        # Define the rotation matrix for a 30-degree rotation around the z-axis
        theta = np.array(angle * np.pi / 180.0)  # Convert degrees to radians
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        R = None
        if axis == "x":
            R = np.array(
                [[1, 0, 0], [0, cos_theta, -sin_theta], [0, sin_theta, cos_theta]],
                dtype=np.float64,
            )
        elif axis == "y":
            # Rotation matrix for y-axis
            R = np.array(
                [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]],
                dtype=np.float64,
            )
        elif axis == "z":
            R = np.array(
                [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]],
                dtype=np.float64,
            )
        else:
            raise ValueError("Unknown axis")
        return R

    @staticmethod
    def get_rotation_matrix(
        angle: Union[float, list[float]], axis: Union[str, list[str]]
    ):
        if isinstance(angle, float) and isinstance(axis, str):
            return VoxelPatchRotationDataAugmentationMixin.dispatch_rotation_matrix(
                angle, axis
            )
        elif isinstance(angle, list) and isinstance(axis, list):
            assert len(angle) == len(axis), "Angle and axis should have same length"
            Rs = np.eye(3, dtype=np.float64)
            for a, ax in zip(angle, axis):
                Rs = (
                    VoxelPatchRotationDataAugmentationMixin.dispatch_rotation_matrix(
                        a, ax
                    )
                    @ Rs
                )
            return Rs
        else:
            raise ValueError(
                "Angle and axis should have same length or both are single value"
            )

    @staticmethod
    def rotate_indices(
        indices: np.ndarray,
        grid_dims: tuple[int],
        angle: Union[float, list[float]] = 30,
        axis: Union[str, list[str]] = "z",
    ):
        # Define the rotation matrix
        Rot = VoxelPatchRotationDataAugmentationMixin.get_rotation_matrix(angle, axis)

        D, H, W = grid_dims
        midpoint = np.array([W // 2, H // 2, D // 2])[None, :]

        translated_indices = indices - midpoint
        rotated_indices = Rot @ translated_indices.T.astype(np.float64)  # Shape: (3, N)

        # Translate coordinates back to their original position
        rotated_indices = round_half_up(rotated_indices + midpoint).T.astype(int)
        return rotated_indices

    @staticmethod
    def rotate_voxel_indices(
        grid_dims: tuple[int],
        atom_indices: np.ndarray,
        other_indices: list[np.ndarray] = [],
        angle: Union[float, list[float]] = 30,
        axis: Union[str, list[str]] = "z",
    ):
        rotated_atom_indices = VoxelPatchRotationDataAugmentationMixin.rotate_indices(
            atom_indices, grid_dims, angle, axis
        )
        rotated_other_indices = [
            VoxelPatchRotationDataAugmentationMixin.rotate_indices(
                indices, grid_dims, angle, axis
            )
            for indices in other_indices
        ]

        # Generate a grid of coordinates
        previous_bouding_box, _ = VoxelGridOperationMixin.shrink_to_bounding_box(
            atom_indices
        )
        ratio = np.mean(np.array(grid_dims) / np.array(previous_bouding_box)).item()
        rotated_bounding_box, _ = VoxelGridOperationMixin.shrink_to_bounding_box(
            rotated_atom_indices
        )
        new_grid_dims = tuple(
            [
                round_half_up(dim * ratio).astype(int).item()
                for dim in rotated_bounding_box
            ]
        )

        rotated_arrays = [rotated_atom_indices] + rotated_other_indices
        return new_grid_dims, rotated_arrays

    @staticmethod
    def rotate_xyz(
        xyz: np.ndarray,
        grid_dims: tuple[int],
        angle: Union[float, list[float]] = 30,
        axis: Union[str, list[str]] = "z",
    ):
        # Define the rotation matrix
        Rot = VoxelPatchRotationDataAugmentationMixin.get_rotation_matrix(angle, axis)
        rotated_indices = Rot @ xyz.T.astype(np.float64)
        return rotated_indices.T

    @staticmethod
    def rotate_and_scale_voxel_xyz(
        grid_dims: tuple[int],
        grid_origin: tuple[float],
        grid_space: float,
        atom_indices: np.ndarray,
        atom_rad: np.ndarray,
        other_indices: list[np.ndarray] = [],
        scale_to_grid_space: float = 0.75,
        angle: Union[float, list[float]] = 30,
        axis: Union[str, list[str]] = "z",
    ):
        # for helping rebuild the voxel grid
        atom_xyz = VoxelGridOperationMixin.indices2xyz(
            atom_indices, grid_origin, grid_space
        )
        other_xyz = [
            VoxelGridOperationMixin.indices2xyz(ind, grid_origin, grid_space)
            for ind in other_indices
        ]

        # rotate the coordinates if needed
        if angle != 0:
            rotated_atom_xyz = VoxelPatchRotationDataAugmentationMixin.rotate_xyz(
                atom_xyz, grid_dims, angle, axis
            )
            rotated_other_xyz = [
                VoxelPatchRotationDataAugmentationMixin.rotate_xyz(
                    xyz, grid_dims, angle, axis
                )
                for xyz in other_xyz
            ]
        else:
            rotated_atom_xyz = atom_xyz
            rotated_other_xyz = other_xyz

        # create a new grid if needed to hold the rotated coordinates
        # 1. if angle is not 0 or
        # 2. if scale_to_grid_space is not equal to grid_space
        rotated_arrays = []
        if angle != 0 or scale_to_grid_space != grid_space:
            new_grid_dims, new_grid_origin, rotated_atom_indices = (
                initialize_grid_space(
                    rotated_atom_xyz,
                    atom_rad=atom_rad,
                    fill_ratio=2,
                    h=scale_to_grid_space,
                )
            )

            rotated_other_indices = [
                VoxelGridOperationMixin.xyz2indices(
                    xyz, new_grid_origin, scale_to_grid_space
                )
                for xyz in rotated_other_xyz
            ]
            rotated_arrays = [
                round_half_up(rotated_atom_indices).astype(int)
            ] + rotated_other_indices
            rotated_grid_dims = new_grid_dims.astype(int).tolist()
            rotated_grid_space = scale_to_grid_space
        else:
            rotated_arrays = [atom_indices] + other_indices
            rotated_grid_dims = grid_dims
            rotated_grid_space = grid_space

        return rotated_grid_dims, rotated_grid_space, rotated_arrays


@dataclass
class AtomicDataCollator:
    contain_boundary: bool = False

    def offset_batch_boundary_indices(self, batch: list[dict]):
        batch_arange = torch.arange(len(batch), dtype=torch.long)
        num_of_atoms = torch.tensor(
            [d["atom_xyz"].shape[0] for d in batch], dtype=torch.long
        )
        num_of_boundaries = torch.tensor(
            [d["boundary_xyz"].shape[0] for d in batch], dtype=torch.long
        )
        num_of_level_set = torch.tensor(
            [d["boundary_level_set_xyz"].shape[0] for d in batch], dtype=torch.long
        )

        batch_of_boundaries = torch.repeat_interleave(
            batch_arange, num_of_boundaries, dim=0
        )
        batch_of_level_set_indices = torch.repeat_interleave(
            batch_arange, num_of_level_set, dim=0
        )

        # offset the atom indices
        atom_side_offsets = torch.cumsum(num_of_atoms, dim=0)
        boundaries_side_offsets = torch.cumsum(num_of_boundaries, dim=0)
        level_set_side_offsets = torch.cumsum(num_of_level_set, dim=0)
        for bidx, d in enumerate(batch):
            if bidx > 0:
                d["atom2boundary_i"] = (
                    d["atom2boundary_i"] + atom_side_offsets[bidx - 1]
                )
                d["atom2boundary_j"] = (
                    d["atom2boundary_j"] + boundaries_side_offsets[bidx - 1]
                )

                d["boundary2level_set_i"] = (
                    d["boundary2level_set_i"] + boundaries_side_offsets[bidx - 1]
                )
                d["boundary2level_set_j"] = (
                    d["boundary2level_set_j"] + level_set_side_offsets[bidx - 1]
                )

        return batch_of_boundaries, batch_of_level_set_indices, batch

    def offset_batch_indices(self, batch: list[dict]):
        batch_arange = torch.arange(len(batch), dtype=torch.long)
        num_of_atoms = torch.tensor(
            [d["atom_xyz"].shape[0] for d in batch], dtype=torch.long
        )
        num_of_level_set = torch.tensor(
            [d["level_set_xyz"].shape[0] for d in batch], dtype=torch.long
        )

        batch_of_atom_indices = torch.repeat_interleave(
            batch_arange, num_of_atoms, dim=0
        )
        batch_of_level_set_indices = torch.repeat_interleave(
            batch_arange, num_of_level_set, dim=0
        )

        # offset the atom indices
        atom_side_offsets = torch.cumsum(num_of_atoms, dim=0)
        level_set_side_offsets = torch.cumsum(num_of_level_set, dim=0)
        for bidx, d in enumerate(batch):
            if bidx > 0:
                if d["atom_idx_i"] is not None and d["atom_idx_j"] is not None:
                    d["atom_idx_i"] = d["atom_idx_i"] + atom_side_offsets[bidx - 1]
                    d["atom_idx_j"] = d["atom_idx_j"] + atom_side_offsets[bidx - 1]

                if (
                    d["atom2level_set_i"] is not None
                    and d["atom2level_set_j"] is not None
                ):
                    d["atom2level_set_i"] = (
                        d["atom2level_set_i"] + atom_side_offsets[bidx - 1]
                    )
                    d["atom2level_set_j"] = (
                        d["atom2level_set_j"] + level_set_side_offsets[bidx - 1]
                    )

        return batch_of_atom_indices, batch_of_level_set_indices, batch

    def __call__(self, batch: list[Union[dict, list[dict]]]):
        collate_batch = dict()
        if isinstance(batch[0], dict):
            batch_of_atom_indices, batch_of_level_set_indices, batch = (
                self.offset_batch_indices(batch)
            )
            collate_batch = dict(
                batch_of_atom_indices=batch_of_atom_indices,
                batch_of_level_set_indices=batch_of_level_set_indices,
            )

            if self.contain_boundary:
                batch_of_boundaries, batch_of_boundaries_level_set_indices, batch = (
                    self.offset_batch_boundary_indices(batch)
                )
                collate_batch.update(
                    dict(
                        batch_of_boundaries=batch_of_boundaries,
                        batch_of_boundaries_level_set_indices=batch_of_boundaries_level_set_indices,
                    )
                )

            for key in batch[0]:
                if isinstance(batch[0][key], torch.Tensor):
                    collate_batch[key] = torch.cat([d[key] for d in batch], dim=0)
                elif isinstance(batch[0][key], (float,)):
                    collate_batch[key] = torch.tensor(
                        [d[key] for d in batch], dtype=torch.float32
                    )
                elif isinstance(batch[0][key], (int,)):
                    collate_batch[key] = torch.tensor(
                        [d[key] for d in batch], dtype=torch.long
                    )
                else:
                    pass

        elif isinstance(batch[0], list):
            # here we assume that the batch size is 1
            return self(batch[0])

        return collate_batch
