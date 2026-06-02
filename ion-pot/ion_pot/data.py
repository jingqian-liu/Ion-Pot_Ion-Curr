import os
import glob
import torch
import math
import numpy as np
import json
import gzip
import pickle
import re

from scipy.spatial import KDTree
from itertools import product
from typing import Any, Dict, List, Tuple, Union
from torch.utils.data import Dataset
from tqdm import tqdm
from dataclasses import dataclass
from .data_aug import (
    VoxelPatchRotationDataAugmentationMixin,
    VoxelGridOperationMixin,
    AtomicDataCollator,
)
from .utils import (
    pad_array,
    pad_3d_array,
    pad_3d_tensor,
    initialize_grid_space,
    round_half_up,
)


def infer_grid_space_from_path(path: str) -> float:
    if "bench_full_0.35_abs" in path or "bench_full_abs_cpu_035" in path:
        return 0.35
    if "bench_full_0.55_abs" in path or "bench_full_abs_cpu_055" in path:
        return 0.55
    if "bench_full_abs_cpu_075" in path:
        return 0.75
    if "bench_full_abs_cpu_095" in path:
        return 0.95
    raise ValueError(
        "Unknown grid space. Provide `grid_space` in the input file or include known "
        "dataset naming in the path."
    )


def infer_ionic_conc_from_path(path: str) -> float:
    """Infer ionic concentration (mol/L) from `_c0p15` in filename or `salt0p15` in path."""
    base = os.path.basename(path)
    match = re.search(r"_c(\d+)p(\d+)(?:_|\.|$)", base)
    if not match:
        normalized_path = path.replace("\\", "/").lower()
        match = re.search(r"(?:^|/)salt(\d+)p(\d+)(?:/|$)", normalized_path)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    return 0.0


def extract_ionic_conc(data_dict: dict, patch_path: str) -> float:
    for key in ("ionic_conc", "ion_conc", "conc"):
        if key in data_dict:
            value = np.asarray(data_dict[key]).reshape(-1)
            if value.size > 0:
                return float(value[0])
    return infer_ionic_conc_from_path(patch_path)


def ensure_dhwc(array: np.ndarray, feature_name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 3:
        return array[..., None]
    if array.ndim == 4:
        return array
    raise ValueError(
        f"Feature `{feature_name}` must be 3D or 4D (DHW or DHWC), got shape {array.shape}"
    )


def load_dense_voxel_features(patch_path: str) -> tuple[dict[str, np.ndarray], float]:
    required_keys = (
        "level_set",
        "atom_charge",
        "atom_type",
        "atom_mask",
        "atom_potential",
    )

    def build_from_xy_schema(
        x: np.ndarray, y: np.ndarray, data_dict: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        # x can be (C,D,H,W) or (D,H,W,C). We map x[0]/x[...,0] -> level_set(vdw),
        # x[1]/x[...,1] -> atom_charge, and y -> atom_potential target.
        x = np.asarray(x)
        y = np.asarray(y)

        if x.ndim != 4:
            raise ValueError(
                f"`x` in {patch_path} must be 4D with 2 channels, got shape {x.shape}"
            )
        if x.shape[0] >= 2:
            level_set = x[0]
            atom_charge = x[1]
        elif x.shape[-1] >= 2:
            level_set = x[..., 0]
            atom_charge = x[..., 1]
        else:
            raise ValueError(
                f"`x` in {patch_path} must contain at least 2 channels, got shape {x.shape}"
            )

        if y.ndim == 4 and y.shape[0] == 1:
            atom_potential = y[0]
        elif y.ndim == 3:
            atom_potential = y
        else:
            raise ValueError(
                f"`y` in {patch_path} must be (1,D,H,W) or (D,H,W), got shape {y.shape}"
            )

        atom_mask = data_dict.get("atom_mask")
        atom_type = data_dict.get("atom_type")
        if atom_mask is None:
            atom_mask = (np.abs(atom_charge) > 1e-12).astype(np.float32)
        if atom_type is None:
            atom_type = atom_mask

        return {
            "level_set": ensure_dhwc(level_set, "level_set").astype(np.float32, copy=False),
            "atom_charge": ensure_dhwc(atom_charge, "atom_charge").astype(
                np.float32, copy=False
            ),
            "atom_type": ensure_dhwc(atom_type, "atom_type").astype(
                np.float32, copy=False
            ),
            "atom_mask": ensure_dhwc(atom_mask, "atom_mask").astype(
                np.float32, copy=False
            ),
            "atom_potential": ensure_dhwc(atom_potential, "atom_potential").astype(
                np.float32, copy=False
            ),
        }

    def build_feature_dict_from_dict(
        data_dict: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        if all(key in data_dict for key in required_keys):
            feat_dict = {}
            for key in required_keys:
                feat_dict[key] = ensure_dhwc(data_dict[key], key).astype(
                    np.float32, copy=False
                )
            return feat_dict

        # Common APBS naming schema
        if all(key in data_dict for key in ("pot", "charge", "vdw")):
            level_set = data_dict["vdw"]
            atom_charge = data_dict["charge"]
            atom_potential = data_dict["pot"]
            atom_mask = data_dict.get("atom_mask")
            atom_type = data_dict.get("atom_type")
            if atom_mask is None:
                atom_mask = (np.abs(atom_charge) > 1e-12).astype(np.float32)
            if atom_type is None:
                atom_type = atom_mask
            return {
                "level_set": ensure_dhwc(level_set, "level_set").astype(
                    np.float32, copy=False
                ),
                "atom_charge": ensure_dhwc(atom_charge, "atom_charge").astype(
                    np.float32, copy=False
                ),
                "atom_type": ensure_dhwc(atom_type, "atom_type").astype(
                    np.float32, copy=False
                ),
                "atom_mask": ensure_dhwc(atom_mask, "atom_mask").astype(
                    np.float32, copy=False
                ),
                "atom_potential": ensure_dhwc(atom_potential, "atom_potential").astype(
                    np.float32, copy=False
                ),
            }

        if "x" in data_dict and "y" in data_dict:
            return build_from_xy_schema(data_dict["x"], data_dict["y"], data_dict)

        raise KeyError(
            "Unsupported dense feature schema in file "
            f"{patch_path}. Expected keys {required_keys}, "
            "or (pot,charge,vdw), or (x,y)."
        )

    if patch_path.endswith(".npz"):
        with np.load(patch_path) as npz_data:
            npz_dict = {k: npz_data[k] for k in npz_data.files}
            feat_dict = build_feature_dict_from_dict(npz_dict)
            feat_dict["ionic_conc"] = np.array(
                extract_ionic_conc(npz_dict, patch_path), dtype=np.float32
            )
            grid_space = (
                float(npz_data["grid_space"])
                if "grid_space" in npz_data
                else infer_grid_space_from_path(patch_path)
            )
        return feat_dict, grid_space

    with gzip.open(patch_path, "rb") as f:
        patch = pickle.load(f)

    feat_dict = patch["feat_info"] if "feat_info" in patch else patch
    normalized_feat_dict = build_feature_dict_from_dict(feat_dict)
    normalized_feat_dict["ionic_conc"] = np.array(
        extract_ionic_conc(feat_dict, patch_path), dtype=np.float32
    )

    grid_space = (
        float(feat_dict["grid_space"])
        if "grid_space" in feat_dict
        else infer_grid_space_from_path(patch_path)
    )
    return normalized_feat_dict, grid_space


class LabelTransformer:
    """
    Normalize label value to range -1 to 1
    """

    def __init__(self, probe_radius_upperbound: float, probe_radius_lowerbound: float):
        self.upperbound = probe_radius_upperbound
        self.lowerbound = probe_radius_lowerbound

    def transform(self, x: torch.Tensor):
        clamp_x = torch.clamp_(x, min=self.lowerbound, max=self.upperbound)
        trans_x = (clamp_x - self.lowerbound) / (
            self.upperbound - self.lowerbound
        ) * 2 - 1
        return trans_x

    def inv_transform(self, x: torch.Tensor):
        inv_x = (x + 1) / 2 * (self.upperbound - self.lowerbound) + self.lowerbound
        return inv_x

    @property
    def sign_threshold(self):
        return self.transform(torch.zeros(1)).item()


class ExpLabelTransformer(LabelTransformer):
    """
    Normalize label value to range -1 to 1
    """

    def __init__(
        self,
        probe_radius_upperbound: float,
        probe_radius_lowerbound: float,
        offset: float,
    ):
        self.upperbound = probe_radius_upperbound
        self.lowerbound = probe_radius_lowerbound
        self.offset = offset

        self.exp_upperbound = math.exp(probe_radius_upperbound + self.offset)
        self.exp_lowerbound = math.exp(probe_radius_lowerbound + self.offset)

    def transform(self, x: torch.Tensor):
        clamp_x = torch.clamp_(x, min=self.lowerbound, max=self.upperbound)
        trans_x = torch.exp(clamp_x + self.offset)
        return trans_x

    def inv_transform(self, x: torch.Tensor):
        clamp_x = torch.clamp(x, min=self.exp_lowerbound, max=self.exp_upperbound)
        inv_x = torch.log(clamp_x) - self.offset
        return inv_x


class BipartScaleTransformer(LabelTransformer):
    """
    Normalize label value to range -1 to 1
    """

    def __init__(self, probe_radius_upperbound: float, probe_radius_lowerbound: float):
        self.upperbound = probe_radius_upperbound
        self.lowerbound = probe_radius_lowerbound

    def transform(self, x: torch.Tensor):
        clamp_x = torch.clamp_(x, min=self.lowerbound, max=self.upperbound)
        trans_x = torch.where(
            clamp_x > 0, clamp_x / abs(self.upperbound), clamp_x / abs(self.lowerbound)
        )
        return trans_x

    def inv_transform(self, x: torch.Tensor):
        inv_x = torch.where(x > 0, x * abs(self.upperbound), x * abs(self.lowerbound))
        return inv_x


class TranslationLabelTransformer(LabelTransformer):
    """
    Normalize label value to range -1 to 1
    """

    def __init__(
        self,
        probe_radius_upperbound: float,
        probe_radius_lowerbound: float,
        do_truncate: bool = True,
    ):
        self.upperbound = probe_radius_upperbound
        self.lowerbound = probe_radius_lowerbound
        self.do_truncate = do_truncate

    def transform(self, x: torch.Tensor):
        if self.do_truncate:
            x = torch.clamp_(x, min=self.lowerbound, max=self.upperbound)
        trans_x = x - self.lowerbound
        return trans_x

    def inv_transform(self, x: torch.Tensor):
        inv_x = x + self.lowerbound
        return inv_x


class RefinedMlsesDataset(Dataset):
    """
    Load all data in dat files into the memory. Comsume too much memory.
    """

    def __init__(self, path, input_dim=200):
        self.path = path
        self.input_dim = input_dim
        self.dat_files = glob.glob(f"{path}/*/*.dat")

        print("Number of data files loading:", len(self.dat_files))

        self.labels = []
        self.features = []
        self.features_length = []
        for file in tqdm(self.dat_files):
            with open(file, "r") as f:
                for line in f:
                    row = line.split()

                    # for each sample, we have at least four elements
                    if len(row) > 3:
                        self.labels.append(float(row[0]))
                        feature = [float(i) for i in row[1:]]
                        self.features.append(feature)
                        self.features_length.append(len(feature))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        feature = self.features[index]
        feature_length = self.features_length[index]

        feature_array = np.array(feature, dtype=np.float32)
        feature_array = np.pad(
            feature_array, (0, self.input_dim - feature_length), "constant"
        )

        label_tensor = torch.LongTensor(
            [
                label,
            ]
        )
        feature_tensor = torch.from_numpy(feature_array)

        return feature_tensor, label_tensor


class RefinedMlsesMapDataset(Dataset):
    """
    Record entries offset inside the data file.
    """

    def __init__(
        self,
        dat_files,
        input_dim=200,
        label_transformer: LabelTransformer = None,
        label_only=False,
        dummy_columns=0,
    ):
        self.input_dim = input_dim
        self.dat_files = dat_files
        self.label_transformer = label_transformer
        self.label_only = label_only
        self.index_byte = 8
        self.dummy_columns = dummy_columns

        print("Number of data files loading:", len(self.dat_files))

        self.files_length = []
        for file in tqdm(self.dat_files):
            # check if offset index file has built
            index_file = self.__get_index_file(file)
            if os.path.isfile(index_file):
                with open(index_file, "rb") as f:
                    head_line = f.read(self.index_byte)
                    total_line = int.from_bytes(head_line, "big")
                    self.files_length.append(total_line)
            else:
                # build offset index file
                with open(file, "r") as f:
                    offset = 0
                    file_offset = []
                    for line in f:
                        file_offset.append(offset)
                        offset += len(line)

                total_line = len(file_offset)
                self.files_length.append(total_line)

                with open(index_file, "wb") as f:
                    f.write(total_line.to_bytes(self.index_byte, "big"))
                    for line_offset in file_offset:
                        f.write(line_offset.to_bytes(self.index_byte, "big"))

        # build offset indexing
        self.files_cumsum = np.cumsum(self.files_length)
        print("Total line:", self.files_cumsum[-1])

    def __len__(self):
        return self.files_cumsum[-1].item()

    def __get_index_file(self, file):
        return f"{file}.offset_index"

    def get_file_name_by_index(self, index):
        file_index = np.searchsorted(self.files_cumsum, index, side="right")
        file_name = self.dat_files[file_index]
        infile_offset = index - (
            0 if file_index == 0 else self.files_cumsum[file_index - 1].item()
        )

        # get index file and read offset
        index_file = self.__get_index_file(file_name)
        with open(index_file, "rb") as f:
            f.seek(self.index_byte * (infile_offset + 1))  # +1 for skip head line
            line_offset = f.read(self.index_byte)
        file_offset = int.from_bytes(line_offset, "big")
        return file_name, file_offset

    def __getitem__(self, index):
        # read line by offset
        file_name, file_offset = self.get_file_name_by_index(index)
        with open(file_name, "r") as f:
            f.seek(file_offset)
            line = f.readline()

        # process line data
        if not self.label_only:
            row = line.split()
            label = float(row[0])
            # feature_num = math.log(len(row) - 1)
            feature = [
                float(i) for i in row[self.dummy_columns + 1 :]
            ]  # here we skip dummy columns

            # TODO: here is an extra colum for some training data files. Need to be fixed in the future.
            if len(feature) % 4 != 0:
                feature = feature[1:]
            assert len(feature) % 4 == 0, (
                f"Feature len ({len(feature)}) cannot divided by 4."
            )

            # truncate feature
            feature = feature[: self.input_dim]

            feature_array = np.array(feature, dtype=np.float32, copy=False)
            feature_array = np.pad(
                feature_array, (0, self.input_dim - len(feature)), "constant"
            )

            label_tensor = torch.FloatTensor(
                [
                    label,
                ]
            )
            feature_tensor = torch.from_numpy(feature_array)
        else:
            row = line.split()
            label = float(row[0])
            label_tensor = torch.FloatTensor(
                [
                    label,
                ]
            )
            feature_num_tensor = torch.LongTensor(
                [
                    len(row) - 1,
                ]
            )

        # do label transform if needed
        if self.label_transformer is not None:
            label_tensor = self.label_transformer.transform(label_tensor)

        if self.label_only:
            return feature_num_tensor, label_tensor
        else:
            return feature_tensor, label_tensor


class Subset(Dataset):
    def __init__(self, dataset: RefinedMlsesMapDataset, indices_path, split) -> None:
        self.dataset = dataset
        self.indices_file = f"{indices_path}/{split}.indices"

        self.length = 0
        with open(self.indices_file, "rb") as f:
            head_line = f.read(self.dataset.index_byte)
            total_line = int.from_bytes(head_line, "big")
            self.length = total_line

        print(f"{split} size:", self.length)

    def __getitem__(self, idx):
        with open(self.indices_file, "rb") as f:
            f.seek(self.dataset.index_byte * (idx + 1))  # +1 for skip head line
            indice_bytes = f.read(self.dataset.index_byte)
        indice = int.from_bytes(indice_bytes, "big")
        return self.dataset[indice]

    def __len__(self):
        return self.length


class RefinedMlsesMemoryMapDataset(Dataset):
    """
    Dataset from Numpy memmap
    """

    def __init__(
        self,
        dat_file,
        input_dim,
        sample_num,
        sample_dim,
        label_transformer: LabelTransformer = None,
        label_only=False,
    ):
        self.input_dim = input_dim
        self.sample_num = sample_num
        self.sample_dim = sample_dim
        self.dat_file = dat_file
        self.label_transformer = label_transformer
        self.label_only = label_only

        self.np_map = np.memmap(
            self.dat_file, dtype=np.float32, mode="r+", shape=(sample_num, sample_dim)
        )
        print(
            "Sample Num:",
            sample_num,
            "Sample Dim:",
            sample_dim,
            "Feature Size:",
            input_dim,
        )

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        # read sample from memory map
        sample_row = self.np_map[index]
        feature_array = sample_row[: self.input_dim]
        label_array = sample_row[self.input_dim :]

        # process line data
        if not self.label_only:
            label_tensor = torch.from_numpy(label_array)
            feature_tensor = torch.from_numpy(feature_array)
        else:
            label_tensor = torch.from_numpy(label_array)
            feature_tensor = None

        # do label transform if needed
        if self.label_transformer is not None:
            label_tensor = self.label_transformer.transform(label_tensor)

        if self.label_only:
            return label_tensor
        else:
            return feature_tensor, label_tensor


class MultitaskRefinedMlsesMapDataset(RefinedMlsesMapDataset):
    def __init__(
        self,
        dat_files,
        input_dim=200,
        lowerbound=-1.0,
        label_transformer: LabelTransformer = None,
        label_only=False,
    ):
        super().__init__(dat_files, input_dim, label_transformer, label_only)
        self.input_dim = input_dim
        self.lowerbound = lowerbound

    def __getitem__(self, index):
        # read line by offset
        file_name, file_offset = self.get_file_name_by_index(index)
        with open(file_name, "r") as f:
            f.seek(file_offset)
            line = f.readline()

        # process line data
        if not self.label_only:
            row = line.split()
            label = float(row[0])
            feature = [float(i) for i in row[1:]]

            # add the length as the first feature
            feature = [len(feature)] + feature

            # truncate feature
            feature = feature[: self.input_dim]

            feature_array = np.array(feature, dtype=np.float32)
            feature_array = np.pad(
                feature_array, (0, self.input_dim - len(feature)), "constant"
            )

            reg_label_tensor = torch.FloatTensor(
                [
                    label,
                ]
            )
            cls_label_tensor = torch.LongTensor(
                [
                    1 if label < self.lowerbound else -1,
                ]
            )  # 1 for trivial samples, -1 for nontrivial samples
            feature_tensor = torch.from_numpy(feature_array)
        else:
            row = line.split()
            label = float(row[0])
            reg_label_tensor = torch.FloatTensor(
                [
                    label,
                ]
            )
            feature_tensor = None

        # do label transform if needed
        if self.label_transformer is not None:
            reg_label_tensor = self.label_transformer.transform(reg_label_tensor)

        if self.label_only:
            return reg_label_tensor
        else:
            return feature_tensor, reg_label_tensor, cls_label_tensor


class MultitaskRefinedMlsesMemoryMapDataset(RefinedMlsesMemoryMapDataset):
    def __init__(
        self,
        dat_file,
        input_dim,
        sample_num,
        sample_dim,
        lowerbound=-1.0,
        label_transformer: LabelTransformer = None,
        label_only=False,
    ):
        super().__init__(dat_file, input_dim, sample_num, sample_dim, None, label_only)
        self.input_dim = input_dim
        self.multitask_label_transformer = label_transformer
        self.sample_num = sample_num
        self.sample_dim = sample_dim
        self.lowerbound = lowerbound

    def __getitem__(self, index):
        if self.label_only:
            return super().__getitem__(index)
        else:
            feature_tensor, label_tensor = super().__getitem__(index)

            if self.multitask_label_transformer is not None:
                reg_label_tensor = self.multitask_label_transformer.transform(
                    label_tensor
                )
            else:
                reg_label_tensor = label_tensor

            cls_label_tensor = torch.LongTensor(
                [
                    1 if label_tensor.item() < self.lowerbound else -1,
                ]
            )  # 1 for trivial samples, -1 for nontrivial samples
            return feature_tensor, reg_label_tensor, cls_label_tensor


class ImageMlsesDataset(Dataset):
    def __init__(
        self,
        patches_index_file,
        patch_size: int,
        label_transformer: LabelTransformer = None,
    ) -> None:
        super().__init__()
        with open(patches_index_file, "r") as f:
            patches_index = json.load(f)

        self.patches_index = np.array(
            [[p["image_path"], p["x0"], p["y0"]] for p in patches_index]
        )
        self.patch_size = patch_size
        self.label_transformer = label_transformer

    def __len__(self):
        return len(self.patches_index)

    def __getitem__(self, index):
        patch = self.patches_index[index]
        image_npz = np.load(patch[0])
        x0 = int(patch[1])
        y0 = int(patch[2])

        pad_image = pad_array(image_npz["image"], x0, y0, self.patch_size)
        pad_label = pad_array(image_npz["label"], x0, y0, self.patch_size)
        pad_mask = pad_array(image_npz["mask"], x0, y0, self.patch_size)

        # prepare tensors and transform them from NHWC to NCHW
        image_tensor = torch.from_numpy(pad_image).permute(2, 0, 1)
        label_tensor = torch.from_numpy(pad_label).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(pad_mask).permute(2, 0, 1)

        # do label transform if needed
        if self.label_transformer is not None:
            label_tensor = self.label_transformer.transform(label_tensor)
        return image_tensor, label_tensor, mask_tensor


class Image3dMlsesDataset(Dataset):
    def __init__(
        self,
        patches_index_file,
        patch_size: int,
        label_transformer: LabelTransformer = None,
    ) -> None:
        super().__init__()
        with open(patches_index_file, "r") as f:
            patches_index = json.load(f)

        self.patches_index = np.array(
            [[p["patch_file_path"], p["x0"], p["y0"], p["z0"]] for p in patches_index]
        )
        self.patch_size = patch_size
        self.label_transformer = label_transformer

    def __len__(self):
        return len(self.patches_index)

    def __getitem__(self, index):
        patch = self.patches_index[index]
        with gzip.GzipFile(patch[0], "rb") as f:
            image_npz = pickle.load(f)

        pad_image = pad_3d_array(image_npz["image"], 0, 0, 0, self.patch_size)
        pad_label = pad_3d_array(image_npz["label"], 0, 0, 0, self.patch_size)
        pad_mask = pad_3d_array(image_npz["mask"], 0, 0, 0, self.patch_size)

        # prepare tensors and transform them from DHWC to CDHW
        dim_inds = (3, 0, 1, 2)
        image_tensor = torch.from_numpy(pad_image).permute(*dim_inds)
        label_tensor = torch.from_numpy(pad_label).permute(*dim_inds)
        mask_tensor = torch.from_numpy(pad_mask).permute(*dim_inds)

        # do label transform if needed
        if self.label_transformer is not None:
            label_tensor = self.label_transformer.transform(label_tensor)
        return image_tensor, label_tensor, mask_tensor


class Image3dEnergyDataset(Dataset):
    def __init__(
        self,
        patch_pkls: list[str],
        patch_size: int,
        epb_mean: float,
        epb_std: float,
    ) -> None:
        super().__init__()
        self.patch_pkls = patch_pkls
        self.patch_size = patch_size
        self.epb_mean = epb_mean
        self.epb_std = epb_std

    def __len__(self):
        return len(self.patch_pkls)

    def process_one_patch(self, patch_path):
        feat_dict, grid_space = load_dense_voxel_features(patch_path)
        ionic_conc = float(feat_dict.get("ionic_conc", 0.0))
        pad_level_set = pad_3d_array(feat_dict["level_set"], 0, 0, 0, self.patch_size)
        pad_atom_charge = pad_3d_array(
            feat_dict["atom_charge"], 0, 0, 0, self.patch_size
        )
        pad_atom_type = pad_3d_array(feat_dict["atom_type"], 0, 0, 0, self.patch_size)
        pad_atom_mask = pad_3d_array(feat_dict["atom_mask"], 0, 0, 0, self.patch_size)
        pad_potential = pad_3d_array(
            feat_dict["atom_potential"], 0, 0, 0, self.patch_size
        )

        # prepare tensors and transform them from DHWC to CDHW
        dim_inds = (3, 0, 1, 2)
        level_set_tensor = torch.from_numpy(pad_level_set).permute(*dim_inds)
        atom_charge_tensor = torch.from_numpy(pad_atom_charge).permute(*dim_inds)
        atom_type_tensor = torch.from_numpy(pad_atom_type).permute(*dim_inds)
        atom_mask_tensor = torch.from_numpy(pad_atom_mask).permute(*dim_inds)
        potential_tensor = torch.from_numpy(pad_potential).permute(*dim_inds)
        grid_info = np.array([grid_space, ionic_conc], dtype=np.float32)
        return (
            level_set_tensor,
            atom_charge_tensor,
            atom_type_tensor,
            atom_mask_tensor,
            potential_tensor,
            grid_info,
        )

    def __getitem__(self, index):
        patch_file = self.patch_pkls[index]
        outputs = self.process_one_patch(patch_file)
        return outputs


@dataclass
class VoxelImage3dEnergyDatasetExtraConfig:
    from_pdb: bool = False
    grid_space_build_from_pdb: float = 0.5
    fill_ratio_build_from_pdb: float = 2
    context_size: int = 0

    # atom neighbor pairs
    process_atom_neighbor_pairs: bool = False
    neighbor_list_cutoff: float = 10.0
    max_num_neighbors: Union[int, None] = None
    natom_threshold_for_max_num_neighbors: int = 50000

    # atom2level-set pairs
    select_nearest_level_set: bool = False
    select_nearest_level_set_k: int = 6

    # boundary positions
    create_boundary_features: bool = False
    boundary_cutoff: float = 1
    boundary_unit: float = 0.5


class VoxelImage3dEnergyDataset(Image3dEnergyDataset):
    def __init__(
        self,
        patch_pkls: list[str],
        patch_size: int,
        epb_mean: float,
        epb_std: float,
        do_random_crop: bool = False,
        random_crop_atom_num: int = 10,
        random_crop_interval: int = 32,
        do_random_rotate: bool = False,
        random_rotate_interval: int = 10,
        given_rotate_angle: Union[List[int], None] = None,
        given_rotate_axis: Union[List[str], None] = None,
        do_fixed_rotate: bool = False,
        rotate_k: int = 1,
        rotate_axis: list[int] = [0, 1],
        do_voxel_grids_shrinking: bool = False,
        do_random_grid_scaling: bool = False,
        random_grid_scaling_left: float = 0.15,
        random_grid_scaling_right: float = 1.0,
        random_grid_scaling_interval: float = 0.05,
        given_grid_scaling_size: Union[float, None] = None,
        extra_config: VoxelImage3dEnergyDatasetExtraConfig = VoxelImage3dEnergyDatasetExtraConfig(),
    ) -> None:
        super().__init__(patch_pkls, patch_size, epb_mean, epb_std)
        self.do_random_crop = do_random_crop
        self.random_crop_atom_num = random_crop_atom_num
        self.random_crop_interval = random_crop_interval

        if do_fixed_rotate and do_random_rotate:
            raise ValueError("Cannot do fixed and random rotate at the same time")
        self.do_random_rotate = do_random_rotate
        self.random_rotate_interval = random_rotate_interval
        self.given_rotate_angle = given_rotate_angle
        self.given_rotate_axis = given_rotate_axis
        self.do_fixed_rotate = do_fixed_rotate
        self.rotate_k = rotate_k
        self.rotate_axis = rotate_axis
        self.do_voxel_grids_shrinking = do_voxel_grids_shrinking

        # random scaling
        self.do_random_grid_scaling = do_random_grid_scaling
        self.random_grid_scaling_left = random_grid_scaling_left
        self.random_grid_scaling_right = random_grid_scaling_right
        self.random_grid_scaling_interval = random_grid_scaling_interval
        self.given_grid_scaling_size = given_grid_scaling_size

        # extra config
        self.extra_config = extra_config

    def get_anchor_point(self, D, H, W, atom_mask: np.ndarray):
        # sample anchor point
        total_atom_num = atom_mask.sum().item()
        if self.do_random_crop:
            while True:
                anchor_D = torch.randint(0, D, (1,)).item()
                anchor_H = torch.randint(0, H, (1,)).item()
                anchor_W = torch.randint(0, W, (1,)).item()

                crop_atom_mask = pad_3d_array(
                    atom_mask,
                    anchor_D,
                    anchor_H,
                    anchor_W,
                    self.patch_size,
                )

                # if there are less than min(random_crop_atom_num, total_atom_num // 2) atoms in the crop, try again
                if crop_atom_mask.sum() > min(
                    self.random_crop_atom_num, total_atom_num // 2
                ):
                    break
        else:
            # try to focus on most atoms
            mid_D = D // 2
            mid_H = H // 2
            mid_W = W // 2

            anchor_D = max(mid_D - self.patch_size // 2, 0)
            anchor_H = max(mid_H - self.patch_size // 2, 0)
            anchor_W = max(mid_W - self.patch_size // 2, 0)
        return anchor_D, anchor_H, anchor_W

    def process_one_patch(self, patch_path):
        feat_dict, grid_space = load_dense_voxel_features(patch_path)
        ionic_conc = float(feat_dict.get("ionic_conc", 0.0))

        # do random crop if needed
        anchor_D, anchor_H, anchor_W = self.get_anchor_point(
            *feat_dict["level_set"].shape[:-1], feat_dict["atom_mask"]
        )

        pad_level_set = pad_3d_array(
            feat_dict["level_set"], anchor_D, anchor_H, anchor_W, self.patch_size
        )
        pad_atom_charge = pad_3d_array(
            feat_dict["atom_charge"], anchor_D, anchor_H, anchor_W, self.patch_size
        )
        pad_atom_type = pad_3d_array(
            feat_dict["atom_type"], anchor_D, anchor_H, anchor_W, self.patch_size
        )
        pad_atom_mask = pad_3d_array(
            feat_dict["atom_mask"], anchor_D, anchor_H, anchor_W, self.patch_size
        )
        pad_potential = pad_3d_array(
            feat_dict["atom_potential"], anchor_D, anchor_H, anchor_W, self.patch_size
        )

        # prepare tensors and transform them from DHWC to CDHW
        dim_inds = (3, 0, 1, 2)
        level_set_tensor = torch.from_numpy(pad_level_set).permute(*dim_inds)
        atom_charge_tensor = torch.from_numpy(pad_atom_charge).permute(*dim_inds)
        atom_type_tensor = torch.from_numpy(pad_atom_type).permute(*dim_inds)
        atom_mask_tensor = torch.from_numpy(pad_atom_mask).permute(*dim_inds)
        potential_tensor = torch.from_numpy(pad_potential).permute(*dim_inds)
        grid_info = np.array([grid_space, ionic_conc], dtype=np.float32)
        return (
            level_set_tensor,
            atom_charge_tensor,
            atom_type_tensor,
            atom_mask_tensor,
            potential_tensor,
            grid_info,
        )


class VoxelImage3dEnergySparseDataset(
    VoxelImage3dEnergyDataset,
    VoxelPatchRotationDataAugmentationMixin,
    VoxelGridOperationMixin,
):
    @property
    def atom_type_map(self):
        return {
            "PAD": 0,
            "UNK": 1,
        }

    @property
    def context_patch_size(self) -> int:
        """the context path is constructed by adding context_size to the patch in both directions"""
        return self.extra_config.context_size * 2 + self.patch_size

    def get_indice_value_pairs_in_context_patch(
        self,
        indices: np.ndarray,
        values: np.ndarray,
        size: tuple[int],
        anchor_offset: tuple[int] = (0, 0, 0),
        load_all_atoms: bool = False,
        return_selected_indices: bool = False,
    ):
        """Return the indices and values in the context based patch of molecule grid"""
        lower_bound = np.array(anchor_offset)
        indices = indices - lower_bound[None, :]
        if load_all_atoms:
            selected_indices = np.all(indices >= 0, axis=1)

            # update indices, values, and size
            indices = indices[selected_indices]
            values = values[selected_indices]
            size = (np.array(size) - lower_bound).tolist()
        else:
            # only load a local patch of molecule
            selected_indices = np.all(indices >= 0, axis=1) & np.all(
                indices < self.context_patch_size, axis=1
            )

            # update indices, values, and size
            indices = indices[selected_indices]
            values = values[selected_indices]
            size = np.minimum(
                np.array(size) - lower_bound,
                np.array((self.context_patch_size,) * 3),
            ).tolist()

        if return_selected_indices:
            return indices, values, size, selected_indices
        else:
            return indices, values, size

    def create_dense_feature(
        self,
        indices: np.ndarray,
        values: np.ndarray,
        size: tuple[int],
        map_dict: dict[str, int] = None,
        anchor_offset: tuple[int] = (0, 0, 0),
        load_all_atoms: bool = False,
        dtype=torch.float32,
    ):
        """load dense feature from a sparse representation

        :param indices: value indices [N, 3]
        :param values: values
        :param size: the original full size of the molecule grid from amber
        :param map_dict: whether to map values, defaults to None
        :param anchor_offset: whether load a local patch of molecule, defaults to (0, 0, 0)
        :return: the loaded patch of molecule grid, if anchor_offset is not (0, 0, 0), the size of the output will be different
        """
        assert len(indices) == len(values), "Indices and values should have same length"
        assert indices.shape[1] == len(size), "Indices should have same length as size"

        indices, values, size = self.get_indice_value_pairs_in_context_patch(
            indices,
            values,
            size,
            anchor_offset,
            load_all_atoms,
        )

        # quick mapping from dict: https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
        if map_dict is not None:
            u, inv = np.unique(values, return_inverse=True)
            values = np.array([map_dict.get(x, map_dict["UNK"]) for x in u])[
                inv
            ].reshape(values.shape)

        sparse_features = torch.sparse_coo_tensor(indices.T, values, size, dtype=dtype)
        return sparse_features.to_dense()

    def sample_with_interval(self, start, end, interval, num_samples):
        # Create the range with the specified interval
        range_tensor = torch.arange(start, end, interval)

        # Sample indices from the range tensor
        indices = torch.randint(0, len(range_tensor), (num_samples,))

        # Gather the sampled values
        samples = range_tensor[indices]

        return samples

    def get_atom_num_from_dense(
        self,
        atom_mask: torch.Tensor,
        anchor_D: int = 0,
        anchor_H: int = 0,
        anchor_W: int = 0,
    ):
        return (
            atom_mask[
                anchor_D : anchor_D + self.patch_size,
                anchor_H : anchor_H + self.patch_size,
                anchor_W : anchor_W + self.patch_size,
            ]
            .sum()
            .item()
        )

    def get_atom_num_from_sparse(
        self,
        atom_indices: np.ndarray,
        anchor_D: int = 0,
        anchor_H: int = 0,
        anchor_W: int = 0,
    ):
        lower_bound = np.array([anchor_D, anchor_H, anchor_W])[None, :]
        indices = atom_indices - lower_bound
        in_range_mask: np.ndarray = np.all((indices >= 0), axis=1) & np.all(
            (indices < self.patch_size), axis=1
        )
        return in_range_mask.sum().item()

    def get_atom_num_in_patch(
        self,
        atom_mask: Union[np.ndarray, None] = None,
        atom_indices: Union[np.ndarray, None] = None,
        anchor_D: int = 0,
        anchor_H: int = 0,
        anchor_W: int = 0,
    ):
        if atom_mask is not None:
            return self.get_atom_num_from_dense(atom_mask, anchor_D, anchor_H, anchor_W)
        elif atom_indices is not None:
            return self.get_atom_num_from_sparse(
                atom_indices, anchor_D, anchor_H, anchor_W
            )
        else:
            raise ValueError("atom_mask or atom_indices should be provided")

    def get_middle_anchor_in_grid(self, D, H, W):
        mid_D = D // 2
        mid_H = H // 2
        mid_W = W // 2

        middle_anchor_D = max(mid_D - self.patch_size // 2, 0)
        middle_anchor_H = max(mid_H - self.patch_size // 2, 0)
        middle_anchor_W = max(mid_W - self.patch_size // 2, 0)
        return middle_anchor_D, middle_anchor_H, middle_anchor_W

    def get_anchor_point(
        self,
        D,
        H,
        W,
        atom_mask: torch.Tensor = None,
        atom_indices: np.ndarray = None,
    ):
        # try to focus on most atoms
        middle_anchor_D, middle_anchor_H, middle_anchor_W = (
            self.get_middle_anchor_in_grid(D, H, W)
        )
        if self.do_random_crop:
            # middle_atom_num for reference
            middle_atom_num = self.get_atom_num_in_patch(
                atom_mask,
                atom_indices,
                middle_anchor_D,
                middle_anchor_H,
                middle_anchor_W,
            )
            while True:
                anchor_D = self.sample_with_interval(
                    max(middle_anchor_D - 3 * self.patch_size, 0),
                    min(middle_anchor_D + 3 * self.patch_size, D),
                    self.random_crop_interval,
                    1,
                ).item()
                anchor_H = self.sample_with_interval(
                    max(middle_anchor_H - 3 * self.patch_size, 0),
                    min(middle_anchor_H + 3 * self.patch_size, H),
                    self.random_crop_interval,
                    1,
                ).item()
                anchor_W = self.sample_with_interval(
                    max(middle_anchor_W - 3 * self.patch_size, 0),
                    min(middle_anchor_W + 3 * self.patch_size, W),
                    self.random_crop_interval,
                    1,
                ).item()

                crop_atom_num = self.get_atom_num_in_patch(
                    atom_mask,
                    atom_indices,
                    anchor_D,
                    anchor_H,
                    anchor_W,
                )
                if isinstance(self.random_crop_atom_num, int):
                    # if there are less than min(random_crop_atom_num, middle_atom_num // 2) atoms in the crop, try again
                    if crop_atom_num > min(
                        self.random_crop_atom_num, middle_atom_num // 2
                    ):
                        break
                elif isinstance(self.random_crop_atom_num, float):
                    # if there are less than int(random_crop_atom_num * middle_atom_num) atoms in the crop, try again
                    if crop_atom_num > int(self.random_crop_atom_num * middle_atom_num):
                        break
                else:
                    raise ValueError("random_crop_atom_num should be int or float")
        else:
            anchor_D = middle_anchor_D
            anchor_H = middle_anchor_H
            anchor_W = middle_anchor_W
        return anchor_D, anchor_H, anchor_W

    def shrink_voxel_grid_if_needed(
        self,
        grid_dims: tuple[int],
        atom_indices: Union[np.ndarray, None] = None,
        *other_indices,
    ):
        """
        This operator is used to shrink the input voxel grids to the smallest bounding box containing all atoms. The reason is Amber would pad the molecule into a ratio, which would introduce a lot of zeros in the voxel grids. This operator is used to remove the zeros and make the input tensor more compact.
        """
        shrinked_grid_dims = grid_dims
        shrinked_atom_indices = atom_indices
        shrinked_other_indices = other_indices
        if self.do_voxel_grids_shrinking:
            if atom_indices is None:
                raise ValueError("Atom indices is required for shrinking")

            shrinked_grid_dims, anchor_offset = self.shrink_to_bounding_box(
                atom_indices
            )
            shrinked_anchor_offset = np.array(anchor_offset)[None, :]
            shrinked_atom_indices = atom_indices - shrinked_anchor_offset
            shrinked_other_indices = [
                other_index - shrinked_anchor_offset for other_index in other_indices
            ]

        return shrinked_grid_dims, shrinked_atom_indices, *shrinked_other_indices

    def rotate_voxel_grid_if_needed(
        self, grid_dims, atom_indices, *other_indices, patch_path="unknown"
    ):
        rotated_arrays = (atom_indices,) + other_indices
        rotated_grid_dims = grid_dims
        if self.do_fixed_rotate:
            raise NotImplementedError(
                "Fixed rotation is not implemented yet for indices"
            )
            # rotated_arrays = [
            #     np.rot90(t, self.rotate_k, self.rotate_axis) for t in other_indices
            # ]
            # rotated_grid_dims = rotated_arrays[0].shape[:3]
        elif self.do_random_rotate:
            # set angle and axis to the given values if provided
            # otherwise, randomly sample them
            angle = None
            axis = None
            if (
                self.given_rotate_angle is not None
                and self.given_rotate_axis is not None
            ):
                angle = self.given_rotate_angle
                axis = self.given_rotate_axis
            else:
                angle_values = np.arange(0, 360, self.random_rotate_interval)
                angle = float(np.random.choice(angle_values))
                axis = np.random.choice(["x", "y", "z"])

            # rotate with nearest interpolation
            if angle != 0:
                try:
                    rotated_grid_dims, rotated_arrays = self.rotate_voxel_indices(
                        grid_dims,
                        atom_indices,
                        other_indices=other_indices,
                        angle=angle,
                        axis=axis,
                    )
                except RuntimeError:
                    print(
                        f"RotationError: Skip this patch {patch_path} due to rotation error with angle {angle} and axis {axis}"
                    )
        return rotated_grid_dims, *rotated_arrays

    def build_grid_from_xyz(
        self,
        grid_space: float,
        fill_ratio: float,
        atom_xyz: np.ndarray,
        atom_rad: np.ndarray,
        other_xyz: list[np.ndarray],
    ):
        # we need to create a grid first
        grid_dims, grid_origin, atom_xyz = initialize_grid_space(
            atom_xyz, atom_rad, fill_ratio=fill_ratio, h=grid_space
        )

        atom_indices = round_half_up(atom_xyz).astype(int)
        other_indices = [
            VoxelGridOperationMixin.xyz2indices(xyz, grid_origin, grid_space)
            for xyz in other_xyz
        ]
        return (
            grid_dims.tolist(),
            grid_origin.tolist(),
            grid_space,
            atom_indices,
            other_indices,
        )

    def rotate_and_scale_xyz_if_needed(
        self,
        grid_dims: tuple[int],
        grid_origin: tuple[float],
        grid_space: float,
        atom_indices: np.ndarray,
        atom_rad: np.ndarray,
        other_indices: list[np.ndarray],
        patch_path="unknown",
    ):
        rotated_arrays = (atom_indices,) + tuple(other_indices)
        rotated_grid_dims = grid_dims
        rotated_grid_space = grid_space

        if self.do_fixed_rotate:
            raise NotImplementedError(
                "Fixed rotation is not implemented yet for xyz..."
            )

        if self.do_random_rotate or self.do_random_grid_scaling:
            # set angle and axis to the given values if provided
            # otherwise, randomly sample them
            angle = None
            axis = None
            scale_to_grid_space = None
            if (
                self.given_rotate_angle is not None
                and self.given_rotate_axis is not None
            ):
                angle = self.given_rotate_angle
                axis = self.given_rotate_axis
            else:
                angle_values = np.arange(0, 360, self.random_rotate_interval)
                angle = float(np.random.choice(angle_values))
                axis = np.random.choice(["x", "y", "z"])

            if self.do_random_grid_scaling:
                if self.given_grid_scaling_size is not None:
                    scale_to_grid_space = self.given_grid_scaling_size
                else:
                    scale_to_grid_space = round(
                        np.random.choice(
                            np.arange(
                                self.random_grid_scaling_left,
                                self.random_grid_scaling_right,
                                self.random_grid_scaling_interval,
                            )
                        ).item(),
                        2,
                    )
            else:
                scale_to_grid_space = grid_space

            # rotate and scale with nearest interpolation if needed
            try:
                rotated_grid_dims, rotated_grid_space, rotated_arrays = (
                    self.rotate_and_scale_voxel_xyz(
                        grid_dims,
                        grid_origin,
                        grid_space,
                        atom_indices,
                        atom_rad=atom_rad,
                        other_indices=other_indices,
                        scale_to_grid_space=scale_to_grid_space,
                        angle=angle,
                        axis=axis,
                    )
                )
            except RuntimeError:
                print(
                    f"RotationAndScaleError: Skip this patch {patch_path} due to rotation error with grid space {scale_to_grid_space}, angle {angle}, and axis {axis}"
                )
        return rotated_grid_dims, rotated_grid_space, *rotated_arrays

    def prepare_voxel_grid(
        self,
        feat_dict: Dict[str, Any],
        patch_path: str,
        from_pdb: bool = False,
    ):
        if from_pdb:
            atom_xyz = feat_dict["atom_xyz"]
            atom_rad = feat_dict["atom_rad"]
            level_set_xyz = feat_dict["level_set_xyz"]
            grid_dims, grid_origin, grid_space, atom_grid_loc, other_indices = (
                self.build_grid_from_xyz(
                    grid_space=self.extra_config.grid_space_build_from_pdb,
                    fill_ratio=self.extra_config.fill_ratio_build_from_pdb,
                    atom_xyz=atom_xyz,
                    atom_rad=atom_rad,
                    other_xyz=[level_set_xyz],
                )
            )
            level_set_grid_loc = other_indices[0]
        else:
            grid_dims = feat_dict["grid_dims"].tolist()
            grid_origin = feat_dict["grid_origin"].tolist()
            grid_space = feat_dict["grid_space"]
            atom_grid_loc = feat_dict["atom_grid_loc"]
            level_set_grid_loc = feat_dict["level_set_grid_loc"]

        # rotate atoms if needed
        grid_dims, grid_space, atom_grid_loc, level_set_grid_loc = (
            self.rotate_and_scale_xyz_if_needed(
                grid_dims,
                grid_origin,
                grid_space,
                atom_grid_loc,
                atom_rad=np.ones((len(atom_grid_loc))) * 2,
                other_indices=[level_set_grid_loc],
                patch_path=patch_path,
            )
        )

        # shrink voxel grids if neccessary
        grid_dims, atom_grid_loc, level_set_grid_loc = self.shrink_voxel_grid_if_needed(
            grid_dims,
            atom_grid_loc,
            level_set_grid_loc,
        )

        # drop invalid indices
        atom_grid_loc_mask = self.get_valid_indices_mask(atom_grid_loc, grid_dims)
        level_set_grid_loc_mask = self.get_valid_indices_mask(
            level_set_grid_loc, grid_dims
        )

        return (
            grid_origin,
            grid_dims,
            grid_space,
            atom_grid_loc,
            atom_grid_loc_mask,
            level_set_grid_loc,
            level_set_grid_loc_mask,
        )

    def get_context_based_anchor_point(self, anchor_offset: tuple[int]):
        """convert the anchor offset to the context based anchor offset"""
        anchor_offset_np = np.array(anchor_offset)
        context_anchor_offset = np.maximum(
            anchor_offset_np - self.extra_config.context_size, 0
        )
        patch_offset_in_context = anchor_offset_np - context_anchor_offset
        return tuple(context_anchor_offset), tuple(patch_offset_in_context)

    def prepare_voxel_patch(
        self,
        feat_dict: Dict[str, Any],
        patch_path: str,
        given_anchor_offset: Union[tuple[int], None] = None,
        load_all_atoms: bool = False,
    ):
        """create patch features from the given feat_dict

        feature dictionary must contain atom_charge, atom_type, level_set_value, and
            atom_xyz, atom_rad, level_set_xyz for pdb type datasets
            grid_dims, grid_origin, grid_space, atom_grid_loc, level_set_grid_loc for amber type datasets
        """
        (
            _,
            grid_dims,
            grid_space,
            atom_grid_loc,
            atom_grid_loc_mask,
            level_set_grid_loc,
            level_set_grid_loc_mask,
        ) = self.prepare_voxel_grid(
            feat_dict,
            patch_path,
            from_pdb=self.extra_config.from_pdb,
        )

        if given_anchor_offset is not None:
            anchor_offset, patch_offset_in_context = (
                self.get_context_based_anchor_point(given_anchor_offset)
            )
        else:
            # randomly crop if necessary
            anchor_D, anchor_H, anchor_W = self.get_anchor_point(
                *grid_dims, atom_indices=atom_grid_loc
            )
            random_crop_anchor_offset = (anchor_D, anchor_H, anchor_W)

            # load required voxel grid patch origined at anchor offset
            anchor_offset, patch_offset_in_context = (
                self.get_context_based_anchor_point(random_crop_anchor_offset)
            )

        # get required dense tensors
        valid_atom_grid_loc = atom_grid_loc[atom_grid_loc_mask]
        valid_level_set_grid_loc = level_set_grid_loc[level_set_grid_loc_mask]
        atom_mask = self.create_dense_feature(
            valid_atom_grid_loc,
            np.ones(len(valid_atom_grid_loc), dtype=np.float32),
            grid_dims,
            anchor_offset=anchor_offset,
            load_all_atoms=load_all_atoms,
        )  # this mask actually is the atom count at each grid location due to the sparce representation
        atom_charge = self.create_dense_feature(
            valid_atom_grid_loc,
            feat_dict["atom_charge"][atom_grid_loc_mask],
            grid_dims,
            anchor_offset=anchor_offset,
            load_all_atoms=load_all_atoms,
        )
        atom_type = self.create_dense_feature(
            valid_atom_grid_loc,
            feat_dict["atom_type"][atom_grid_loc_mask],
            grid_dims,
            self.atom_type_map,
            anchor_offset=anchor_offset,
            load_all_atoms=load_all_atoms,
        )
        level_set_mask = self.create_dense_feature(
            valid_level_set_grid_loc,
            np.ones(len(valid_level_set_grid_loc), dtype=np.float32),
            grid_dims,
            anchor_offset=anchor_offset,
            load_all_atoms=load_all_atoms,
        )  # this mask actually is the level set value count at each grid location due to the sparce representation
        level_set = self.create_dense_feature(
            valid_level_set_grid_loc,
            feat_dict["level_set_value"][level_set_grid_loc_mask],
            grid_dims,
            anchor_offset=anchor_offset,
            load_all_atoms=load_all_atoms,
        )

        # for multiple atoms at the same grid location, we normalize the level set values. For charges, we keep them #
        # because it can be a good indicator for multiple potential values.
        level_set = torch.where(
            level_set_mask > 0,
            level_set / level_set_mask,
            0.0,
        )

        # to accurately calculate the potential value, we need to get the original atom charge and
        # potential values in the patch because some atoms may fall at the same grid location
        # let's say two atoms with c1 and c2 charges fall at the same grid location, and the potential
        # values are p1 and p2, respectively. The potential value at this grid location should be
        # (c1 * p1 + c2 * p2) / (c1 + c2). Carefully calculate the potential value in this way.
        atom_charge_potential_product_sum = self.create_dense_feature(
            valid_atom_grid_loc,
            feat_dict["atom_charge"][atom_grid_loc_mask]
            * feat_dict["atom_potential"][atom_grid_loc_mask],
            grid_dims,
            anchor_offset=anchor_offset,
            load_all_atoms=load_all_atoms,
        )

        # further mask the potential values and atom mask that are not in the patch but in the context
        # since they are used for inference only and should not be used for training
        in_context_mask = torch.ones_like(atom_mask, dtype=torch.bool)
        in_context_mask[
            patch_offset_in_context[0] : patch_offset_in_context[0] + self.patch_size,
            patch_offset_in_context[1] : patch_offset_in_context[1] + self.patch_size,
            patch_offset_in_context[2] : patch_offset_in_context[2] + self.patch_size,
        ] = False
        atom_mask = torch.where(in_context_mask, 0.0, atom_mask)
        atom_potential = torch.where(
            atom_mask > 0,
            atom_charge_potential_product_sum / atom_charge,
            0.0,
        )

        return (
            grid_space,
            grid_dims,
            atom_mask,
            atom_charge,
            atom_type,
            atom_potential,
            level_set,
        )

    def process_one_patch(self, patch_path):
        with gzip.open(patch_path, "rb") as f:
            feat_dict = pickle.load(f)
        ionic_conc = extract_ionic_conc(feat_dict, patch_path)

        # prepare voxel patch
        (
            grid_space,
            grid_dims,
            atom_mask,
            atom_charge,
            atom_type,
            atom_potential,
            level_set,
        ) = self.prepare_voxel_patch(feat_dict, patch_path)

        # we don't need to create new anchors due to the `prepare_voxel_grids` function
        # already does the cropping if needed
        anchor_D, anchor_H, anchor_W = 0, 0, 0

        pad_level_set = pad_3d_tensor(
            level_set, anchor_D, anchor_H, anchor_W, self.context_patch_size
        )
        pad_atom_charge = pad_3d_tensor(
            atom_charge, anchor_D, anchor_H, anchor_W, self.context_patch_size
        )
        pad_atom_type = pad_3d_tensor(
            atom_type, anchor_D, anchor_H, anchor_W, self.context_patch_size
        )
        pad_atom_mask = pad_3d_tensor(
            atom_mask, anchor_D, anchor_H, anchor_W, self.context_patch_size
        )
        pad_potential = pad_3d_tensor(
            atom_potential, anchor_D, anchor_H, anchor_W, self.context_patch_size
        )

        # prepare tensors and transform them from DHWC to CDHW
        dim_inds = (3, 0, 1, 2)
        level_set_tensor = pad_level_set.unsqueeze(-1).permute(*dim_inds)
        atom_charge_tensor = pad_atom_charge.unsqueeze(-1).permute(*dim_inds)
        atom_type_tensor = pad_atom_type.unsqueeze(-1).permute(*dim_inds)
        atom_mask_tensor = pad_atom_mask.unsqueeze(-1).permute(*dim_inds)
        potential_tensor = pad_potential.unsqueeze(-1).permute(*dim_inds)
        grid_info = np.array([grid_space, ionic_conc], dtype=np.float32)
        return (
            level_set_tensor,
            atom_charge_tensor,
            atom_type_tensor,
            atom_mask_tensor,
            potential_tensor,
            grid_info,
        )


class VoxelImage3dEnergySparseDatasetForFullyConverage(VoxelImage3dEnergySparseDataset):
    """This dataset should be used with `batch size = 1` for evaluation, which generates patches covering all atoms."""

    def get_enumerating_anchor_points(
        self,
        D,
        H,
        W,
        atom_mask: torch.Tensor = None,
        atom_indices: np.ndarray = None,
    ):
        anchor_Ds = []
        anchor_Hs = []
        anchor_Ws = []

        # enumerating all possible crops starting from the middle
        middle_anchor_D, middle_anchor_H, middle_anchor_W = (
            self.get_middle_anchor_in_grid(D, H, W)
        )
        coverage_atom_num = 0

        # enumerate all possible crops starting from the middle anchor, however, this cannot coverage all atoms
        # for i in np.concatenate(
        #     (
        #         np.arange(middle_anchor_D, -1, -self.patch_size),
        #         np.arange(middle_anchor_D + self.patch_size, D, self.patch_size),
        #     )
        # ):
        #     for j in np.concatenate(
        #         (
        #             np.arange(middle_anchor_H, -1, -self.patch_size),
        #             np.arange(middle_anchor_H + self.patch_size, H, self.patch_size),
        #         )
        #     ):
        #         for k in np.concatenate(
        #             (
        #                 np.arange(middle_anchor_W, -1, -self.patch_size),
        #                 np.arange(
        #                     middle_anchor_W + self.patch_size, W, self.patch_size
        #                 ),
        #             )
        #         ):
        #             crop_atom_num = self.get_atom_num_in_patch(
        #                 atom_mask,
        #                 atom_indices,
        #                 i,
        #                 j,
        #                 k,
        #             )

        #             # only consider crops that cover atoms
        #             if crop_atom_num > 0:
        #                 anchor_Ds.append(i)
        #                 anchor_Hs.append(j)
        #                 anchor_Ws.append(k)
        #                 coverage_atom_num += crop_atom_num

        # enumerate all possible crops
        for i in range(0, D, self.patch_size):
            for j in range(0, H, self.patch_size):
                for k in range(0, W, self.patch_size):
                    crop_atom_num = self.get_atom_num_in_patch(
                        atom_mask,
                        atom_indices,
                        i,
                        j,
                        k,
                    )

                    # only consider crops that cover atoms
                    if crop_atom_num > 0:
                        anchor_Ds.append(i)
                        anchor_Hs.append(j)
                        anchor_Ws.append(k)
                        coverage_atom_num += crop_atom_num
        return anchor_Ds, anchor_Hs, anchor_Ws, coverage_atom_num

    def get_fully_convered_anchor_points(
        self,
        D,
        H,
        W,
        atom_mask: torch.Tensor = None,
        atom_indices: np.ndarray = None,
    ):
        # sample anchor point
        total_atom_num = (
            atom_mask.sum().item() if atom_mask is not None else len(atom_indices)
        )

        # try to focus on most atoms
        middle_anchor_D, middle_anchor_H, middle_anchor_W = (
            self.get_middle_anchor_in_grid(D, H, W)
        )
        crop_atom_num = self.get_atom_num_in_patch(
            atom_mask,
            atom_indices,
            middle_anchor_D,
            middle_anchor_H,
            middle_anchor_W,
        )
        if crop_atom_num == total_atom_num:
            # patch size can hold all atoms
            anchors = ([middle_anchor_D], [middle_anchor_H], [middle_anchor_W])
        else:
            *anchors, crop_atom_num = self.get_enumerating_anchor_points(
                D, H, W, atom_mask, atom_indices
            )
        return anchors

    def process_one_patch(self, patch_path):
        with gzip.open(patch_path, "rb") as f:
            feat_dict = pickle.load(f)
        ionic_conc = extract_ionic_conc(feat_dict, patch_path)

        # get fully converged anchor points
        _, grid_dims, _, atom_grid_loc, atom_grid_loc_mask, *_ = (
            self.prepare_voxel_grid(feat_dict, patch_path)
        )
        anchor_Ds, anchor_Hs, anchor_Ws = self.get_fully_convered_anchor_points(
            *grid_dims, atom_indices=atom_grid_loc[atom_grid_loc_mask]
        )

        level_set_tensors = []
        atom_charge_tensors = []
        atom_type_tensors = []
        atom_mask_tensors = []
        potential_tensors = []
        grid_spaces = []
        for anchor_D, anchor_H, anchor_W in zip(anchor_Ds, anchor_Hs, anchor_Ws):
            (
                grid_space,
                grid_dims,
                atom_mask,
                atom_charge,
                atom_type,
                atom_potential,
                level_set,
            ) = self.prepare_voxel_patch(
                feat_dict,
                patch_path,
                given_anchor_offset=(anchor_D, anchor_H, anchor_W),
            )

            # we don't need to create new anchors due to the `prepare_voxel_grids` function already does the cropping
            # by setting the anchor offset to `(anchor_D, anchor_H, anchor_W)`
            anchor_D, anchor_H, anchor_W = 0, 0, 0
            pad_level_set = pad_3d_tensor(
                level_set, anchor_D, anchor_H, anchor_W, self.context_patch_size
            )
            pad_atom_charge = pad_3d_tensor(
                atom_charge, anchor_D, anchor_H, anchor_W, self.context_patch_size
            )
            pad_atom_type = pad_3d_tensor(
                atom_type, anchor_D, anchor_H, anchor_W, self.context_patch_size
            )
            pad_atom_mask = pad_3d_tensor(
                atom_mask, anchor_D, anchor_H, anchor_W, self.context_patch_size
            )
            pad_potential = pad_3d_tensor(
                atom_potential, anchor_D, anchor_H, anchor_W, self.context_patch_size
            )

            # prepare tensors and transform them from DHWC to CDHW
            dim_inds = (3, 0, 1, 2)
            level_set_tensor = pad_level_set.unsqueeze(-1).permute(*dim_inds)
            atom_charge_tensor = pad_atom_charge.unsqueeze(-1).permute(*dim_inds)
            atom_type_tensor = pad_atom_type.unsqueeze(-1).permute(*dim_inds)
            atom_mask_tensor = pad_atom_mask.unsqueeze(-1).permute(*dim_inds)
            potential_tensor = pad_potential.unsqueeze(-1).permute(*dim_inds)

            level_set_tensors.append(level_set_tensor)
            atom_charge_tensors.append(atom_charge_tensor)
            atom_type_tensors.append(atom_type_tensor)
            atom_mask_tensors.append(atom_mask_tensor)
            potential_tensors.append(potential_tensor)
            grid_spaces.append([grid_space, ionic_conc])

        outputs = (
            torch.stack(level_set_tensors),
            torch.stack(atom_charge_tensors),
            torch.stack(atom_type_tensors),
            torch.stack(atom_mask_tensors),
            torch.stack(potential_tensors),
            torch.tensor(grid_spaces, dtype=torch.float32),
        )
        return outputs


class VoxelImage3dEnergySparseAtomicDataset(VoxelImage3dEnergySparseDataset):
    """This dataset produce atoms in an array format for evaluation, which generates patches covering all atoms."""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # atomic dataset overide the patch size semantic. if patch size = -1, it means to load all atoms
        # this is achieved by setting the patch size to the maximum value of grid dimensions
        self._patch_size = self.patch_size

    def prepare_atom_neighbor_features(
        self,
        atom_xyz: np.ndarray,
    ):
        atom_idx_i = None
        atom_idx_j = None
        # get the k-neighbor level-set values for each atoms if needed
        if self.extra_config.process_atom_neighbor_pairs:
            """ this is using the KDTree to get the nearest neighbors """
            kdtree = KDTree(atom_xyz)
            if (
                self.extra_config.max_num_neighbors is not None
                and atom_xyz.shape[0]
                > self.extra_config.natom_threshold_for_max_num_neighbors
            ):
                _, atom_indices = kdtree.query(
                    atom_xyz,
                    k=self.extra_config.max_num_neighbors + 1,
                    p=2,
                    distance_upper_bound=self.extra_config.neighbor_list_cutoff,
                )  # there is a point itself, so we add max_num_neighbors + 1

                atom_idx_i = np.repeat(
                    np.arange(len(atom_xyz)), [len(indices) for indices in atom_indices]
                )
                atom_idx_j = np.concatenate([indices for indices in atom_indices])

                # drop self inerataion
                mask = atom_idx_i != atom_idx_j
                atom_idx_i = atom_idx_i[mask]
                atom_idx_j = atom_idx_j[mask]
            else:
                atom_pairs = kdtree.query_pairs(
                    r=self.extra_config.neighbor_list_cutoff,
                    p=2.0,
                    output_type="ndarray",
                )
                atom_idx_i = atom_pairs[:, 0]
                atom_idx_j = atom_pairs[:, 1]

                # the obtained pairs are single direction, we need to make it bidirectional
                atom_idx_i = np.concatenate([atom_idx_i, atom_idx_j])
                atom_idx_j = np.concatenate([atom_idx_j, atom_idx_i])
                sort_by_i = np.argsort(atom_idx_i)
                atom_idx_i = atom_idx_i[sort_by_i]
                atom_idx_j = atom_idx_j[sort_by_i]

        atom_idx_i_tensor = (
            torch.from_numpy(
                atom_idx_i,
            ).to(dtype=torch.long)
            if atom_idx_i is not None
            else None
        )
        atom_idx_j_tensor = (
            torch.from_numpy(
                atom_idx_j,
            ).to(dtype=torch.long)
            if atom_idx_j is not None
            else None
        )
        return (
            atom_idx_i_tensor,
            atom_idx_j_tensor,
        )

    def prepare_level_set_neighbor_features(
        self,
        atom_xyz: np.ndarray,
        level_set_xyz: np.ndarray,
        level_set_values: np.ndarray,
    ):
        atom2level_set_i = None
        atom2level_set_j = None

        if self.extra_config.select_nearest_level_set:
            # get the k-neighbor level-set values for each atoms
            kdtree = KDTree(level_set_xyz)
            _, nearest_indices = kdtree.query(
                atom_xyz, k=self.extra_config.select_nearest_level_set_k, p=1
            )

            # atom without any nearest level set will be marked as n = len(level_set_xyz)
            valid_neighbors_mask = nearest_indices != len(level_set_xyz)
            nearest_neighbors_num = valid_neighbors_mask.sum(-1).astype(np.int64)
            atom2level_set_i = np.arange(len(atom_xyz), dtype=np.int64)
            atom2level_set_i = np.repeat(atom2level_set_i, nearest_neighbors_num)
            atom2level_set_j = nearest_indices.flatten()[valid_neighbors_mask.flatten()]
            level_set_xyz = level_set_xyz[atom2level_set_j]
            level_set_values = level_set_values[atom2level_set_j]

            # atom2level_set_j should start from 0
            atom2level_set_j = np.arange(len(atom2level_set_j), dtype=np.int64)

        atom2level_set_i_tensor = (
            torch.from_numpy(
                atom2level_set_i,
            ).to(dtype=torch.long)
            if atom2level_set_i is not None
            else None
        )
        atom2level_set_j_tensor = (
            torch.from_numpy(
                atom2level_set_j,
            ).to(dtype=torch.long)
            if atom2level_set_j is not None
            else None
        )
        level_set_xyz_tensor = torch.from_numpy(
            level_set_xyz,
        ).to(dtype=torch.float32)
        level_set_values_tensor = torch.from_numpy(
            level_set_values,
        ).to(dtype=torch.float32)
        return (
            atom2level_set_i_tensor,
            atom2level_set_j_tensor,
            level_set_xyz_tensor,
            level_set_values_tensor,
        )

    def prepare_boundary_atom_features(
        self,
        atom_xyz: np.ndarray,
        atom_charge: np.ndarray,
    ):
        atom2boundary_i = None
        atom2boundary_j = None
        boundary_xyz = None
        boundary_charge = None
        if self.extra_config.create_boundary_features:
            # Generate all displacement combinations for the given range [-max_distance, max_distance]
            grid_space = self.extra_config.boundary_unit
            boundary_cutoff_in_grids = int(
                self.extra_config.boundary_cutoff // grid_space
            )
            displacements = [
                (dx, dy, dz)
                for dx, dy, dz in product(
                    range(-boundary_cutoff_in_grids, boundary_cutoff_in_grids + 1),
                    repeat=3,
                )
                if not (dx == 0 and dy == 0 and dz == 0)  # Exclude the origin (0, 0, 0)
            ]

            # Convert displacements to a NumPy array
            displacements = np.array(displacements) * grid_space

            # Generate neighbors by adding each displacement to every atom position
            neighbors = atom_xyz[:, np.newaxis, :] + displacements[np.newaxis, :, :]
            boundary_xyz = neighbors.reshape(-1, 3)

            # query charge
            kdtree = KDTree(atom_xyz)
            atom_indices = kdtree.query_ball_point(
                boundary_xyz, r=self.extra_config.boundary_cutoff, p=2
            )

            # Compute boundary charges
            boundary_charge = np.zeros(len(boundary_xyz))
            boundary_connected_mask = np.zeros(len(boundary_xyz), dtype=np.bool_)
            # Loop through each boundary point to calculate the charge contribution from nearby atoms
            for i, indices in enumerate(atom_indices):
                if indices:  # Check if there are nearby atoms within the cutoff radius
                    # Calculate distances from the boundary point to each nearby atom
                    distances = np.linalg.norm(
                        atom_xyz[indices] - boundary_xyz[i : i + 1], axis=1
                    )

                    # Get the charges of the nearby atoms
                    charges = atom_charge[indices]

                    # Avoid division by zero if any distance is zero by adding a small epsilon
                    distances = np.maximum(distances, 1e-6)

                    # Calculate the weighted charge contribution (e.g., inverse distance weighting)
                    weighted_charges = charges / distances

                    # Sum the weighted charges for this boundary point
                    boundary_charge[i] = np.sum(weighted_charges)
                    boundary_connected_mask[i] = 1

            # shrink unconncected boundary
            boundary_xyz = boundary_xyz[boundary_connected_mask]
            boundary_charge = boundary_charge[boundary_connected_mask]

            # create atom2boundary_i and atom2boundary_j
            atom2boundary_i = np.concatenate(
                [indices for indices in atom_indices[boundary_connected_mask]]
            )
            atom2boundary_j = np.repeat(
                np.arange(len(boundary_xyz)),
                [len(indices) for indices in atom_indices[boundary_connected_mask]],
            )

            sort_by_i = np.argsort(atom2boundary_i)
            atom2boundary_i = atom2boundary_i[sort_by_i]
            atom2boundary_j = atom2boundary_j[sort_by_i]

        atom2boundary_i_tensor = (
            torch.from_numpy(
                atom2boundary_i,
            ).to(dtype=torch.long)
            if atom2boundary_i is not None
            else None
        )
        atom2boundary_j_tensor = (
            torch.from_numpy(
                atom2boundary_j,
            ).to(dtype=torch.long)
            if atom2boundary_j is not None
            else None
        )
        boundary_xyz_tensor = (
            torch.from_numpy(
                boundary_xyz,
            ).to(dtype=torch.float32)
            if boundary_xyz is not None
            else None
        )
        boundary_charge_tensor = (
            torch.from_numpy(
                boundary_charge,
            ).to(dtype=torch.float32)
            if boundary_charge is not None
            else None
        )
        return (
            atom2boundary_i_tensor,
            atom2boundary_j_tensor,
            boundary_xyz_tensor,
            boundary_charge_tensor,
        )

    def create_atomic_feature(
        self,
        indices: np.ndarray,
        values: np.ndarray,
        size: tuple[int],
        map_dict: dict[str, int] = None,
        anchor_offset: tuple[int] = (0, 0, 0),
        load_all_atoms: bool = False,
        dtype=torch.float32,
    ):
        """load dense feature from a sparse representation

        :param indices: value indices [N, 3]
        :param values: values
        :param size: the original full size of the molecule grid from amber
        :param map_dict: whether to map values, defaults to None
        :param anchor_offset: whether load a local patch of molecule, defaults to (0, 0, 0)
        :return: the loaded patch of molecule grid, if anchor_offset is not (0, 0, 0), the size of the output will be different
        """
        assert len(indices) == len(values), "Indices and values should have same length"
        assert indices.shape[1] == len(size), "Indices should have same length as size"

        indices, values, size = self.get_indice_value_pairs_in_context_patch(
            indices,
            values,
            size,
            anchor_offset,
            load_all_atoms,
        )

        # quick mapping from dict: https://stackoverflow.com/questions/16992713/translate-every-element-in-numpy-array-according-to-key
        if map_dict is not None:
            u, inv = np.unique(values, return_inverse=True)
            values = np.array([map_dict.get(x, map_dict["UNK"]) for x in u])[
                inv
            ].reshape(values.shape)

        return torch.from_numpy(values).to(dtype)

    def prepare_voxel_patch(
        self,
        feat_dict: Dict[str, Any],
        patch_path: str,
        given_anchor_offset: Union[tuple[int], None] = None,
        load_all_atoms: bool = False,
    ):
        """create patch features from the given feat_dict

        feature dictionary must contain atom_charge, atom_type, level_set_value, and
            atom_xyz, atom_rad, level_set_xyz for pdb type datasets
            grid_dims, grid_origin, grid_space, atom_grid_loc, level_set_grid_loc for amber type datasets
        """
        (
            grid_origin,
            grid_dims,
            grid_space,
            atom_grid_loc,
            atom_grid_loc_mask,
            level_set_grid_loc,
            level_set_grid_loc_mask,
        ) = self.prepare_voxel_grid(
            feat_dict,
            patch_path,
            from_pdb=self.extra_config.from_pdb,
        )

        # if patch size is -1, load all atoms
        if self._patch_size == -1:
            self.patch_size = max(grid_dims)
            given_anchor_offset = (0, 0, 0)
            load_all_atoms = True
            assert self.context_patch_size == self.patch_size, (
                "Context size should be 0"
            )

        if given_anchor_offset is not None:
            anchor_offset, patch_offset_in_context = (
                self.get_context_based_anchor_point(given_anchor_offset)
            )
        else:
            # randomly crop if necessary
            anchor_D, anchor_H, anchor_W = self.get_anchor_point(
                *grid_dims, atom_indices=atom_grid_loc
            )
            random_crop_anchor_offset = (anchor_D, anchor_H, anchor_W)

            # load required voxel grid patch origined at anchor offset
            anchor_offset, patch_offset_in_context = (
                self.get_context_based_anchor_point(random_crop_anchor_offset)
            )

        # get required dense tensors
        valid_atom_grid_loc = atom_grid_loc[atom_grid_loc_mask]
        valid_level_set_grid_loc = level_set_grid_loc[level_set_grid_loc_mask]
        atom_charge = self.create_atomic_feature(
            valid_atom_grid_loc,
            feat_dict["atom_charge"][atom_grid_loc_mask],
            grid_dims,
            anchor_offset=anchor_offset,
            load_all_atoms=load_all_atoms,
        )
        atom_type = self.create_atomic_feature(
            valid_atom_grid_loc,
            feat_dict["atom_type"][atom_grid_loc_mask],
            grid_dims,
            self.atom_type_map,
            anchor_offset=anchor_offset,
            load_all_atoms=load_all_atoms,
        )
        level_set = self.create_atomic_feature(
            valid_level_set_grid_loc,
            feat_dict["level_set_value"][level_set_grid_loc_mask],
            grid_dims,
            anchor_offset=anchor_offset,
            load_all_atoms=load_all_atoms,
        )
        atom_mask = self.create_atomic_feature(
            valid_atom_grid_loc,
            np.ones(len(valid_atom_grid_loc), dtype=np.float32),
            grid_dims,
            anchor_offset=anchor_offset,
            load_all_atoms=load_all_atoms,
        )
        atom_potential = self.create_atomic_feature(
            valid_atom_grid_loc,
            feat_dict["atom_potential"][atom_grid_loc_mask],
            grid_dims,
            anchor_offset=anchor_offset,
            load_all_atoms=load_all_atoms,
        )

        # further mask the potential values and atom mask that are not in the patch but in the context
        # since they are used for inference only and should not be used for training
        # only load a local patch of molecule
        *_, selected_mask = self.get_indice_value_pairs_in_context_patch(
            valid_atom_grid_loc,
            np.ones(len(valid_atom_grid_loc)),
            grid_dims,
            anchor_offset,
            load_all_atoms,
            return_selected_indices=True,
        )
        indices = (
            valid_atom_grid_loc[selected_mask]
            - np.array(anchor_offset)[None, :]
            - np.array(patch_offset_in_context)[None, :]
        )
        selected_indices = np.all(indices >= 0, axis=1) & np.all(
            indices < self.patch_size, axis=1
        )
        in_patch_mask = torch.from_numpy(selected_indices).to(torch.bool)

        # atom related features
        atom_xyz_np = self.indices2xyz(
            valid_atom_grid_loc[selected_mask], grid_origin, grid_space
        )
        atom_xyz = torch.from_numpy(
            atom_xyz_np,
        ).to(dtype=torch.float32)
        atom_mask = torch.where(in_patch_mask, atom_mask, 0.0)
        atom_potential = torch.where(
            in_patch_mask,
            atom_potential,
            0.0,
        )

        # process level set features
        level_set_xyz_np = self.indices2xyz(
            valid_level_set_grid_loc, grid_origin, grid_space
        )
        (
            atom2level_set_i,
            atom2level_set_j,
            level_set_xyz,
            level_set,
        ) = self.prepare_level_set_neighbor_features(
            atom_xyz_np,
            level_set_xyz_np,
            feat_dict["level_set_value"][level_set_grid_loc_mask],
        )

        # create neighbor list (it can be processed in the model or here)
        atom_idx_i, atom_idx_j = self.prepare_atom_neighbor_features(atom_xyz_np)

        patch = dict(
            grid_space=grid_space,
            atom_xyz=atom_xyz,
            atom_mask=atom_mask,
            atom_charge=atom_charge,
            atom_type=atom_type,
            atom_potential=atom_potential,
            atom_idx_i=atom_idx_i,
            atom_idx_j=atom_idx_j,
            atom2level_set_i=atom2level_set_i,
            atom2level_set_j=atom2level_set_j,
            level_set_xyz=level_set_xyz,
            level_set=level_set,
        )

        if self.extra_config.create_boundary_features:
            # create boundary atoms features
            atom2boundary_i, atom2boundary_j, boundary_xyz, boundary_charge = (
                self.prepare_boundary_atom_features(
                    atom_xyz_np, feat_dict["atom_charge"][atom_grid_loc_mask]
                )
            )

            (
                boundary2level_set_i,
                boundary2level_set_j,
                boundary_level_set_xyz,
                boundary_level_set,
            ) = self.prepare_level_set_neighbor_features(
                boundary_xyz.numpy(),
                level_set_xyz_np,
                feat_dict["level_set_value"][level_set_grid_loc_mask],
            )

            boundary_feat = dict(
                atom2boundary_i=atom2boundary_i,
                atom2boundary_j=atom2boundary_j,
                boundary_xyz=boundary_xyz,
                boundary_charge=boundary_charge,
                boundary2level_set_i=boundary2level_set_i,
                boundary2level_set_j=boundary2level_set_j,
                boundary_level_set_xyz=boundary_level_set_xyz,
                boundary_level_set=boundary_level_set,
            )
            patch.update(boundary_feat)

        return patch

    def process_one_patch(self, patch_path):
        with gzip.open(patch_path, "rb") as f:
            feat_dict = pickle.load(f)

        # prepare voxel patch
        patch = self.prepare_voxel_patch(feat_dict, patch_path)
        return patch


class VoxelImage3dEnergySparseAtomicDatasetForFullyConverage(
    VoxelImage3dEnergySparseAtomicDataset
):
    def get_enumerating_anchor_points(
        self,
        D,
        H,
        W,
        atom_mask: torch.Tensor = None,
        atom_indices: np.ndarray = None,
    ):
        anchor_Ds = []
        anchor_Hs = []
        anchor_Ws = []

        # enumerating all possible crops starting from the middle
        middle_anchor_D, middle_anchor_H, middle_anchor_W = (
            self.get_middle_anchor_in_grid(D, H, W)
        )
        coverage_atom_num = 0

        # enumerate all possible crops
        for i in range(0, D, self.patch_size):
            for j in range(0, H, self.patch_size):
                for k in range(0, W, self.patch_size):
                    crop_atom_num = self.get_atom_num_in_patch(
                        atom_mask,
                        atom_indices,
                        i,
                        j,
                        k,
                    )

                    # only consider crops that cover atoms
                    if crop_atom_num > 0:
                        anchor_Ds.append(i)
                        anchor_Hs.append(j)
                        anchor_Ws.append(k)
                        coverage_atom_num += crop_atom_num
        return anchor_Ds, anchor_Hs, anchor_Ws, coverage_atom_num

    def get_fully_convered_anchor_points(
        self,
        D,
        H,
        W,
        atom_mask: torch.Tensor = None,
        atom_indices: np.ndarray = None,
    ):
        # sample anchor point
        total_atom_num = (
            atom_mask.sum().item() if atom_mask is not None else len(atom_indices)
        )

        # try to focus on most atoms
        middle_anchor_D, middle_anchor_H, middle_anchor_W = (
            self.get_middle_anchor_in_grid(D, H, W)
        )
        crop_atom_num = self.get_atom_num_in_patch(
            atom_mask,
            atom_indices,
            middle_anchor_D,
            middle_anchor_H,
            middle_anchor_W,
        )
        if crop_atom_num == total_atom_num:
            # patch size can hold all atoms
            anchors = ([middle_anchor_D], [middle_anchor_H], [middle_anchor_W])
        else:
            *anchors, crop_atom_num = self.get_enumerating_anchor_points(
                D, H, W, atom_mask, atom_indices
            )
        return anchors

    def process_one_patch(self, patch_path):
        with gzip.open(patch_path, "rb") as f:
            feat_dict = pickle.load(f)

        ground_true_epb = (
            np.sum(feat_dict["atom_potential"] * feat_dict["atom_charge"]) * 0.5
        ).item()

        # if patch size is -1, load all atoms automatically
        if self._patch_size == -1:
            patch = self.prepare_voxel_patch(feat_dict, patch_path)

            patch.update(
                dict(
                    ground_true_epb=ground_true_epb,
                )
            )
            return patch
        else:
            # get fully converged anchor points
            _, grid_dims, _, atom_grid_loc, atom_grid_loc_mask, *_ = (
                self.prepare_voxel_grid(feat_dict, patch_path)
            )
            anchor_Ds, anchor_Hs, anchor_Ws = self.get_fully_convered_anchor_points(
                *grid_dims, atom_indices=atom_grid_loc[atom_grid_loc_mask]
            )

            patches = []
            for anchor_D, anchor_H, anchor_W in zip(anchor_Ds, anchor_Hs, anchor_Ws):
                patch = self.prepare_voxel_patch(
                    feat_dict,
                    patch_path,
                    given_anchor_offset=(anchor_D, anchor_H, anchor_W),
                )

                patch.update(
                    dict(
                        ground_true_epb=ground_true_epb,
                    )
                )
                patches.append(patch)
            return patches


if __name__ == "__main__":
    # dataset = RefinedMlsesDataset("dataset/benchmark_sample")
    # print(len(dataset))
    # print(dataset[200])
    # print(dataset[200][0].shape)

    # dataset = Image3dEnergyDataset(
    #     "/home/junhal11/refined-mlses-mirror/datasets/benchmark_3d_energy/moles_index_sz64.json",
    #     patch_size=64,
    # )
    # from torch.utils.data import DataLoader

    # dl = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)
    # for i in dl:
    #     breakpoint()

    import glob
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from .utils import seed_everything

    seed_everything(2023)
    # dataset = VoxelImage3dEnergySparseDataset(
    #     glob.glob(
    #         "datasets/processed/new_full_3d_energy_sparse_v2/*/protein_protein_test/*/*.pkl.gz"
    #     ),
    #     patch_size=128,
    #     epb_mean=0.0,
    #     epb_std=1.0,
    #     do_random_crop=True,
    #     random_crop_atom_num=10,
    #     do_random_rotate=True,
    #     do_voxel_grids_shrinking=True,
    #     do_random_grid_scaling=True,
    # )
    # dl = DataLoader(dataset, batch_size=8, num_workers=8)
    # for i in tqdm(dl):
    #     print(i[0].shape)

    # dataset = VoxelImage3dEnergySparseDatasetForFullyConverage(
    #     glob.glob(
    #         "datasets/processed/new_full_3d_energy_sparse_v2/bench_full_abs_cpu_075/protein_protein_test/*/*.pkl.gz"
    #     ),
    #     patch_size=32,
    #     epb_mean=0.0,
    #     epb_std=1.0,
    #     do_random_crop=True,
    #     random_crop_atom_num=10,
    #     do_random_rotate=True,
    #     given_rotate_angle=[30],
    #     given_rotate_axis=["z"],
    #     do_voxel_grids_shrinking=True,
    #     do_random_grid_scaling=False,
    #     given_grid_scaling_size=0.95,
    #     extra_config=VoxelImage3dEnergyDatasetExtraConfig(context_size=48),
    # )
    # dl = DataLoader(dataset, batch_size=1, num_workers=0)
    # for i in tqdm(dl):
    #     print(i[0].shape)

    dataset = VoxelImage3dEnergySparseAtomicDataset(
        glob.glob(
            "datasets/processed/new_full_3d_energy_sparse_v2/bench_full_abs_cpu_075/protein_protein_test/*/*.pkl.gz"
        ),
        patch_size=-1,
        epb_mean=0.0,
        epb_std=1.0,
        do_random_crop=True,
        random_crop_atom_num=10,
        do_random_rotate=True,
        given_rotate_angle=[30],
        given_rotate_axis=["z"],
        do_voxel_grids_shrinking=True,
        do_random_grid_scaling=False,
        given_grid_scaling_size=0.95,
        extra_config=VoxelImage3dEnergyDatasetExtraConfig(
            context_size=0,
            process_atom_neighbor_pairs=True,
            neighbor_list_cutoff=30,
            max_num_neighbors=16,
            natom_threshold_for_max_num_neighbors=10,
            select_nearest_level_set=True,
            select_nearest_level_set_k=6,
            create_boundary_features=True,
        ),
    )
    dl = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        collate_fn=AtomicDataCollator(contain_boundary=True),
    )
    for i in tqdm(dl):
        breakpoint()
        print(i)

    # dataset = VoxelImage3dEnergySparseAtomicDatasetForFullyConverage(
    #     glob.glob(
    #         "datasets/processed/new_full_3d_energy_sparse_v2/bench_full_abs_cpu_075/protein_protein_test/*/*.pkl.gz"
    #     ),
    #     patch_size=-1,
    #     epb_mean=0.0,
    #     epb_std=1.0,
    #     do_random_crop=True,
    #     random_crop_atom_num=10,
    #     do_random_rotate=True,
    #     given_rotate_angle=[30],
    #     given_rotate_axis=["z"],
    #     do_voxel_grids_shrinking=True,
    #     do_random_grid_scaling=False,
    #     given_grid_scaling_size=0.95,
    #     extra_config=VoxelImage3dEnergyDatasetExtraConfig(context_size=0, process_atom_neighbor_pairs=30, select_nearest_level_set=True, select_nearest_level_set_k=6),
    # )
    # dl = DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=AtomicDataCollator())
    # for i in tqdm(dl):
    #     breakpoint()
    #     print(i)
