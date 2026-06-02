import tyro
import json
import os
import glob
import gzip
import pickle
import numpy as np

from typing import List, Tuple
from tqdm import tqdm
from filelock import SoftFileLock
from ion_pot.preprocess.file_helper import (
    read_dat,
    read_bounding_box_info,
    parse_dat_path,
    read_pdb_file,
    small_mol_read_pdb_file,
    read_potential_file,
    get_grid_locs,
)
from ion_pot.configs.preprocess.epb_dataset import EPBDatasetConfig, ConfiguredEPBDatasetConfig
from pathlib import Path


def make_3d_image(
    abs_grid_feats: np.ndarray,
    grid_labels: np.ndarray,
    abs_grid_dims: np.ndarray,
    is_converted_locs: bool = True,
    abs_grid_origin=None,
    h_space=None,
    precision=None,
):
    """Generate input image, level-set image, and a mask based on the absolute location of grid point and relative grid features

    x -> d
    y -> h
    z -> w

    :param abs_grid_feats: absolute location (x, y, z) of grid point [N, 3]
    :param rel_grid_feats: accoding relative grid features of grid point [N, feat_dim]
    :param grid_labels: level-set value at each grid point [N,]
    :return: a tuple of input image, level-set image, and a mask
    """
    if is_converted_locs:
        grid_locs = abs_grid_feats.astype(int)
    else:
        grid_locs = get_grid_locs(
            abs_grid_feats, abs_grid_origin, h_space, precision=precision
        )
    # the grid point feature should be the same
    unique_grid_locs, unique_grid_locs_counts = np.unique(
        grid_locs, axis=0, return_counts=True
    )
    if len(unique_grid_locs) != len(grid_locs):
        raise Exception("Duplicate grid points")

    return {
        "level_set_grid_loc": grid_locs,
        "level_set_value": grid_labels,
        "level_set_mask": np.ones_like(grid_labels),
    }


def convert_dat_to_3d_image(
    abs_dat_feats: np.ndarray,
    abs_dat_labels: np.ndarray,
    abs_grid_dims: np.ndarray,
    is_converted_locs: bool = True,
    abs_grid_origin=None,
    h_space=None,
    precision=None,
):
    return make_3d_image(
        abs_dat_feats[:, :3],
        abs_dat_labels,
        abs_grid_dims,
        is_converted_locs=is_converted_locs,
        abs_grid_origin=abs_grid_origin,
        h_space=h_space,
        precision=precision,
    )


def convert_pdb_to_3d_image(
    abs_grid_feats: np.ndarray,
    abs_atom_types: np.ndarray,
    abs_grid_charges: np.ndarray,
    abs_atom_structs: np.ndarray,
    abs_grid_potential: np.ndarray,
    abs_grid_dims: np.ndarray,
    abs_grid_origin: np.ndarray,
    h_space: float,
    precision: int = 2,
) -> dict[str, np.ndarray]:
    """Generate input image, level-set image, and a mask based on the absolute location of grid point and relative grid features

    x -> d
    y -> h
    z -> w

    :param abs_grid_feats: absolute location (x, y, z) of grid point [N, 3]
    :param rel_grid_feats: accoding relative grid features of grid point [N, feat_dim]
    :param grid_labels: level-set value at each grid point [N,]
    :return: a tuple of input image, level-set image, and a mask
    """
    grid_locs = get_grid_locs(
        abs_grid_feats, abs_grid_origin, h_space, precision=precision
    )

    # the grid point feature should be the same
    unique_grid_locs, unique_grid_locs_counts = np.unique(
        grid_locs, axis=0, return_counts=True
    )

    if len(unique_grid_locs) != len(grid_locs):
        raise Exception("Duplicate grid points")

    return {
        "atom_grid_loc": grid_locs,
        "atom_charge": abs_grid_charges,
        "atom_type": abs_atom_types,
        "atom_struct": abs_atom_structs,
        "atom_potential": abs_grid_potential,
        "atom_mask": np.ones_like(abs_grid_charges),
    }


def old_per_sample_read_energy_dat_then_process(
    abs_dat_path,
    surface_charge_dat_folder,
    atom_pdb_folder,
    atom_potential_dat_folder,
    ATOM_TYPE_STATS: dict[str, int],
    STRC_TYPE_STATS: dict[str, int],
    processing_log_path: str,
    patch_size: int = 64,
    file_lock: bool = True,
    out_file_suffix: str = ".ipb2.out.bench",
    sliding_window_zoom_patch_size=32,
    sliding_window_stride=32,
    energy_image_dataset_path="./",
):
    try:
        category, case, dat_name = parse_dat_path(abs_dat_path)
        mole_name = dat_name[:6]
        abs_dat_folder = os.path.dirname(abs_dat_path)
        abs_out_path = glob.glob(
            os.path.join(abs_dat_folder, f"{mole_name}{out_file_suffix}*")
        )[0]

        try:
            atom_pdb_path = glob.glob(f"{atom_pdb_folder}/{case}/{mole_name}*.pqr")[0]
        except IndexError:
            raise Exception(f"Missing atom pdb file for {mole_name}")

        try:
            atom_potential_path = glob.glob(
                f"{atom_potential_dat_folder}/{mole_name}*.dat"
            )[0]
        except IndexError:
            raise Exception(f"Missing atom potential file for {mole_name}")

        # read grid origin in absolute coordinate
        grid_dims, grid_origin, grid_space = read_bounding_box_info(abs_out_path)
        assert grid_dims is not None and grid_origin is not None

        # read atom positions
        abs_atom_xyz, abs_atom_types, abs_atom_charges, abs_atom_structures = (
            read_pdb_file(atom_pdb_path, ATOM_TYPE_STATS, STRC_TYPE_STATS)
        )
        abs_atom_potential = read_potential_file(atom_potential_path)
        atom_3d_image = convert_pdb_to_3d_image(
            abs_atom_xyz,
            abs_atom_types,
            abs_atom_charges,
            abs_atom_structures,
            abs_atom_potential,
            grid_dims,
            grid_origin,
            h_space=grid_space,
        )

        # process level-set in this mole
        abs_dat_labels, abs_dat_feats = read_dat(abs_dat_path, max_length=3)
        level_set_3d_image = convert_dat_to_3d_image(
            abs_dat_feats,
            abs_dat_labels,
            grid_dims,
            is_converted_locs=False,
            abs_grid_origin=grid_origin,
            h_space=grid_space,
            precision=2,
        )

        # prepare sparse 3d information
        sparse_3d_info = {
            "grid_dims": grid_dims,
            "grid_origin": grid_origin,
            "grid_space": grid_space,
        }
        sparse_3d_info.update(atom_3d_image)
        sparse_3d_info.update(level_set_3d_image)

        # save the raw data
        saved_dir = os.path.join(energy_image_dataset_path, category, case, mole_name)
        os.makedirs(saved_dir, exist_ok=True)
        with gzip.GzipFile(
            os.path.join(
                saved_dir,
                f"{mole_name}_sparse.pkl.gz",
            ),
            "wb",
        ) as f:
            pickle.dump(sparse_3d_info, f)
    except BaseException as err:
        # stop the process if it is a keyboard interrupt
        if isinstance(err, KeyboardInterrupt):
            raise err

        # otherwise, log the error
        if not file_lock:
            with open(processing_log_path, "a") as processing_log:
                print(f"\nError: {err}", file=processing_log)
                print(f"abs: {abs_dat_path}", file=processing_log)
        else:
            file_lock = SoftFileLock(processing_log_path + ".lock")
            with file_lock:
                with open(processing_log_path, "a") as processing_log:
                    print(f"\nError: {err}", file=processing_log)
                    print(f"abs: {abs_dat_path}", file=processing_log)
        return


def read_energy_dat_then_process(
    abs_dat_paths: List[str],
    surface_charge_dat_folders: List[str],
    atom_pdb_folders: List[str],
    atom_potential_dat_folders: List[str],
    patch_size: int = 64,
    sliding_window_zoom_patch_size=32,
    sliding_window_stride=32,
    energy_image_dataset_path: str = "./",
    parsing_method="old",
):
    if os.path.exists(energy_image_dataset_path):
        decision = input(
            f"Path {energy_image_dataset_path} already exists. You are overwriting it. (y/n)?"
        )
        if decision.lower() != "y":
            return
    os.makedirs(energy_image_dataset_path, exist_ok=True)

    processing_log_path = f"{energy_image_dataset_path}/log.txt"

    ATOM_TYPE_STATS = {}
    STRC_TYPE_STATS = {}

    for (
        abs_dat_path,
        # surface_charge_dat_folder,
        atom_pdb_folder,
        atom_potential_dat_folder,
    ) in zip(
        abs_dat_paths,
        # surface_charge_dat_folders,
        atom_pdb_folders,
        atom_potential_dat_folders,
    ):
        file_paths: list[str] = glob.glob(abs_dat_path)
        for file_path in tqdm(file_paths):
            per_sample_read_energy_dat_then_process(
                file_path,
                "",
                atom_pdb_folder,
                atom_potential_dat_folder,
                ATOM_TYPE_STATS,
                STRC_TYPE_STATS,
                processing_log_path=processing_log_path,
                file_lock=False,
                patch_size=patch_size,
                sliding_window_zoom_patch_size=sliding_window_zoom_patch_size,
                sliding_window_stride=sliding_window_stride,
                energy_image_dataset_path=energy_image_dataset_path,
                parsing_method=parsing_method,
            )

    with open(f"{energy_image_dataset_path}/meta_info.json", "w") as f:
        meta_info = {
            "atom_stats": ATOM_TYPE_STATS,
            "strc_stats": STRC_TYPE_STATS,
            # "patch_info": {
            #     "patch_size": patch_size,
            #     "sliding_window_zoom_patch_size": sliding_window_zoom_patch_size,
            #     "sliding_window_stride": sliding_window_stride,
            # },
        }
        json.dump(
            meta_info,
            f,
            indent=4,
        )


def small_mol_parse_data_path(
    abs_dat_paths: str, atom_potential_dat_folders: str, atom_pdb_folders: str
) -> Tuple[str, str, str, str, str]:
    data_path = Path(abs_dat_paths)
    _, data_folder, data_group, mol_name, _ = data_path.parts

    # Outfile pattern
    outfile_pattern = data_path.parent / f"{mol_name}*.out.*"
    outfile_paths = glob.glob(str(outfile_pattern))
    if not outfile_paths:
        raise FileNotFoundError(f"No outfile found for pattern: {outfile_pattern}")
    outfile_path = outfile_paths[0]

    # Datfile pattern
    datfile_pattern = data_path.parent / "*.dat"
    datfile_paths = glob.glob(str(datfile_pattern))
    if not datfile_paths:
        raise FileNotFoundError(f"No datfile found for pattern: {datfile_pattern}")
    datfile_path = datfile_paths[0]

    # Potential out pattern
    potential_out_pattern = f"{atom_potential_dat_folders}/{mol_name}/{mol_name}*.out.*"
    potential_out_paths = glob.glob(potential_out_pattern)
    if not potential_out_paths:
        raise FileNotFoundError(
            f"No potential out file found for pattern: {potential_out_pattern}"
        )
    potential_out_path = potential_out_paths[0]

    # Potential dat pattern
    potential_dat_pattern = f"{atom_potential_dat_folders}/{mol_name}/*.dat"
    potential_dat_paths = glob.glob(potential_dat_pattern)
    if not potential_dat_paths:
        raise FileNotFoundError(
            f"No potential dat file found for pattern: {potential_dat_pattern}"
        )
    potential_dat_path = potential_dat_paths[0]

    # PDB pattern
    if "protein-testCase" in atom_pdb_folders:
        pdb_pattern = f"{atom_pdb_folders}/*/{mol_name}*.pqr"
    else:
        pdb_pattern = f"{atom_pdb_folders}/{mol_name}*.pdb"

    pdb_paths = glob.glob(pdb_pattern)
    if not pdb_paths:
        raise FileNotFoundError(f"No PDB file found for pattern: {pdb_pattern}")
    pdb_path = pdb_paths[0]

    mol_type = "small_mol"
    return (
        outfile_path,
        datfile_path,
        potential_out_path,
        potential_dat_path,
        pdb_path,
        data_group,
        mol_type,
        mol_name,
    )


def new_parse_data_path(
    abs_dat_paths: str, atom_potential_dat_folders: str, atom_pdb_folders: str
) -> Tuple[str, str, str, str, str]:
    data_path = Path(abs_dat_paths)
    _, data_folder, data_group, mol_type, mol_name, _ = data_path.parts

    # Outfile pattern
    outfile_pattern = data_path.parent / f"{mol_name}*.out.*"
    outfile_paths = glob.glob(str(outfile_pattern))
    if not outfile_paths:
        raise FileNotFoundError(f"No outfile found for pattern: {outfile_pattern}")
    outfile_path = outfile_paths[0]

    # Datfile pattern
    datfile_pattern = data_path.parent / "*.dat"
    datfile_paths = glob.glob(str(datfile_pattern))
    if not datfile_paths:
        raise FileNotFoundError(f"No datfile found for pattern: {datfile_pattern}")
    datfile_path = datfile_paths[0]

    # Potential out pattern
    potential_out_pattern = f"{atom_potential_dat_folders}/{mol_name}/{mol_name}*.out.*"
    potential_out_paths = glob.glob(potential_out_pattern)
    if not potential_out_paths:
        raise FileNotFoundError(
            f"No potential out file found for pattern: {potential_out_pattern}"
        )
    potential_out_path = potential_out_paths[0]

    # Potential dat pattern
    potential_dat_pattern = f"{atom_potential_dat_folders}/{mol_name}/*.dat"
    potential_dat_paths = glob.glob(potential_dat_pattern)
    if not potential_dat_paths:
        raise FileNotFoundError(
            f"No potential dat file found for pattern: {potential_dat_pattern}"
        )
    potential_dat_path = potential_dat_paths[0]

    # PDB pattern
    if "protein-testCase" in atom_pdb_folders:
        pdb_pattern = f"{atom_pdb_folders}/*/{mol_name}*.pqr"
    else:
        pdb_pattern = f"{atom_pdb_folders}/{mol_name}*.pdb"

    pdb_paths = glob.glob(pdb_pattern)
    if not pdb_paths:
        raise FileNotFoundError(f"No PDB file found for pattern: {pdb_pattern}")
    pdb_path = pdb_paths[0]

    return (
        outfile_path,
        datfile_path,
        potential_out_path,
        potential_dat_path,
        pdb_path,
        data_group,
        mol_type,
        mol_name,
    )


def old_get_file_paths(
    abs_dat_path: str,
    atom_pdb_folder: str,
    atom_potential_dat_folder: str,
    out_file_suffix: str,
) -> Tuple[str, str, str]:
    # Parse the data path
    category, case, dat_name = parse_dat_path(abs_dat_path)
    mole_name = dat_name[:6]

    # Determine the folder containing the data file
    abs_dat_folder = os.path.dirname(abs_dat_path)

    # Find the corresponding output file
    abs_out_pattern = os.path.join(abs_dat_folder, f"{mole_name}{out_file_suffix}*")
    abs_out_paths = glob.glob(abs_out_pattern)
    if not abs_out_paths:
        raise FileNotFoundError(f"No output file found for pattern: {abs_out_pattern}")
    abs_out_path = abs_out_paths[0]

    # Find the corresponding PDB file
    atom_pdb_pattern = f"{atom_pdb_folder}/{case}/{mole_name}*.pqr"
    atom_pdb_paths = glob.glob(atom_pdb_pattern)
    if not atom_pdb_paths:
        raise FileNotFoundError(f"Missing atom pdb file for {mole_name}")
    atom_pdb_path = atom_pdb_paths[0]

    # Find the corresponding potential data file
    atom_potential_pattern = f"{atom_potential_dat_folder}/{mole_name}*.dat"
    atom_potential_paths = glob.glob(atom_potential_pattern)
    if not atom_potential_paths:
        raise FileNotFoundError(f"Missing atom potential file for {mole_name}")
    atom_potential_path = atom_potential_paths[0]

    return abs_out_path, atom_pdb_path, atom_potential_path, category, case, mole_name


def per_sample_read_energy_dat_then_process(
    abs_dat_path,
    surface_charge_dat_folder,
    atom_pdb_folder,
    atom_potential_dat_folder,
    ATOM_TYPE_STATS: dict[str, int],
    STRC_TYPE_STATS: dict[str, int],
    processing_log_path: str,
    patch_size: int = 64,
    file_lock: bool = True,
    out_file_suffix: str = ".ipb2.out.bench",
    sliding_window_zoom_patch_size=32,
    sliding_window_stride=32,
    energy_image_dataset_path="./",
    parsing_method="old",
):
    try:
        if parsing_method == "old":
            (
                abs_out_path,
                atom_pdb_path,
                atom_potential_path,
                category,
                case,
                mole_name,
            ) = old_get_file_paths(
                abs_dat_path,
                atom_pdb_folder,
                atom_potential_dat_folder,
                out_file_suffix,
            )
        elif parsing_method == "small_mol":
            (
                abs_out_path,
                abs_dat_path,
                potential_out_path,
                atom_potential_path,
                atom_pdb_path,
                data_group,
                mol_type,
                mol_name,
            ) = small_mol_parse_data_path(
                abs_dat_path, atom_potential_dat_folder, atom_pdb_folder
            )
            category = data_group
            case = mol_type
            mole_name = mol_name
        else:
            (
                abs_out_path,
                abs_dat_path,
                potential_out_path,
                atom_potential_path,
                atom_pdb_path,
                data_group,
                mol_type,
                mol_name,
            ) = new_parse_data_path(
                abs_dat_path, atom_potential_dat_folder, atom_pdb_folder
            )
            category = data_group
            case = mol_type
            mole_name = mol_name

        # read grid origin in absolute coordinate
        grid_dims, grid_origin, grid_space = read_bounding_box_info(abs_out_path)
        assert grid_dims is not None and grid_origin is not None

        # read atom positions
        if parsing_method == "small_mol":
            abs_atom_xyz, abs_atom_types, abs_atom_charges, abs_atom_structures = (
                small_mol_read_pdb_file(atom_pdb_path, ATOM_TYPE_STATS, STRC_TYPE_STATS)
            )
        else:
            abs_atom_xyz, abs_atom_types, abs_atom_charges, abs_atom_structures = (
                read_pdb_file(atom_pdb_path, ATOM_TYPE_STATS, STRC_TYPE_STATS)
            )
        abs_atom_potential = read_potential_file(atom_potential_path)
        atom_3d_image = convert_pdb_to_3d_image(
            abs_atom_xyz,
            abs_atom_types,
            abs_atom_charges,
            abs_atom_structures,
            abs_atom_potential,
            grid_dims,
            grid_origin,
            h_space=grid_space,
        )

        # process level-set in this mole
        abs_dat_labels, abs_dat_feats = read_dat(abs_dat_path, max_length=3)
        level_set_3d_image = convert_dat_to_3d_image(
            abs_dat_feats,
            abs_dat_labels,
            grid_dims,
            is_converted_locs=False,
            abs_grid_origin=grid_origin,
            h_space=grid_space,
            precision=2,
        )

        # prepare sparse 3d information
        sparse_3d_info = {
            "grid_dims": grid_dims,
            "grid_origin": grid_origin,
            "grid_space": grid_space,
        }
        sparse_3d_info.update(atom_3d_image)
        sparse_3d_info.update(level_set_3d_image)

        # save the raw data
        saved_dir = os.path.join(energy_image_dataset_path, category, case, mole_name)
        os.makedirs(saved_dir, exist_ok=True)
        with gzip.GzipFile(
            os.path.join(
                saved_dir,
                f"{mole_name}_sparse.pkl.gz",
            ),
            "wb",
        ) as f:
            pickle.dump(sparse_3d_info, f)
    except BaseException as err:
        # stop the process if it is a keyboard interrupt
        if isinstance(err, KeyboardInterrupt):
            raise err

        # otherwise, log the error
        if not file_lock:
            with open(processing_log_path, "a") as processing_log:
                print(f"\nError: {err}", file=processing_log)
                print(f"abs: {abs_dat_path}", file=processing_log)
        else:
            file_lock = SoftFileLock(processing_log_path + ".lock")
            with file_lock:
                with open(processing_log_path, "a") as processing_log:
                    print(f"\nError: {err}", file=processing_log)
                    print(f"abs: {abs_dat_path}", file=processing_log)
        return


if __name__ == "__main__":
    args: EPBDatasetConfig = tyro.cli(ConfiguredEPBDatasetConfig)
    if args.mode == "raw_pkl":
        # this dat file is used to indicate the molecule name
        read_energy_dat_then_process(
            args.abs_dat_paths,
            surface_charge_dat_folders=args.surface_charge_dat_folders,
            atom_pdb_folders=args.atom_pdb_folders,
            atom_potential_dat_folders=args.atom_potential_dat_folders,
            patch_size=args.patch_size,
            sliding_window_zoom_patch_size=args.sliding_window_zoom_patch_size,
            sliding_window_stride=args.sliding_window_stride,
            energy_image_dataset_path=args.created_energy_image_sparse_dataset_path,
            parsing_method=args.parsing_method,
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
