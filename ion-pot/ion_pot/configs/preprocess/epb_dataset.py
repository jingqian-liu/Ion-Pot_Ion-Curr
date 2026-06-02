import tyro

from typing import List, Literal
from dataclasses import dataclass, field


@dataclass
class EPBDatasetConfig:
    mode: Literal["raw_pkl"] = "raw_pkl"
    parsing_method: Literal["old", "new", "small_mol"] = "old"

    # data sources
    abs_dat_paths: List[str] = field(
        default_factory=lambda: [
            "datasets/benchmark_data_0.9/bench_full_0.55_abs/*/*.dat",
        ]
    )
    surface_charge_dat_folders: List[str] = field(
        default_factory=lambda: [
            "datasets/benchmark_surf_charge/surf_charge_055/protein_test/",
        ]
    )
    atom_potential_dat_folders: List[str] = field(
        default_factory=lambda: [
            "datasets/benchmark_surf_potential/surf_potential_055/protein_test"
        ]
    )
    atom_pdb_folders: List[str] = field(
        default_factory=lambda: [
            "datasets/raw_pdb/protein-testCase/",
        ]
    )

    # output paths
    created_energy_image_dataset_path: str = "datasets/benchmark_3d_energy/"
    created_energy_image_sparse_dataset_path: str = "datasets/benchmark_3d_energy/"

    # other parameters
    patch_size: int = 64
    sliding_window_zoom_patch_size: int = 32
    sliding_window_stride: int = 32


configurations = dict(
    # protein + nucleic acid + protein_protein epb dataset
    amber_pbsa_epb_sparse_datset=EPBDatasetConfig(
        mode="raw_pkl",
        parsing_method="new",
        abs_dat_paths=[
            "datasets/refined-mlses_data/bench_full_abs_cpu_075/protein_test/*/*.dat",
            "datasets/refined-mlses_data/bench_full_abs_cpu_075/nucleicacidtest/*/*.dat",
            "datasets/refined-mlses_data/bench_full_abs_cpu_075/protein_protein_test/*/*.dat",
            "datasets/refined-mlses_data/bench_full_abs_cpu_055/protein_test/*/*.dat",
            "datasets/refined-mlses_data/bench_full_abs_cpu_055/nucleicacidtest/*/*.dat",
            "datasets/refined-mlses_data/bench_full_abs_cpu_055/protein_protein_test/*/*.dat",
            "datasets/refined-mlses_data/bench_full_abs_cpu_035/protein_test/*/*.dat",
            "datasets/refined-mlses_data/bench_full_abs_cpu_035/nucleicacidtest/*/*.dat",
            "datasets/refined-mlses_data/bench_full_abs_cpu_035/protein_protein_test/*/*.dat",
        ],
        surface_charge_dat_folders=[
            "",
        ]
        * 9,
        atom_potential_dat_folders=[
            "datasets/refined-mlses_data/surf_potential_075/protein_test",
            "datasets/refined-mlses_data/surf_potential_075/nucleicacidtest",
            "datasets/refined-mlses_data/surf_potential_075/protein_protein_test",
            "datasets/refined-mlses_data/surf_potential_055/protein_test",
            "datasets/refined-mlses_data/surf_potential_055/nucleicacidtest",
            "datasets/refined-mlses_data/surf_potential_055/protein_protein_test",
            "datasets/refined-mlses_data/surf_potential_035/protein_test",
            "datasets/refined-mlses_data/surf_potential_035/nucleicacidtest",
            "datasets/refined-mlses_data/surf_potential_035/protein_protein_test",
        ],
        atom_pdb_folders=[
            "datasets/raw_pdb/protein-testCase",
            "datasets/raw_pdb/nucleicacidtest",
            "datasets/raw_pdb/protein_protein_test",
        ]
        * 3,
        created_energy_image_sparse_dataset_path="datasets/processed/new_full_3d_energy_sparse",
    ),
    # small molecules
    pbsmall_epb_sparse_datset=EPBDatasetConfig(
        mode="raw_pkl",
        parsing_method="small_mol",
        abs_dat_paths=[
            "datasets/small_mol_data/bench_full_abs_cpu_095/*/*.dat",
            "datasets/small_mol_data/bench_full_abs_cpu_075/*/*.dat",
            "datasets/small_mol_data/bench_full_abs_cpu_055/*/*.dat",
            "datasets/small_mol_data/bench_full_abs_cpu_035/*/*.dat",
        ],
        surface_charge_dat_folders=[
            "",
        ]
        * 4,
        atom_potential_dat_folders=[
            "datasets/small_mol_data/surf_potential_cpu_095",
            "datasets/small_mol_data/surf_potential_cpu_075",
            "datasets/small_mol_data/surf_potential_cpu_055",
            "datasets/small_mol_data/surf_potential_cpu_035",
        ],
        atom_pdb_folders=[
            "datasets/small_mol_test-pqr",
        ]
        * 4,
        created_energy_image_sparse_dataset_path="datasets/processed/small_mol_full_3d_energy_sparse",
    ),
)


ConfiguredEPBDatasetConfig = tyro.extras.subcommand_type_from_defaults(configurations)
