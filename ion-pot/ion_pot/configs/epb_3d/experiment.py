from .experiment_config import ExperimentConfig, EvalExperimentConfig
from . import model
from . import model_unet
from .model import (
    epb_model,
    epb_psz128_model,
    epb_psz128_model_medium,
    epb_psz128_all_in_one_model_medium,
    epb_psz64_small_mol_model_medium,
    epb_psz64_model,
    epb_psz64_model_medium,
)
from . import trainer
from .trainer import (
    epb_model_trainer,
    epb_model_debug_trainer,
    epb_model_wo_random_crop_trainer,
    epb_model_psz128_sparse_trainer,
    epb_model_psz128_sparse_trainer_medium,
    epb_model_psz128_sparse_all_in_one_trainer_medium,
    epb_model_psz128_sparse_all_in_one_eval_trainer_medium,
    epb_model_psz128_sparse_all_in_one_fully_coverage_trainer_medium,
    epb_model_psz128_sparse_all_in_one_fully_coverage_rotation_augmented_trainer_medium,
    epb_model_psz64_sparse_small_mol_fully_coverage_rotation_augmented_trainer_medium,
    epb_model_psz128_sparse_all_in_one_grid035_fully_coverage_trainer_medium,
    epb_model_psz128_sparse_all_in_one_rotate90_eval_trainer_medium,
    epb_model_psz128_sparse_all_in_one_rotate_any_eval_trainer_medium,
    epb_model_psz128_sparse_small_mol_rotate_any_eval_trainer_medium,
    epb_model_psz64_sparse_trainer,
    epb_model_psz64_sparse_trainer_medium,
    epb_model_psz64_sparse_debug_trainer_medium,
)
from .misc import (
    debug_slurm,
    distributed_slurm,
    debug_distributed_slurm,
    distributed_eval_slurm,
    wandb,
)

experiments = {}

experiments["3d_energy_voxel_distributed_training"] = ExperimentConfig(
    model_cls="voxel_based",
    energy_model=epb_model,
    trainer=epb_model_trainer,
    wandb=wandb,
    slurm=distributed_slurm,
)

experiments["3d_energy_voxel_distributed_training_psz128_sparse_dataset"] = (
    ExperimentConfig(
        model_cls="voxel_based",
        energy_model=epb_psz128_model,
        trainer=epb_model_psz128_sparse_trainer,
        wandb=wandb,
        slurm=distributed_slurm,
    )
)

experiments["3d_energy_voxel_distributed_training_psz128_sparse_dataset_medium"] = (
    ExperimentConfig(
        model_cls="voxel_based",
        energy_model=epb_psz128_model_medium,
        trainer=epb_model_psz128_sparse_trainer_medium,
        wandb=wandb,
        slurm=distributed_slurm,
    )
)

experiments[
    "3d_energy_voxel_distributed_training_psz128_sparse_dataset_all_in_one_medium"
] = ExperimentConfig(
    model_cls="voxel_based",
    energy_model=epb_psz128_all_in_one_model_medium,
    trainer=epb_model_psz128_sparse_all_in_one_trainer_medium,
    wandb=wandb,
    slurm=distributed_slurm,
)

experiments[
    "3d_energy_voxel_distributed_training_psz128_sparse_dataset_all_in_one_fully_coverage_medium"
] = ExperimentConfig(
    model_cls="voxel_based",
    energy_model=epb_psz128_all_in_one_model_medium,
    trainer=epb_model_psz128_sparse_all_in_one_fully_coverage_trainer_medium,
    wandb=wandb,
    slurm=distributed_slurm,
)

experiments[
    "3d_energy_voxel_distributed_training_psz128_sparse_dataset_all_in_one_fully_coverage_rotation_augmented_medium"
] = ExperimentConfig(
    model_cls="voxel_based",
    energy_model=epb_psz128_all_in_one_model_medium,
    trainer=epb_model_psz128_sparse_all_in_one_fully_coverage_rotation_augmented_trainer_medium,
    wandb=wandb,
    slurm=distributed_slurm,
)

experiments[
    "3d_energy_voxel_distributed_training_psz128_sparse_dataset_all_in_one_wo_lset_fully_coverage_rotation_augmented_medium"
] = ExperimentConfig(
    model_cls="voxel_based",
    energy_model=model.epb_psz128_all_in_one_wo_lset_model_medium,
    trainer=epb_model_psz128_sparse_all_in_one_fully_coverage_rotation_augmented_trainer_medium,
    wandb=wandb,
    slurm=distributed_slurm,
)

experiments[
    "3d_energy_voxel_distributed_training_psz128_sparse_dataset_protein_fully_coverage_rotation_augmented_medium"
] = ExperimentConfig(
    model_cls="voxel_based",
    energy_model=epb_psz128_all_in_one_model_medium,
    trainer=trainer.epb_model_psz128_sparse_protein_fully_coverage_rotation_augmented_trainer_medium,
    wandb=wandb,
    slurm=distributed_slurm,
)

experiments[
    "3d_energy_voxel_distributed_training_psz128_sparse_dataset_protein_wo_lset_fully_coverage_rotation_augmented_medium"
] = ExperimentConfig(
    model_cls="voxel_based",
    energy_model=model.epb_psz128_all_in_one_wo_lset_model_medium,
    trainer=trainer.epb_model_psz128_sparse_protein_fully_coverage_rotation_augmented_trainer_medium,
    wandb=wandb,
    slurm=distributed_slurm,
)

experiments[
    "3d_energy_voxel_distributed_training_psz128_sparse_dataset_nucleicacid_wo_lset_fully_coverage_rotation_augmented_medium"
] = ExperimentConfig(
    model_cls="voxel_based",
    energy_model=model.epb_psz128_all_in_one_wo_lset_model_medium,
    trainer=trainer.epb_model_psz128_sparse_nucleicacid_fully_coverage_rotation_augmented_trainer_medium,
    wandb=wandb,
    slurm=distributed_slurm,
)

experiments[
    "3d_energy_voxel_distributed_training_psz128_sparse_dataset_protein_complex_wo_lset_fully_coverage_rotation_augmented_medium"
] = ExperimentConfig(
    model_cls="voxel_based",
    energy_model=model.epb_psz128_all_in_one_wo_lset_model_medium,
    trainer=trainer.epb_model_psz128_sparse_protein_complex_fully_coverage_rotation_augmented_trainer_medium,
    wandb=wandb,
    slurm=distributed_slurm,
)

experiments[
    "3d_energy_voxel_distributed_training_psz128_sparse_dataset_protein_complex_wo_lset_fully_coverage_rotation_augmented_medium_model_v2"
] = ExperimentConfig(
    model_cls="voxel_based",
    energy_model=model_unet.epb_psz128_all_in_one_wo_lset_model_medium,
    trainer=trainer.epb_model_psz128_sparse_protein_complex_fully_coverage_rotation_augmented_trainer_medium,
    wandb=wandb,
    slurm=distributed_slurm,
)

experiments[
    "3d_energy_voxel_distributed_training_psz64_sparse_dataset_protein_complex_wo_lset_fully_coverage_rotation_augmented_medium"
] = ExperimentConfig(
    model_cls="voxel_based",
    energy_model=model.epb_psz64_all_in_one_wo_lset_model_medium,
    trainer=trainer.epb_model_psz64_sparse_protein_complex_fully_coverage_rotation_augmented_trainer_medium,
    wandb=wandb,
    slurm=distributed_slurm,
)

experiments[
    "3d_energy_voxel_distributed_training_psz64_sparse_dataset_small_mol_fully_coverage_rotation_augmented_medium"
] = ExperimentConfig(
    model_cls="voxel_based",
    energy_model=epb_psz64_small_mol_model_medium,
    trainer=epb_model_psz64_sparse_small_mol_fully_coverage_rotation_augmented_trainer_medium,
    wandb=wandb,
    slurm=distributed_slurm,
)

experiments[
    "3d_energy_voxel_distributed_training_psz64_sparse_dataset_small_mol_wo_lset_fully_coverage_rotation_augmented_medium"
] = ExperimentConfig(
    model_cls="voxel_based",
    energy_model=model.epb_psz64_small_mol_wo_lset_model_medium,
    trainer=epb_model_psz64_sparse_small_mol_fully_coverage_rotation_augmented_trainer_medium,
    wandb=wandb,
    slurm=distributed_slurm,
)

experiments[
    "3d_energy_voxel_distributed_training_psz128_sparse_dataset_all_in_one_grid035_fully_coverage_medium"
] = ExperimentConfig(
    model_cls="voxel_based",
    energy_model=epb_psz128_all_in_one_model_medium,
    trainer=epb_model_psz128_sparse_all_in_one_grid035_fully_coverage_trainer_medium,
    wandb=wandb,
    slurm=distributed_slurm,
)

experiments["3d_energy_voxel_distributed_training_psz64_sparse_dataset"] = (
    ExperimentConfig(
        model_cls="voxel_based",
        energy_model=epb_psz64_model,
        trainer=epb_model_psz64_sparse_trainer,
        wandb=wandb,
        slurm=distributed_slurm,
    )
)

experiments["3d_energy_voxel_distributed_training_psz64_sparse_dataset_medium"] = (
    ExperimentConfig(
        model_cls="voxel_based",
        energy_model=epb_psz64_model_medium,
        trainer=epb_model_psz64_sparse_trainer_medium,
        wandb=wandb,
        slurm=distributed_slurm,
    )
)

experiments["3d_energy_voxel_distributed_training_wo_random_crop"] = ExperimentConfig(
    model_cls="voxel_based",
    energy_model=epb_model,
    trainer=epb_model_wo_random_crop_trainer,
    wandb=wandb,
    slurm=distributed_slurm,
)

experiments["3d_energy_voxel_distributed_training_debug"] = ExperimentConfig(
    model_cls="voxel_based",
    energy_model=epb_psz64_model_medium,
    trainer=epb_model_psz64_sparse_debug_trainer_medium,
    wandb=wandb,
    slurm=debug_slurm,
)

eval_experiments = dict(
    eval_3d_energy_voxel_distributed_training_psz128_sparse_dataset_all_in_one_medium=EvalExperimentConfig(
        model_ckpt_path="",
        model_cls="voxel_based",
        energy_model=epb_psz128_all_in_one_model_medium,
        trainer=epb_model_psz128_sparse_all_in_one_eval_trainer_medium,
        wandb=wandb,
        slurm=distributed_eval_slurm,
    ),
    eval_3d_energy_voxel_distributed_training_psz128_sparse_dataset_all_in_one_medium_rotate90=EvalExperimentConfig(
        model_ckpt_path="",
        model_cls="voxel_based",
        energy_model=epb_psz128_all_in_one_model_medium,
        trainer=epb_model_psz128_sparse_all_in_one_rotate90_eval_trainer_medium,
        wandb=wandb,
        slurm=distributed_eval_slurm,
    ),
    eval_3d_energy_voxel_distributed_training_psz128_sparse_dataset_all_in_one_medium_rotate_any=EvalExperimentConfig(
        model_ckpt_path="",
        model_cls="voxel_based",
        energy_model=epb_psz128_all_in_one_model_medium,
        trainer=epb_model_psz128_sparse_all_in_one_rotate_any_eval_trainer_medium,
        wandb=wandb,
        slurm=distributed_eval_slurm,
    ),
    eval_3d_energy_voxel_distributed_training_psz128_sparse_dataset_all_in_one_medium_rotate_any_small_mol=EvalExperimentConfig(
        model_ckpt_path="",
        model_cls="voxel_based",
        energy_model=epb_psz128_all_in_one_model_medium,
        trainer=epb_model_psz128_sparse_small_mol_rotate_any_eval_trainer_medium,
        wandb=wandb,
        slurm=distributed_eval_slurm,
    ),
    eval_3d_energy_voxel_distributed_training_psz64_sparse_dataset_small_mol_medium_rotate_any=EvalExperimentConfig(
        model_ckpt_path="",
        model_cls="voxel_based",
        energy_model=epb_psz64_small_mol_model_medium,
        trainer=epb_model_psz64_sparse_small_mol_fully_coverage_rotation_augmented_trainer_medium,
        wandb=wandb,
        slurm=distributed_eval_slurm,
    ),
    eval_3d_energy_voxel_distributed_training_psz128_sparse_dataset_all_in_one_medium_debug=EvalExperimentConfig(
        model_ckpt_path="",
        model_cls="voxel_based",
        energy_model=epb_psz128_all_in_one_model_medium,
        trainer=epb_model_psz128_sparse_all_in_one_eval_trainer_medium,
        wandb=wandb,
        slurm=distributed_eval_slurm,
    ),
)
