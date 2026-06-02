from .experiment_config import ExperimentConfig, EvalExperimentConfig
from . import model_unet
from . import trainer
from . import misc

experiments = dict(
    unet_psz64_ctx32_protein_complex_wo_lset_fully_coverage_rotation_augmented_medium=ExperimentConfig(
        model_cls="voxel_based",
        energy_model=model_unet.epb_psz64_ctx32_all_in_one_wo_lset_model_medium,
        trainer=trainer.epb_model_psz64_ctx32_sparse_protein_complex_fully_coverage_rotation_augmented_trainer_medium,
        wandb=misc.wandb,
        slurm=misc.distributed_slurm,
    ),
    unet_psz32_ctx48_protein_complex_wo_lset_fully_coverage_rotation_augmented_medium=ExperimentConfig(
        model_cls="voxel_based",
        energy_model=model_unet.epb_psz64_ctx32_all_in_one_wo_lset_model_medium,
        trainer=trainer.epb_model_psz32_ctx48_sparse_protein_complex_fully_coverage_rotation_augmented_trainer_medium,
        wandb=misc.wandb,
        slurm=misc.distributed_slurm,
    ),
    unet_psz32_ctx48_protein_complex_with_lset_fully_coverage_rotation_augmented_medium=ExperimentConfig(
        model_cls="voxel_based",
        energy_model=model_unet.epb_psz64_ctx32_all_in_one_with_lset_model_medium,
        trainer=trainer.epb_model_psz32_ctx48_sparse_protein_complex_fully_coverage_rotation_augmented_trainer_medium,
        wandb=misc.wandb,
        slurm=misc.distributed_slurm,
    ),
    unet_psz64_ctx32_all_in_one_wo_lset_fully_coverage_rotation_augmented_medium=ExperimentConfig(
        model_cls="voxel_based",
        energy_model=model_unet.epb_psz64_ctx32_all_in_one_wo_lset_model_medium,
        trainer=trainer.epb_model_psz64_ctx32_sparse_all_in_one_fully_coverage_rotation_augmented_trainer_medium,
        wandb=misc.wandb,
        slurm=misc.distributed_slurm,
    ),
)
