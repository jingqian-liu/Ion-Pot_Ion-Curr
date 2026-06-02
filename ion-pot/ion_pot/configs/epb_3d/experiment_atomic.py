from dataclasses import replace
from .experiment_config import ExperimentConfig, EvalExperimentConfig
from . import model_atomic
from . import trainer_atomic
from . import misc

experiments = dict(
    # pbsmall
    pbsmall=ExperimentConfig(
        model_cls="atomic_based",
        atomic_energy_model=model_atomic.bndy_fusion_model_small,
        trainer=trainer_atomic.pbsmall_atomic_trainer,
        wandb=misc.wandb,
        slurm=misc.distributed_slurm,
    ),
    debug_pbsmall=ExperimentConfig(
        model_cls="atomic_based",
        atomic_energy_model=model_atomic.bndy_fusion_model_small,
        trainer=replace(
            trainer_atomic.pbsmall_atomic_trainer,
            eval_batch_size=1,
        ),
        wandb=misc.wandb,
        slurm=misc.debug_slurm,
    ),
    pbsmall_single_gpu=ExperimentConfig(
        model_cls="atomic_based",
        atomic_energy_model=model_atomic.bndy_fusion_model_small,
        trainer=replace(
            trainer_atomic.pbsmall_atomic_trainer,
            eval_batch_size=1,
        ),
        wandb=misc.wandb,
        slurm=misc.atomic_slurm,
    ),
    # amber_pbsa
    amber_pbsa=ExperimentConfig(
        model_cls="atomic_based",
        atomic_energy_model=model_atomic.model_medium,
        trainer=trainer_atomic.amber_pbsa_atomic_trainer,
        wandb=misc.wandb,
        slurm=misc.distributed_slurm,
    ),
    debug_amber_pbsa=ExperimentConfig(
        model_cls="atomic_based",
        atomic_energy_model=model_atomic.fusion_model_medium,
        trainer=trainer_atomic.amber_pbsa_atomic_trainer,
        wandb=misc.wandb,
        slurm=misc.distributed_slurm,
    ),
    amber_pbsa_single_gpu=ExperimentConfig(
        model_cls="atomic_based",
        atomic_energy_model=model_atomic.model_medium,
        trainer=replace(trainer_atomic.amber_pbsa_atomic_trainer, eval_batch_size=1),
        wandb=misc.wandb,
        slurm=misc.atomic_slurm,
    ),
)
