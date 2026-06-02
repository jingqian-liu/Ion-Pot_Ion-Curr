from typing import Literal
from dataclasses import dataclass, field
from nntool.slurm import SlurmConfig
from nntool.wandb import WandbConfig
from ion_pot.model import MultiScaleConv3dEPBModelConfig, MultiScaleAtomicEPBModelConfig
from ion_pot.trainer import EPB3dEnergyTrainerConfig


@dataclass
class ExperimentConfig:
    model_cls: Literal["patch_based", "voxel_based", "atomic_based"] = "patch_based"

    energy_model: MultiScaleConv3dEPBModelConfig = field(
        default_factory=MultiScaleConv3dEPBModelConfig
    )

    atomic_energy_model: MultiScaleAtomicEPBModelConfig = field(
        default_factory=MultiScaleAtomicEPBModelConfig
    )

    trainer: EPB3dEnergyTrainerConfig = field(default_factory=EPB3dEnergyTrainerConfig)

    wandb: WandbConfig = field(default_factory=WandbConfig)

    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    # trainer should be resumed
    resume_trainer: bool = False

    # trainer checkpoint path
    trainer_ckpt_path: str = ""

    # random seed
    seed: int = 2024

    # model checkpoint path for testing
    model_ckpt_path: str = ""


@dataclass
class EvalExperimentConfig:
    # model checkpoint path
    model_ckpt_path: str

    model_cls: Literal["patch_based", "voxel_based"] = "patch_based"

    energy_model: MultiScaleConv3dEPBModelConfig = field(
        default_factory=MultiScaleConv3dEPBModelConfig
    )

    trainer: EPB3dEnergyTrainerConfig = field(default_factory=EPB3dEnergyTrainerConfig)

    wandb: WandbConfig = field(default_factory=WandbConfig)

    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    # random seed
    seed: int = 2024
