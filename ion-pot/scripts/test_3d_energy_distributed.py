from typing import Union
import os
import shutil
import tyro
import joblib

from dataclasses import asdict
from nntool.slurm import slurm_function
from nntool.wandb import init_wandb
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed, ProjectConfiguration
from safetensors.torch import load
from ion_pot.model import MultiScaleConv3dEPBModel, MultiScaleAtomicEPBModel
from ion_pot.trainer import (
    EPB3dEnergyTrainer,
    Bf16EPB3dEnergyTrainer,
    AccelerateVoxelEPB3dEnergyTrainer,
    AccelerateAtomicEPB3dEnergyTrainer,
)
from ion_pot.configs.epb_3d.config import (
    ConfiguredEvalExperimentConfig,
    EvalExperimentConfig,
    ExperimentConfig,
)


def use_wandb() -> bool:
    if str(os.getenv("WANDB_DISABLED", "")).lower() in {"1", "true", "yes"}:
        return False
    if str(os.getenv("WANDB_MODE", "")).lower() == "disabled":
        return False
    if str(os.getenv("PBGNN_DISABLE_WANDB", "")).lower() in {"1", "true", "yes"}:
        return False
    return True


def run(args: Union[EvalExperimentConfig, ExperimentConfig]):
    # use the same seed for all processes
    set_seed(args.seed)
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(
            split_batches=args.trainer.split_batches
        ),
        project_config=ProjectConfiguration(
            args.trainer.output_folder, automatic_checkpoint_naming=True, total_limit=10
        ),
        gradient_accumulation_steps=args.trainer.gradient_accumulation_steps,
    )

    has_wandb_writer = accelerator.is_main_process and use_wandb()
    if has_wandb_writer:
        try:
            init_wandb(args.wandb, asdict(args))
        except Exception as exc:
            print(f"W&B init failed ({exc}). Continuing without W&B.")
            has_wandb_writer = False

    if (
        args.trainer.use_full_coverage_sparse_dataset
        and args.trainer.eval_batch_size != accelerator.num_processes
    ):
        print(
            "Adjusting trainer.eval_batch_size from "
            f"{args.trainer.eval_batch_size} to {accelerator.num_processes} "
            "to match accelerator num_processes for full-coverage evaluation."
        )
        args.trainer.eval_batch_size = accelerator.num_processes

    # reload model checkpoint
    if args.model_cls == "atomic_based":
        energy_model = MultiScaleAtomicEPBModel(args.atomic_energy_model)
    else:
        energy_model = MultiScaleConv3dEPBModel(args.energy_model)
    with open(args.model_ckpt_path, "rb") as f:
        data = f.read()
    energy_model.load_state_dict(load(data))

    TRAINER_CLS = {
        "patch_based": (
            Bf16EPB3dEnergyTrainer
            if args.trainer.do_bf16_training
            else EPB3dEnergyTrainer
        ),
        "voxel_based": AccelerateVoxelEPB3dEnergyTrainer,
        "atomic_based": AccelerateAtomicEPB3dEnergyTrainer,
    }
    trainer_cls = TRAINER_CLS[args.model_cls]
    trainer = trainer_cls(
        accelerator,
        energy_model,
        args.seed,
        args=args.trainer,
        has_wandb_writer=has_wandb_writer,
    )

    # eval the model checkpoint
    scores, eval_output = trainer.eval_full_converage(trainer.eval_dl)
    trainer.log(scores, section="eval")
    trainer.accelerator.print("eval score:", scores)
    scores, test_output = trainer.eval_full_converage(trainer.test_dl)
    trainer.log(scores, section="test")
    trainer.accelerator.print("test score:", scores)

    # save ouputs
    outputs = {
        "eval": eval_output,
        "test": test_output,
    }
    joblib.dump(outputs, f"{trainer.output_folder}/epb_outputs.joblib")


@slurm_function
def main(args: Union[EvalExperimentConfig, ExperimentConfig]):
    return run(args)


def launch(args: Union[EvalExperimentConfig, ExperimentConfig]):
    slurm_mode = str(getattr(args.slurm, "mode", "")).lower()
    if slurm_mode == "slurm" and shutil.which("sbatch") is None:
        print("Slurm mode requested but 'sbatch' was not found. Falling back to local execution.")
        run(args)
        return

    try:
        main(args.slurm)(args)
    except KeyError as exc:
        if exc.args == ("submitit",):
            print(
                "Submitit plugin discovery failed (missing 'submitit' entry points). "
                "Falling back to local execution."
            )
            run(args)
        else:
            raise


if __name__ == "__main__":
    args: EvalExperimentConfig = tyro.parse(ConfiguredEvalExperimentConfig)
    launch(args)
