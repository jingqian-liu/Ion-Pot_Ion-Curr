import json
import os
import torch
import warnings

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

from typing import Dict, Optional, Set
from pathlib import Path
from abc import abstractmethod


class BaseTrainer(object):
    def __init__(self, output_folder: str, has_wandb_writer: bool = False) -> None:
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self._metrics_jsonl_handle: Optional[object] = None
        self._tb_writer: Optional[SummaryWriter] = None

        self.has_writer = has_wandb_writer
        if self.has_writer and wandb is None:
            warnings.warn("wandb is not installed; disabling wandb logging.")
            self.has_writer = False

        if str(os.getenv("PBGNN_DISABLE_TENSORBOARD", "")).lower() not in {"1", "true", "yes"}:
            if SummaryWriter is None:
                warnings.warn("tensorboard is not installed; disabling tensorboard logging.")
            else:
                self._tb_writer = SummaryWriter(log_dir=str(self.output_folder / "tensorboard"))

        if str(os.getenv("PBGNN_DISABLE_JSONL_LOGGING", "")).lower() not in {"1", "true", "yes"}:
            self._metrics_jsonl_handle = open(
                self.output_folder / "metrics.jsonl", "a", encoding="utf-8"
            )

        self._wandb_defined_metrics: Set[str] = set()
        if self.has_writer:
            self._init_wandb()

    def _get_wandb_step_name(self, wandb_section: str = "trainer_state") -> str:
        return f"{wandb_section}/{self.__class__.__name__}_step"

    def _to_scalar(self, value):
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return float(value.detach().cpu().mean().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _init_wandb(self):
        wandb.define_metric(self._get_wandb_step_name())
        self._wandb_defined_metrics.add(self._get_wandb_step_name())

    def log(self, log_dict: dict, step: int = None, section: str = "train"):
        step_value = step if step is not None else self.global_step

        # add section to each metric and convert to scalar values
        scalar_log_dict = {}
        for key, value in log_dict.items():
            scalar_value = self._to_scalar(value)
            if scalar_value is None:
                continue
            scalar_log_dict[f"{section}/{key}"] = scalar_value

        if self._tb_writer is not None:
            for key, value in scalar_log_dict.items():
                self._tb_writer.add_scalar(key, value, step_value)
            self._tb_writer.flush()

        if self._metrics_jsonl_handle is not None:
            self._metrics_jsonl_handle.write(
                json.dumps(
                    {
                        "step": int(step_value),
                        "section": section,
                        "metrics": scalar_log_dict,
                    }
                )
                + "\n"
            )
            self._metrics_jsonl_handle.flush()

        if not self.has_writer:
            return

        # define metrics against custom step name
        for key in scalar_log_dict.keys():
            if key not in self._wandb_defined_metrics:
                wandb.define_metric(key, step_metric=self._get_wandb_step_name())
                self._wandb_defined_metrics.add(key)

        scalar_log_dict.update({self._get_wandb_step_name(): step_value})
        wandb.log(scalar_log_dict)

    def save(self, milestone: str):
        state = self.get_state()
        torch.save(state, str(self.output_folder / f"model-{milestone}.pt"))

    def load(self, milestone: str):
        state = torch.load(
            str(self.output_folder / f"model-{milestone}.pt"), map_location=self.device
        )
        self.load_state(state)

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def set_model_state(self, train: bool = True):
        raise NotImplementedError

    @torch.inference_mode()
    @abstractmethod
    def eval(self, dataloader: torch.utils.data.DataLoader):
        raise NotImplementedError

    @torch.inference_mode()
    @abstractmethod
    def eval_during_training(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def device(self) -> torch.DeviceObjType:
        raise NotImplementedError

    @property
    @abstractmethod
    def global_step(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> Dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def load_state(self, state: Dict[str, object]):
        raise NotImplementedError

    def __del__(self):
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._metrics_jsonl_handle is not None:
            self._metrics_jsonl_handle.close()
