import glob
import logging
import os
import errno
import torch
import torch.nn as nn
import numpy as np
import json
import time

from typing import List, Tuple, Union, Literal
from dataclasses import dataclass, field
from torch.nn import MSELoss, L1Loss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import random_split
from torch.nn.functional import softplus
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from accelerate import Accelerator
from ion_pot.model import PerceptronLoss
from ion_pot.utils import (
    group_pkl_by_mol,
    mae_score,
    eval_r2_score,
    eval_mape_score,
    divisible_by,
    cycle,
    move_dict_to_device,
)
from ion_pot.base import BaseTrainer
from ion_pot.data import (
    ImageMlsesDataset,
    TranslationLabelTransformer,
    Image3dMlsesDataset,
    Image3dEnergyDataset,
    VoxelImage3dEnergyDataset,
    VoxelImage3dEnergySparseDataset,
    VoxelImage3dEnergySparseDatasetForFullyConverage,
    VoxelImage3dEnergyDatasetExtraConfig,
    VoxelImage3dEnergySparseAtomicDataset,
    VoxelImage3dEnergySparseAtomicDatasetForFullyConverage,
)
from ion_pot.data_aug import AtomicDataCollator


class Trainer:
    def __init__(self, args, model):
        # set parsed arguments
        self.args = args

        # init logger and tensorboard
        self._init_logger()
        self._set_writer()

        # init ingredients
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # init model
        self.model = model
        self.model = self.model.to(self.device)

        # init optimizer and learning rate scheduler
        self.optimizer = SGD(self.model.parameters(), lr=self.args.lr)

        # log status
        self.logger.info("Experiment setting:")
        for k, v in sorted(vars(self.args).items()):
            self.logger.info(f"{k}: {v}")

    def _get_lr(self, epoch_index, min_lr=1e-6) -> float:
        start_reduce_epoch = self.args.epoch // 2
        if epoch_index < start_reduce_epoch:
            return self.args.lr

        delta_lr = (self.args.lr - min_lr) / (self.args.epoch - start_reduce_epoch)
        next_lr = self.args.lr - delta_lr * (epoch_index - start_reduce_epoch)
        return next_lr

    def resume(self, resume_ckpt_path: str):
        # resume checkpoint
        self.logger.info(f"Resume model checkpoint from {resume_ckpt_path}...")
        self.model.load_state_dict(torch.load(resume_ckpt_path))

    def train_loop(self, train_dataset, eval_dataset, step_func):
        """Training loop function for model training and finetuning.

        :param train_dataset: training dataset
        :param eval_dataset: evaluation dataset
        :param step_func: a callable function doing forward and optimize step and return loss log
        """
        self.model.train()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.number_worker,
        )

        global_step = 0
        for epoch in range(0, self.args.epoch):
            # update learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self._get_lr(epoch)

            # train steps
            for step, (feat, label, mask) in enumerate(train_dataloader):
                feat: torch.Tensor = feat.to(self.device)
                label: torch.Tensor = label.to(self.device)
                mask: torch.Tensor = mask.to(self.device)

                # run step
                input_feats = {"feat": feat, "label": label, "mask": mask}
                loss_log = step_func(input_feats)

                # print loss
                if step % self.args.log_freq == 0:
                    loss_str = " ".join([f"{k}: {v:.4f}" for k, v in loss_log.items()])
                    self.logger.info(f"Epoch: {epoch} Step: {step} | Loss: {loss_str}")
                    for k, v in loss_log.items():
                        self.writer.add_scalar(f"train/{k}", v, global_step)

                    # log current learning rate
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("train/lr", current_lr, global_step)

                # increase step
                global_step += 1

            if epoch % self.args.eval_freq == 0:
                self.logger.info(f"Evaluate eval dataset at epoch {epoch}...")
                eval_output, _ = self.eval(eval_dataset)
                for k, v in eval_output.items():
                    self.logger.info(f"{k}: {v}")
                    self.writer.add_scalar(f"train/eval_{k}", v, epoch)

                torch.save(self.model.state_dict(), f"{self.args.ckpt_dir}/model_{epoch}.pth")

        # save the final model after training
        torch.save(self.model.state_dict(), f"{self.args.ckpt_dir}/model_final.pth")

    def train_step(self, input_feats):
        feat, label, mask = (
            input_feats["feat"],
            input_feats["label"],
            input_feats["mask"],
        )

        # clean gradient and forward
        self.optimizer.zero_grad()
        pred = self.model(feat)

        # compute weighted loss
        negative_mask = torch.logical_and(mask == 1, label < 1.5)
        positive_mask = torch.logical_and(mask == 1, torch.logical_not(negative_mask))
        regr_loss_fn = MSELoss(reduction="none")
        regr_loss_tn = regr_loss_fn(pred, label)
        weighted_regr_loss_tn = regr_loss_tn * 0.5 * negative_mask + regr_loss_tn * positive_mask
        regr_loss = torch.mean(torch.masked_select(weighted_regr_loss_tn, mask == 1))
        # sign_loss_fn = PerceptronLoss(self.args.sign_threshold)
        # sign_loss = sign_loss_fn(pred, label)
        sign_loss = torch.zeros_like(regr_loss)
        loss = regr_loss + sign_loss * self.args.lambda1

        # backward and update parameters
        loss.backward()
        self.optimizer.step()

        # prepare log dict
        log = {
            "loss": loss.item(),
            "regr_loss": regr_loss.item(),
            "sign_loss": sign_loss.item(),
        }
        return log

    def finetune_step(self, input_feats):
        feat, label = input_feats["feat"], input_feats["label"]

        # clean gradient and forward
        self.optimizer.zero_grad()
        pred = self.model(feat)

        # compute loss
        # regr_loss_fn = L1Loss()
        # regr_loss = regr_loss_fn(pred, label)
        sign_loss_fn = PerceptronLoss(self.args.sign_threshold)
        sign_loss = sign_loss_fn(pred, label)
        regr_loss = torch.zeros_like(sign_loss)
        loss = sign_loss

        # backward and update parameters
        loss.backward()
        self.optimizer.step()

        # prepare log dict
        log = {
            "loss": loss.item(),
            "regr_loss": regr_loss.item(),
            "sign_loss": sign_loss.item(),
        }
        return log

    def train(self, train_dataset, eval_dataset):
        self.train_loop(train_dataset, eval_dataset, self.train_step)

    def finetune(self, train_dataset, eval_dataset):
        self.train_loop(train_dataset, eval_dataset, self.finetune_step)

    @torch.inference_mode()
    def eval(self, dataset, num_workers: int = 0):
        self.model.eval()
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        y_pred_list = []
        y_true_list = []
        for feat, label, mask in tqdm(dataloader):
            feat: torch.Tensor = feat.to(self.device)
            pred = self.model(feat)

            y_pred_list.append(torch.masked_select(pred.cpu(), mask == 1))
            y_true_list.append(torch.masked_select(label, mask == 1))

        y_pred = torch.cat(y_pred_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)
        score = self.eval_on_prediction(y_pred, y_true)

        output = {}
        output["y_pred"] = y_pred
        output["y_true"] = y_true
        return score, output

    @torch.inference_mode()
    def eval_on_prediction(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        mae = mae_score(y_pred, y_true)
        r2 = eval_r2_score(y_pred, y_true)

        threshold = self.args.sign_threshold
        diff_sign_mask = torch.logical_or(
            torch.logical_and(y_true < threshold, y_pred > threshold),
            torch.logical_and(y_true > threshold, y_pred < threshold),
        )
        sign_error_num = diff_sign_mask.float().sum().item()

        score = {}
        score["absolute_mae"] = mae
        score["r2"] = r2
        score["sign_error_num"] = sign_error_num
        return score

    def _set_writer(self):
        self.logger.info("Create writer at '{}'".format(self.args.ckpt_dir))
        self.writer = SummaryWriter(self.args.ckpt_dir)

    def _init_logger(self):
        logging.basicConfig(
            filename=os.path.join(self.args.ckpt_dir, f"mlses_{self.args.mode}.log"),
            level=logging.INFO,
            datefmt="%Y/%m/%d %H:%M:%S",
            format="%(asctime)s: %(name)s [%(levelname)s] %(message)s",
        )
        formatter = logging.Formatter(
            "%(asctime)s: %(name)s [%(levelname)s] %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
        )
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(stream_handler)


@dataclass
class EPBSurfaceTrainerConfig:
    # dataset path
    dataset_path: str = ""

    # patch size
    patch_size: int = 64

    # output path
    output_folder: str = "outputs"

    # dataset split proportions
    dataset_split_proportions: Tuple[float, float, float] = (0.7, 0.1, 0.2)

    # model lr
    train_lr: float = 1e-4

    # use perceptron loss
    use_perceptron_loss: bool = False

    # loss weight for sign loss
    lambda1: float = 1

    # morel adam
    adam_betas: Tuple[float, float] = (0.9, 0.99)

    # train step
    train_num_steps: int = 1000

    # model training batch size
    train_batch_size: int = 64

    # model evaluation batch size
    eval_batch_size: Union[int, None] = 64

    # eval stop at step (used for debugging)
    eval_early_stop: int = -1

    # eval and save model every
    save_and_eval_every: int = 1000

    # probe radius upperbound
    probe_radius_upperbound: float = 1.5

    # probe radius lowerbound
    probe_radius_lowerbound: float = -5

    # dataloader num workers
    num_workers: int = 0

    # device
    use_cuda: bool = True


class EPBSurfaceTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        seed: int,
        *,
        args: EPBSurfaceTrainerConfig,
        has_wandb_writer: bool = False,
    ) -> None:
        super().__init__(args.output_folder, has_wandb_writer)
        # device setting
        self.use_cuda = args.use_cuda
        self.seed = seed

        self.dataloader_worker = args.num_workers
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.save_and_eval_every = args.save_and_eval_every
        self.lambda1 = args.lambda1
        self.eval_early_stop = args.eval_early_stop
        self.use_perceptron_loss = args.use_perceptron_loss

        self.label_transformer = TranslationLabelTransformer(
            args.probe_radius_upperbound,
            args.probe_radius_lowerbound,
        )

        (
            self.sign_threshold,
            train_dataset,
            eval_dataset,
            test_dataset,
        ) = self.build_dataset(
            args.dataset_path,
            args.patch_size,
            args.dataset_split_proportions,
        )
        self.dl = cycle(
            DataLoader(
                train_dataset,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=self.dataloader_worker,
            )
        )
        self.eval_dl = DataLoader(
            eval_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.dataloader_worker,
        )
        self.test_dl = DataLoader(
            test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.dataloader_worker,
        )

        # model settings
        self.model = model.to(self.device)
        self.opt = Adam(self.model.parameters(), lr=args.train_lr, betas=args.adam_betas)

        # step counter state
        self.step = 0
        self.train_num_steps = args.train_num_steps

    def build_dataset(
        self,
        dataset_path: str,
        patch_size: int,
        split_proportions: Tuple[float, float, float],
    ):
        sign_threshold = self.label_transformer.transform(torch.zeros(1)).item()

        dataset = ImageMlsesDataset(
            dataset_path,
            patch_size=patch_size,
            label_transformer=self.label_transformer,
        )

        # split training, developing, and testing datasets
        lengths = [int(p * len(dataset)) for p in split_proportions]
        lengths[-1] = len(dataset) - sum(lengths[:-1])

        generator = torch.Generator().manual_seed(self.seed)
        train_dataset, eval_dataset, test_dataset = random_split(dataset, lengths, generator)
        return sign_threshold, train_dataset, eval_dataset, test_dataset

    def get_state(self):
        state = {
            "step": self.step,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
        }
        return state

    def load_state(self, state):
        self.model.load_state_dict(state["model"])
        self.step = state["step"]
        self.opt.load_state_dict(state["opt"])

    @property
    def device(self):
        return torch.device("cuda") if self.use_cuda else torch.device("cpu")

    @property
    def global_step(self) -> int:
        return self.step

    def set_model_state(self, train: bool = True):
        self.model.train(train)

    @torch.inference_mode()
    def get_metrics(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        mae = mae_score(y_pred, y_true)
        r2 = eval_r2_score(y_pred, y_true)

        threshold = self.sign_threshold
        diff_sign_mask = torch.logical_or(
            torch.logical_and(y_true < threshold, y_pred > threshold),
            torch.logical_and(y_true > threshold, y_pred < threshold),
        )
        sign_error_num = diff_sign_mask.float().sum().item()

        score = {}
        score["absolute_mae"] = mae
        score["r2"] = r2
        score["sign_error_num"] = sign_error_num
        score["sign_error_ratio"] = sign_error_num / len(y_true) * 100
        return score

    @torch.inference_mode()
    def eval_during_training(self):
        outputs = self.eval(self.eval_dl)
        self.set_model_state(True)
        return outputs

    @torch.inference_mode()
    def eval(self, dataloader: DataLoader):
        self.set_model_state(False)
        y_pred_list = []
        y_true_list = []
        preds = []
        labels = []
        masks = []

        step = 0
        for feat, label, mask in tqdm(dataloader):
            if self.eval_early_stop != -1 and step == self.eval_early_stop:
                break

            feat: torch.Tensor = feat.to(self.device)
            pred = self.model(feat)

            preds.append(pred.cpu().numpy())
            labels.append(label.cpu().numpy())
            masks.append(mask.cpu().numpy())
            y_pred_list.append(torch.masked_select(pred.cpu(), mask == 1))
            y_true_list.append(torch.masked_select(label, mask == 1))
            step += 1

        y_pred = torch.cat(y_pred_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)
        score = self.get_metrics(y_pred, y_true)

        output = {}
        output["y_pred"] = y_pred
        output["y_true"] = y_true
        output["preds"] = np.concatenate(preds, axis=0)
        output["labels"] = np.concatenate(labels, axis=0)
        output["masks"] = np.concatenate(masks, axis=0)
        return score, output

    def compute_loss(self, pred, label, mask):
        # compute weighted loss
        negative_mask = torch.logical_and(mask == 1, label < self.sign_threshold)
        positive_mask = torch.logical_and(mask == 1, torch.logical_not(negative_mask))
        regr_loss_fn = MSELoss(reduction="none")
        regr_loss_tn = regr_loss_fn(pred, label)
        weighted_regr_loss_tn = regr_loss_tn * 0.5 * negative_mask + regr_loss_tn * positive_mask
        regr_loss = torch.mean(torch.masked_select(weighted_regr_loss_tn, mask == 1))

        if self.use_perceptron_loss:
            sign_loss_fn = PerceptronLoss(self.sign_threshold)
            sign_loss = sign_loss_fn(pred, label)
        else:
            sign_loss = torch.zeros_like(regr_loss)
        loss = regr_loss + sign_loss * self.lambda1

        loss_dict = {
            "loss": loss.item(),
            "regr_loss": regr_loss.item(),
            "sign_loss": sign_loss.item(),
        }
        return loss, loss_dict

    def train(self):
        self.set_model_state(True)

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                feat, label, mask = next(self.dl)
                feat: torch.Tensor = feat.to(self.device)
                label: torch.Tensor = label.to(self.device)
                mask: torch.Tensor = mask.to(self.device)

                pred = self.model(feat)
                loss, loss_dict = self.compute_loss(pred, label, mask)

                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                pbar.set_description(f"loss: {loss.item():.4f}")
                self.log(loss_dict, section="train")

                if self.step != 0 and divisible_by(self.step, self.save_and_eval_every):
                    scores, _ = self.eval_during_training()
                    self.log(scores, section="eval")
                    milestone = self.step // self.save_and_eval_every
                    self.save(milestone)

                self.step += 1
                pbar.update(1)

        scores, _ = self.eval(self.eval_dl)
        self.log(scores, section="eval")
        scores, _ = self.eval(self.test_dl)
        self.log(scores, section="test")
        self.save("final")
        print("Training done!")


class GENIUSESTrainer(EPBSurfaceTrainer):
    @staticmethod
    def flatten(img: torch.Tensor) -> torch.Tensor:
        """Flatten input tensor into the channel-last format

        :param img: _description_
        :return: _description_
        """
        img = img.permute(0, 2, 3, 1)  # NCHW -> NHWC
        img = img.reshape(-1, img.shape[-1])  # N * H * W, C
        return img

    @staticmethod
    def inv_flatten(img: torch.Tensor, *shape) -> torch.Tensor:
        N, H, W = shape
        img = img.reshape((N, H, W, -1))  # NHWC
        img = img.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return img

    @torch.inference_mode()
    def eval(self, dataloader: DataLoader):
        self.set_model_state(False)
        y_pred_list = []
        y_true_list = []
        preds = []
        labels = []
        masks = []

        step = 0
        for feat, label, mask in tqdm(dataloader):
            if self.eval_early_stop != -1 and step == self.eval_early_stop:
                break

            N, _, H, W = feat.shape
            feat = self.flatten(feat)
            label = self.flatten(label)
            mask = self.flatten(mask)

            feat: torch.Tensor = feat.to(self.device)
            pred = self.model(feat)

            preds.append(self.inv_flatten(pred, N, H, W).cpu().numpy())
            labels.append(self.inv_flatten(label, N, H, W).numpy())
            masks.append(self.inv_flatten(mask, N, H, W).numpy())

            y_pred_list.append(torch.masked_select(pred.cpu(), mask == 1))
            y_true_list.append(torch.masked_select(label, mask == 1))
            step += 1

        y_pred = torch.cat(y_pred_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)
        score = self.get_metrics(y_pred, y_true)

        output = {}
        output["y_pred"] = y_pred
        output["y_true"] = y_true
        output["preds"] = np.concatenate(preds, axis=0)
        output["labels"] = np.concatenate(labels, axis=0)
        output["masks"] = np.concatenate(masks, axis=0)
        return score, output

    def train(self):
        self.set_model_state(True)

        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                feat, label, mask = next(self.dl)
                feat: torch.Tensor = feat.to(self.device)
                label: torch.Tensor = label.to(self.device)
                mask: torch.Tensor = mask.to(self.device)

                feat = self.flatten(feat)
                label = self.flatten(label)
                mask = self.flatten(mask)

                pred = self.model(feat)

                # compute weighted loss
                regr_loss_fn = L1Loss()

                regr_loss = regr_loss_fn(pred, label)
                sign_loss = torch.zeros_like(regr_loss)
                loss = regr_loss + sign_loss * self.lambda1

                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                pbar.set_description(f"loss: {loss.item():.4f}")
                self.log({"loss": loss.item()}, section="train")

                if self.step != 0 and divisible_by(self.step, self.save_and_eval_every):
                    scores, _ = self.eval_during_training()
                    self.log(scores, section="eval")
                    milestone = self.step // self.save_and_eval_every
                    self.save(milestone)

                self.step += 1
                pbar.update(1)

        scores, _ = self.eval(self.eval_dl)
        self.log(scores, section="eval")
        scores, _ = self.eval(self.test_dl)
        self.log(scores, section="test")
        self.save("final")
        print("Training done!")


class EPB3dSurfaceTrainer(EPBSurfaceTrainer):
    def build_dataset(
        self,
        dataset_path: str,
        patch_size: int,
        split_proportions: Tuple[float, float, float],
    ):
        sign_threshold = self.label_transformer.transform(torch.zeros(1)).item()

        dataset = Image3dMlsesDataset(
            dataset_path,
            patch_size=patch_size,
            label_transformer=self.label_transformer,
        )

        # split training, developing, and testing datasets
        lengths = [int(p * len(dataset)) for p in split_proportions]
        lengths[-1] = len(dataset) - sum(lengths[:-1])

        generator = torch.Generator().manual_seed(self.seed)
        train_dataset, eval_dataset, test_dataset = random_split(dataset, lengths, generator)
        return sign_threshold, train_dataset, eval_dataset, test_dataset


class GENIUSES3dTrainer(EPB3dSurfaceTrainer):
    @staticmethod
    def flatten(img: torch.Tensor) -> torch.Tensor:
        """Flatten input tensor into the channel-last format

        :param img: _description_
        :return: _description_
        """
        img = img.permute(0, 2, 3, 4, 1)  # NCDHW -> NDHWC
        img = img.reshape(-1, img.shape[-1])  # N * D * H * W, C
        return img

    @staticmethod
    def inv_flatten(img: torch.Tensor, *shape) -> torch.Tensor:
        N, D, H, W = shape
        img = img.reshape((N, D, H, W, -1))  # NDHWC
        img = img.permute(0, 4, 1, 2, 3)  # NDHWC -> NCDHW
        return img

    @torch.inference_mode()
    def eval(self, dataloader: DataLoader):
        self.set_model_state(False)
        y_pred_list = []
        y_true_list = []
        preds = []
        labels = []
        masks = []

        step = 0
        for feat, label, mask in tqdm(dataloader):
            if self.eval_early_stop != -1 and step == self.eval_early_stop:
                break

            N, _, D, H, W = feat.shape
            feat = self.flatten(feat)
            label = self.flatten(label)
            mask = self.flatten(mask)

            feat: torch.Tensor = feat.to(self.device)
            pred = self.model(feat)

            preds.append(self.inv_flatten(pred, N, D, H, W).cpu().numpy())
            labels.append(self.inv_flatten(label, N, D, H, W).numpy())
            masks.append(self.inv_flatten(mask, N, D, H, W).numpy())

            y_pred_list.append(torch.masked_select(pred.cpu(), mask == 1))
            y_true_list.append(torch.masked_select(label, mask == 1))
            step += 1

        y_pred = torch.cat(y_pred_list, dim=0)
        y_true = torch.cat(y_true_list, dim=0)
        score = self.get_metrics(y_pred, y_true)

        output = {}
        output["y_pred"] = y_pred
        output["y_true"] = y_true
        output["preds"] = np.concatenate(preds, axis=0)
        output["labels"] = np.concatenate(labels, axis=0)
        output["masks"] = np.concatenate(masks, axis=0)
        return score, output


@dataclass
class EPB3dEnergyTrainerConfig(EPBSurfaceTrainerConfig):
    final_train_lr: Union[float, None] = None

    smooth_l1_loss_beta: float = 1.0

    lambda2: float = 1e-1

    lambda3: float = 1

    update_potential_in_constraint: int = 10

    epb_mean: float = -1952.1868788908766

    epb_std: float = 1150.1646687339198

    do_bf16_training: bool = False

    gradient_accumulation_steps: int = 1

    # accelerator
    split_batches: bool = True

    # ddp settings
    find_unused_parameters: bool = False

    # whether do randomly crop in the traindataset
    do_random_crop: bool = False

    # random crop atom threshold
    random_crop_atom_num: Union[int, float] = 10

    # random sample interval
    random_crop_interval: int = 32

    # use sparse dataset
    use_sparse_dataset: bool = False

    # use full coverage sparse dataset for evaluation
    use_full_coverage_sparse_dataset: bool = False

    # use atomic sparse dataset
    use_atomic_sparse_dataset: bool = False

    # full coverage sparse data chunk size
    full_coverage_chunk_size: int = 2

    # pkl filter
    pkl_filter: str = ""

    # grouping suffix used for train/eval/test splitting by molecule id.
    # examples: "_sparse.pkl.gz", "_raw.pkl.gz", "_full.npz"
    dataset_group_pattern: Union[str, None] = None

    # train directly on dense potential maps instead of scalar EPB energy.
    train_on_potential_map: bool = False

    # loss function for dense potential map regression
    potential_map_loss: Literal["smooth_l1", "l1", "mse"] = "smooth_l1"

    # beta for smooth l1 potential map loss
    potential_map_smooth_l1_beta: float = 1e-3

    # if true, apply map loss/metrics only where atom_mask > 0
    potential_map_use_atom_mask: bool = False

    # epsilon for mape denominator
    potential_map_mape_eps: float = 1e-6

    # random rotate voxel grids
    do_random_rotate: bool = False

    # random rotate voxel grids in evaluation
    do_random_rotate_in_eval: bool = False

    # random rotate interval for sampling
    random_rotate_interval: int = 10

    # do specific angle at random rotation
    given_rotate_angle: Union[List[int], None] = None

    # do specific axis at random rotation
    given_rotate_axis: Union[List[str], None] = None

    # rotate voxel grids with specific parameters
    do_fixed_rotate: bool = False

    # rotate voxel grids with specific parameters in evaluation
    do_fixed_rotate_in_eval: bool = False

    # times to rotate voxel grids
    rotate_k: int = 1

    # rotate axis
    rotate_axis: list[int] = field(default_factory=lambda: [0, 1])

    # shrink voxel grids
    do_voxel_grids_shrinking: bool = False

    # randomly scaling voxel grids
    do_random_grid_scaling: bool = False

    # random  scaling voxel grids in evaluation
    do_random_grid_scaling_in_eval: bool = False

    # random scaling range of left (inclusive)
    random_grid_scaling_left: float = 0.15

    # random scaling range of right (exclusive)
    random_grid_scaling_right: float = 1.0

    # random scaling range of interval
    random_grid_scaling_interval: float = 0.05

    # do specific scaling grid size at random scaling
    given_grid_scaling_size: Union[float, None] = None

    # provided data split json path
    data_split_json_path: Union[str, None] = None

    # extra config for voxel image 3d energy dataset
    train_dataset_extra_config: VoxelImage3dEnergyDatasetExtraConfig = field(
        default_factory=VoxelImage3dEnergyDatasetExtraConfig
    )

    # extra config for voxel image 3d energy dataset
    eval_dataset_extra_config: VoxelImage3dEnergyDatasetExtraConfig = field(
        default_factory=VoxelImage3dEnergyDatasetExtraConfig
    )


class EPB3dEnergyTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        seed: int,
        *,
        args: EPB3dEnergyTrainerConfig,
        has_wandb_writer: bool = False,
    ) -> None:
        super().__init__(args.output_folder, has_wandb_writer)
        # set args
        self.args = args

        # device setting
        self.use_cuda = args.use_cuda
        self.seed = seed

        self.dataloader_worker = args.num_workers
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.save_and_eval_every = args.save_and_eval_every
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
        self.update_potential_in_constraint = args.update_potential_in_constraint
        self.eval_early_stop = args.eval_early_stop
        self.use_perceptron_loss = args.use_perceptron_loss
        self.epb_mean = args.epb_mean
        self.epb_std = args.epb_std

        (self.train_dataset, self.eval_dataset, self.test_dataset, self.collate_fn) = (
            self.build_dataset(
                args.dataset_path,
                args.patch_size,
                args.dataset_split_proportions,
            )
        )
        pin_memory = self.device.type == "cuda"
        persistent_workers = self.dataloader_worker > 0
        self.dl = DataLoader(
            self.train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=self.dataloader_worker,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.eval_dl = DataLoader(
            self.eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=self.dataloader_worker,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        self.test_dl = DataLoader(
            self.test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=self.dataloader_worker,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        # model settings
        self.model = model.to(self.device)
        self.opt = Adam(self.model.parameters(), lr=args.train_lr, betas=args.adam_betas)
        if args.final_train_lr is not None:
            gamma = pow(args.final_train_lr / args.train_lr, 1 / args.train_num_steps)
            self.lr_scheduler = StepLR(self.opt, step_size=1, gamma=gamma)
        else:
            self.lr_scheduler = StepLR(self.opt, step_size=1, gamma=1)

        # step counter state
        self.step = 0
        self.train_num_steps = args.train_num_steps

    def build_dataset(
        self,
        dataset_path: str,
        patch_size: int,
        split_proportions: Tuple[float, float, float],
    ):
        pkl_paths = glob.glob(dataset_path)
        dataset = Image3dEnergyDataset(
            pkl_paths,
            patch_size=patch_size,
            epb_mean=self.epb_mean,
            epb_std=self.epb_std,
        )

        # split training, developing, and testing datasets
        lengths = [int(p * len(dataset)) for p in split_proportions]
        lengths[-1] = len(dataset) - sum(lengths[:-1])

        generator = torch.Generator().manual_seed(self.seed)
        train_dataset, eval_dataset, test_dataset = random_split(dataset, lengths, generator)

        collate_fn = None
        return train_dataset, eval_dataset, test_dataset, collate_fn

    def get_state(self):
        state = {
            "step": self.step,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
        }
        return state

    def load_state(self, state):
        self.model.load_state_dict(state["model"])
        self.step = state["step"]
        self.opt.load_state_dict(state["opt"])

    @property
    def device(self):
        return torch.device("cuda") if self.use_cuda else torch.device("cpu")

    @property
    def global_step(self) -> int:
        return self.step

    def set_model_state(self, train: bool = True):
        self.model.train(train)

    @torch.inference_mode()
    def get_metrics(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, atom_num: torch.Tensor = None
    ):
        mae = mae_score(y_pred, y_true)
        mape = eval_mape_score(y_pred, y_true)
        r2 = eval_r2_score(y_pred, y_true)

        score = {}
        score["absolute_mae"] = mae
        score["mape"] = mape
        score["r2"] = r2

        if atom_num is not None:
            mae_loss = L1Loss(reduction="none")
            mae_per_mol = mae_loss(y_pred, y_true)
            score["absolute_mae_per_atom"] = (mae_per_mol / atom_num).mean().item()
            score["mape_per_atom"] = (mae_per_mol / torch.abs(y_true) / atom_num).mean().item()
        return score

    @torch.no_grad()
    def eval_during_training(self):
        outputs = self.eval(self.eval_dl)
        self.set_model_state(True)
        return outputs

    @torch.inference_mode()
    def eval(self, dataloader: DataLoader):
        self.set_model_state(False)
        y_pred_epb_list = []
        y_true_epb_list = []

        step = 0
        for (
            level_set,
            atom_charge,
            atom_type,
            atom_mask,
            potential,
            grid_space,
        ) in tqdm(dataloader):
            if self.eval_early_stop != -1 and step == self.eval_early_stop:
                break

            level_set = level_set.to(self.device)
            atom_charge = atom_charge.to(self.device)
            atom_type = atom_type.to(self.device)
            atom_mask = atom_mask.to(self.device)
            potential = potential.to(self.device)
            grid_space = grid_space.to(self.device)

            pred_epb = self.model(level_set, atom_charge, atom_type, atom_mask, grid_space)
            epb_label = self.model.compute_patch_epb(atom_charge, atom_mask, potential)

            y_pred_epb_list.append(pred_epb.cpu())
            y_true_epb_list.append(epb_label.cpu())
            step += 1

        y_pred_epb = torch.cat(y_pred_epb_list, dim=0)
        y_true_epb = torch.cat(y_true_epb_list, dim=0)
        epb_score = self.get_metrics(y_pred_epb, y_true_epb)
        epb_loss = nn.SmoothL1Loss()(y_pred_epb, y_true_epb).item()

        eval_score = {}
        eval_score.update({f"epb_{k}": v for k, v in epb_score.items()})
        eval_score["epb_loss"] = epb_loss
        eval_score["loss"] = epb_loss

        output = {}
        return eval_score, output

    def compute_loss(self, pred_epb, atom_charge, atom_mask, potential):
        epb_criterion = nn.SmoothL1Loss()
        epb_label = self.model.compute_patch_epb(atom_charge, atom_mask, potential)
        epb_loss = epb_criterion(pred_epb, epb_label)

        # observe the learned covariances
        with torch.no_grad():
            covariances = softplus(self.model.diffusion_sigma).tolist()

        loss = epb_loss
        loss_dict = {
            "loss": loss.item(),
            "epb_loss": epb_loss.item(),
        }
        loss_dict.update({f"covariance_{i}": v for i, v in enumerate(covariances)})
        return loss, loss_dict

    def train(self):
        self.set_model_state(True)

        dl = cycle(self.dl)
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                (
                    level_set,
                    atom_charge,
                    atom_type,
                    atom_mask,
                    potential,
                    grid_space,
                ) = next(dl)

                level_set = level_set.to(self.device)
                atom_charge = atom_charge.to(self.device)
                atom_type = atom_type.to(self.device)
                atom_mask = atom_mask.to(self.device)
                potential = potential.to(self.device)
                grid_space = grid_space.to(self.device)

                pred_epb = self.model(level_set, atom_charge, atom_type, atom_mask, grid_space)
                loss, loss_dict = self.compute_loss(pred_epb, atom_charge, atom_mask, potential)

                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                pbar.set_description(f"loss: {loss.item():.4f}")
                self.log(loss_dict, section="train")

                if self.step != 0 and divisible_by(self.step, self.save_and_eval_every):
                    scores, _ = self.eval_during_training()
                    self.log(scores, section="eval")
                    milestone = self.step // self.save_and_eval_every
                    self.save(milestone)

                self.step += 1
                pbar.update(1)

        scores, _ = self.eval(self.eval_dl)
        self.log(scores, section="eval")
        scores, _ = self.eval(self.test_dl)
        self.log(scores, section="test")
        self.save("final")
        print("Training done!")


class Bf16EPB3dEnergyTrainer(EPB3dEnergyTrainer):
    def __init__(
        self,
        model: nn.Module,
        seed: int,
        *,
        args: EPB3dEnergyTrainerConfig,
        has_wandb_writer: bool = False,
    ) -> None:
        super().__init__(model, seed, args=args, has_wandb_writer=has_wandb_writer)
        self.scaler = GradScaler()

    def get_state(self):
        state = super().get_state()
        state.update({"scaler": self.scaler.state_dict()})
        return state

    def load_state(self, state):
        super().load_state(state)
        self.scaler.load_state_dict(state["scaler"])

    def train(self):
        self.set_model_state(True)

        dl = cycle(self.dl)
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                (
                    level_set,
                    atom_charge,
                    atom_type,
                    atom_mask,
                    potential,
                    grid_space,
                ) = next(dl)

                level_set = level_set.to(self.device)
                atom_charge = atom_charge.to(self.device)
                atom_type = atom_type.to(self.device)
                atom_mask = atom_mask.to(self.device)
                potential = potential.to(self.device)
                grid_space = grid_space.to(self.device)

                with torch.autocast(
                    device_type="cuda" if self.use_cuda else "cpu", dtype=torch.bfloat16
                ):
                    pred_epb = self.model(level_set, atom_charge, atom_type, atom_mask, grid_space)
                    loss, loss_dict = self.compute_loss(pred_epb, atom_charge, atom_mask, potential)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()
                pbar.set_description(f"loss: {loss.item():.4f}")
                self.log(loss_dict, section="train")

                if self.step != 0 and divisible_by(self.step, self.save_and_eval_every):
                    scores, _ = self.eval_during_training()
                    self.log(scores, section="eval")
                    milestone = self.step // self.save_and_eval_every
                    self.save(milestone)

                self.step += 1
                pbar.update(1)

        scores, _ = self.eval(self.eval_dl)
        self.log(scores, section="eval")
        scores, _ = self.eval(self.test_dl)
        self.log(scores, section="test")
        self.save("final")
        print("Training done!")


class AccelerateEPB3dEnergyTrainer(EPB3dEnergyTrainer):
    def __init__(
        self,
        accelerator: Accelerator,
        model: nn.Module,
        seed: int,
        *,
        args: EPB3dEnergyTrainerConfig,
        has_wandb_writer: bool = False,
    ) -> None:
        # accelerator
        self.accelerator = accelerator
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        super().__init__(model, seed, args=args, has_wandb_writer=has_wandb_writer)
        pin_memory = self.device.type == "cuda"

        # fix dataloader with each accelerator process index
        train_persistent_workers = self.dataloader_worker > 0
        self.dl = DataLoader(
            self.train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=self.dataloader_worker,
            collate_fn=self.collate_fn,
            generator=torch.Generator().manual_seed(self.seed + self.accelerator.process_index),
            pin_memory=pin_memory,
            persistent_workers=train_persistent_workers,
        )

        eval_num_workers = (
            1
            if args.use_full_coverage_sparse_dataset and not args.use_atomic_sparse_dataset
            else self.dataloader_worker
        )
        eval_persistent_workers = eval_num_workers > 0
        # set eval and test dataloader with worker num 1 if using fully coverage sparse dataset
        self.eval_dl = DataLoader(
            self.eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=eval_num_workers,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory,
            persistent_workers=eval_persistent_workers,
        )
        self.test_dl = DataLoader(
            self.test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=eval_num_workers,
            collate_fn=self.collate_fn,
            pin_memory=pin_memory,
            persistent_workers=eval_persistent_workers,
        )

        # model settings
        self.model = model
        self.opt = Adam(self.model.parameters(), lr=args.train_lr, betas=args.adam_betas)
        if args.final_train_lr is not None:
            gamma = pow(args.final_train_lr / args.train_lr, 1 / args.train_num_steps)
            self.lr_scheduler = StepLR(self.opt, step_size=1, gamma=gamma)
        else:
            self.lr_scheduler = StepLR(self.opt, step_size=1, gamma=1)

        self.model, self.opt, self.lr_scheduler, self.dl, self.eval_dl, self.test_dl = (
            self.accelerator.prepare(
                self.model,
                self.opt,
                self.lr_scheduler,
                self.dl,
                self.eval_dl,
                self.test_dl,
            )
        )

    @property
    def device(self):
        return self.accelerator.device

    def get_state(self):
        state = {
            "step": self.step,
        }
        return state

    def save(self, milestone: str):
        # save checkpoint for all processes
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.accelerator.save_state()
                break
            except OSError as exc:
                # Accelerate can race while removing old auto-named checkpoints on shared FS.
                if exc.errno != errno.ENOTEMPTY or attempt == max_retries - 1:
                    raise
                self.accelerator.print(
                    f"Checkpoint cleanup race detected ({exc}); retrying save "
                    f"{attempt + 1}/{max_retries - 1}..."
                )
                self.accelerator.wait_for_everyone()
                time.sleep(1.0 + attempt)

        self.accelerator.wait_for_everyone()

        if not self.accelerator.is_main_process:
            return

        # save state from the get_state method
        super().save(milestone)

        # save model
        self.accelerator.save_model(self.model, f"{self.output_folder}/models/model_{milestone}")

    def load_state(self, state):
        self.step = state["step"]

    def load(
        self,
        milestone: str,
        file_name: str = "model.safetensors",
        load_checkpoint: bool = True,
    ):
        # load the state from load_state method
        super().load(milestone)

        # load the latest state from checkpoint folder
        if load_checkpoint:
            self.accelerator.load_state()
        else:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            path_to_checkpoint = os.path.join(
                f"{self.output_folder}/models/model_{milestone}", file_name
            )
            unwrapped_model.load_state_dict(torch.load(path_to_checkpoint))

    def log(self, log_dict: dict, step: int = None, section: str = "train"):
        if not self.accelerator.is_main_process:
            return

        super().log(log_dict, step, section)

    @torch.no_grad()
    def eval(self, dataloader: DataLoader):
        self.accelerator.wait_for_everyone()
        self.set_model_state(False)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if self.args.train_on_potential_map:
            totals = torch.zeros((7,), device=self.device, dtype=torch.float64)
            # totals = [abs_err_sum, sq_err_sum, ape_sum, y_sum, y_sq_sum, n, loss_sum]
            for step, (
                level_set,
                atom_charge,
                atom_type,
                atom_mask,
                potential,
                grid_space,
            ) in enumerate(
                tqdm(
                    dataloader,
                    total=len(dataloader),
                    disable=not self.accelerator.is_main_process,
                )
            ):
                if self.eval_early_stop != -1 and step == self.eval_early_stop:
                    break

                level_set = level_set.to(self.device, non_blocking=True)
                atom_charge = atom_charge.to(self.device, non_blocking=True)
                atom_type = atom_type.to(self.device, non_blocking=True)
                atom_mask = atom_mask.to(self.device, non_blocking=True)
                potential = potential.to(self.device, non_blocking=True)
                grid_space = grid_space.to(self.device, non_blocking=True)

                with self.accelerator.autocast():
                    _, pred_per_epb = self.model(
                        level_set,
                        atom_charge,
                        atom_type,
                        atom_mask,
                        grid_space,
                        return_per_epb=True,
                    )

                map_mask = self.get_potential_map_mask(atom_mask, potential)
                pred_map = torch.masked_select(pred_per_epb.float(), map_mask)
                true_map = torch.masked_select(potential.float(), map_mask)
                if true_map.numel() == 0:
                    continue

                diff = pred_map - true_map
                abs_err = torch.abs(diff)
                totals[0] += abs_err.sum().double()
                totals[1] += torch.sum(diff * diff).double()
                totals[2] += torch.sum(
                    abs_err / torch.abs(true_map).clamp_min(self.args.potential_map_mape_eps)
                ).double()
                totals[3] += true_map.sum().double()
                totals[4] += torch.sum(true_map * true_map).double()
                totals[5] += float(true_map.numel())
                if self.args.potential_map_loss == "l1":
                    loss_map = abs_err
                elif self.args.potential_map_loss == "mse":
                    loss_map = diff * diff
                else:
                    beta = self.args.potential_map_smooth_l1_beta
                    if beta <= 0:
                        loss_map = abs_err
                    else:
                        loss_map = torch.where(
                            abs_err < beta,
                            0.5 * (abs_err * abs_err) / beta,
                            abs_err - 0.5 * beta,
                        )
                totals[6] += loss_map.sum().double()

            if self.accelerator.use_distributed:
                totals = self.accelerator.gather(totals).sum(dim=0)

            abs_err_sum, sq_err_sum, ape_sum, y_sum, y_sq_sum, n, loss_sum = totals.tolist()
            n = max(n, 1.0)
            mae = abs_err_sum / n
            rmse = np.sqrt(sq_err_sum / n)
            mape = ape_sum / n
            ss_tot = y_sq_sum - (y_sum * y_sum) / n
            r2 = float("nan") if ss_tot <= 1e-12 else (1.0 - sq_err_sum / ss_tot)
            pot_map_loss = loss_sum / n

            eval_score = {
                "pot_mae": mae,
                "pot_rmse": rmse,
                "pot_mape": mape,
                "pot_r2": r2,
                "pot_map_loss": pot_map_loss,
                "loss": pot_map_loss,
            }
            output = {}
            return eval_score, output

        samples_seen = 0
        sample_num = len(dataloader.dataset)
        y_pred_epb_list = []
        y_true_epb_list = []
        for step, (
            level_set,
            atom_charge,
            atom_type,
            atom_mask,
            potential,
            grid_space,
        ) in enumerate(
            tqdm(
                dataloader,
                total=len(dataloader),
                disable=not self.accelerator.is_main_process,
            )
        ):
            if self.eval_early_stop != -1 and step == self.eval_early_stop:
                break

            level_set = level_set.to(self.device, non_blocking=True)
            atom_charge = atom_charge.to(self.device, non_blocking=True)
            atom_type = atom_type.to(self.device, non_blocking=True)
            atom_mask = atom_mask.to(self.device, non_blocking=True)
            potential = potential.to(self.device, non_blocking=True)
            grid_space = grid_space.to(self.device, non_blocking=True)

            with self.accelerator.autocast():
                pred_epb = self.model(level_set, atom_charge, atom_type, atom_mask, grid_space)
            epb_label = unwrapped_model.compute_patch_epb(atom_charge, atom_mask, potential)

            # Synchronize predictions across processes
            # need to truncate with the sample number, since `gather` will Accelerate will add samples to make sure each
            # process gets the same batch size. See: https://github.com/huggingface/accelerate/blob/main/examples/by_feature/multi_process_metrics.py
            pred_epb, epb_label = self.accelerator.gather((pred_epb, epb_label))

            if self.accelerator.use_distributed:
                # Then see if we're on the last batch of our eval dataloader
                if step == len(dataloader) - 1:
                    # Last batch needs to be truncated on distributed systems as it contains additional samples
                    pred_epb = pred_epb[: sample_num - samples_seen]
                    epb_label = epb_label[: sample_num - samples_seen]
                else:
                    # Otherwise we add the number of samples seen
                    samples_seen += epb_label.shape[0]

            y_pred_epb_list.append(pred_epb)
            y_true_epb_list.append(epb_label)

        y_pred_epb = torch.cat(y_pred_epb_list, dim=0)
        y_true_epb = torch.cat(y_true_epb_list, dim=0)
        epb_score = self.get_metrics(y_pred_epb, y_true_epb)
        epb_loss = torch.mean(torch.abs(y_pred_epb.float() - y_true_epb.float())).item()

        eval_score = {}
        eval_score.update({f"epb_{k}": v for k, v in epb_score.items()})
        eval_score["epb_loss"] = epb_loss
        eval_score["loss"] = epb_loss
        output = {}
        return eval_score, output

    def get_potential_map_mask(self, atom_mask: torch.Tensor, potential: torch.Tensor):
        if self.args.potential_map_use_atom_mask:
            base_mask = atom_mask > 0
        else:
            base_mask = torch.ones_like(potential, dtype=torch.bool)
        finite_mask = torch.isfinite(potential)
        return base_mask & finite_mask

    def compute_loss(self, pred_epb, atom_charge, atom_mask, potential, pred_per_epb=None):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if self.args.train_on_potential_map:
            if pred_per_epb is None:
                raise ValueError(
                    "When train_on_potential_map is True, pred_per_epb must be provided."
                )

            map_mask = self.get_potential_map_mask(atom_mask, potential).float()
            if self.args.potential_map_loss == "l1":
                per_map_loss = torch.abs(pred_per_epb - potential)
            elif self.args.potential_map_loss == "mse":
                per_map_loss = torch.square(pred_per_epb - potential)
            else:
                map_criterion = nn.SmoothL1Loss(
                    reduction="none",
                    beta=self.args.potential_map_smooth_l1_beta,
                )
                per_map_loss = map_criterion(pred_per_epb, potential)

            epb_loss = torch.sum(per_map_loss * map_mask) / map_mask.sum().clamp_min(1.0)
            loss = epb_loss
            loss_dict = {
                "loss": loss.item(),
                "pot_map_loss": epb_loss.item(),
            }
            if hasattr(unwrapped_model, "diffusion_sigma"):
                with torch.no_grad():
                    covariances = softplus(unwrapped_model.diffusion_sigma).tolist()
                loss_dict.update({f"covariance_{i}": v for i, v in enumerate(covariances)})
            return loss, loss_dict

        if unwrapped_model.is_atom_wise_potential_trained:
            if pred_per_epb is None:
                raise ValueError(
                    "When is_atom_wise_potential_trained is True, pred_per_epb must be provided."
                )
            epb_criterion = nn.SmoothL1Loss(reduction="none")
            per_epb_loss = epb_criterion(pred_per_epb, potential)
            atom_bool_mask = atom_mask > 0
            epb_loss = (
                torch.sum(per_epb_loss * torch.abs(atom_charge) * atom_bool_mask.float())
                / atom_mask.sum()
            )
        else:
            epb_criterion = nn.L1Loss()
            epb_label = unwrapped_model.compute_patch_epb(atom_charge, atom_mask, potential)
            epb_loss = epb_criterion(pred_epb, epb_label)

        loss = epb_loss
        loss_dict = {
            "loss": loss.item(),
            "epb_loss": epb_loss.item(),
        }

        # observe the learned covariances
        if hasattr(unwrapped_model, "diffusion_sigma"):
            with torch.no_grad():
                covariances = softplus(unwrapped_model.diffusion_sigma).tolist()
            loss_dict.update({f"covariance_{i}": v for i, v in enumerate(covariances)})
        return loss, loss_dict

    def train(self):
        self.set_model_state(True)

        dl = cycle(self.dl)
        accum_step = 0
        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not self.accelerator.is_main_process,
        ) as pbar:
            while self.step < self.train_num_steps:
                (
                    level_set,
                    atom_charge,
                    atom_type,
                    atom_mask,
                    potential,
                    grid_space,
                ) = next(dl)

                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        pred_epb, pred_per_epb = self.model(
                            level_set,
                            atom_charge,
                            atom_type,
                            atom_mask,
                            grid_space,
                            return_per_epb=True,
                        )
                        loss, loss_dict = self.compute_loss(
                            pred_epb,
                            atom_charge,
                            atom_mask,
                            potential,
                            pred_per_epb=pred_per_epb,
                        )

                    # visualize learning rate
                    loss_dict.update({"lr": self.lr_scheduler.get_last_lr()[0]})

                    self.accelerator.backward(loss)
                    self.accelerator.wait_for_everyone()
                    self.opt.step()
                    self.opt.zero_grad()
                    self.lr_scheduler.step()

                if (accum_step + 1) % self.gradient_accumulation_steps == 0:
                    pbar.set_description(f"loss: {loss.item():.4f}")
                    self.log(loss_dict, section="train")
                    if self.step != 0 and divisible_by(self.step, self.save_and_eval_every):
                        scores, _ = self.eval_during_training()
                        self.log(scores, section="eval")
                        self.accelerator.print("eval score:", scores)

                        # test score
                        scores, _ = self.eval(self.test_dl)
                        self.log(scores, section="test")
                        self.accelerator.print("test score:", scores)
                        self.set_model_state(True)

                        milestone = self.step // self.save_and_eval_every
                        self.save(milestone)

                    self.accelerator.wait_for_everyone()
                    self.step += 1
                    pbar.update(1)

                accum_step += 1

        # eval at the final step
        scores, _ = self.eval(self.eval_dl)
        self.log(scores, section="eval")
        self.accelerator.print("eval score:", scores)
        scores, _ = self.eval(self.test_dl)
        self.log(scores, section="test")
        self.accelerator.print("test score:", scores)
        self.save("final")
        self.accelerator.print("Training done!")


class Bf16VoxelEPB3dEnergyTrainer(Bf16EPB3dEnergyTrainer):
    def build_dataset(
        self,
        dataset_path: str,
        patch_size: int,
        split_proportions: Tuple[float, float, float],
    ):
        pkl_paths = glob.glob(dataset_path)
        dataset = VoxelImage3dEnergyDataset(
            pkl_paths,
            patch_size=patch_size,
            epb_mean=self.epb_mean,
            epb_std=self.epb_std,
        )

        # split training, developing, and testing datasets
        lengths = [int(p * len(dataset)) for p in split_proportions]
        lengths[-1] = len(dataset) - sum(lengths[:-1])

        generator = torch.Generator().manual_seed(self.seed)
        train_dataset, eval_dataset, test_dataset = random_split(dataset, lengths, generator)

        collate_fn = None
        return train_dataset, eval_dataset, test_dataset, collate_fn


class AccelerateVoxelEPB3dEnergyTrainer(AccelerateEPB3dEnergyTrainer):
    def __init__(
        self,
        accelerator: Accelerator,
        model: nn.Module,
        seed: int,
        *,
        args: EPB3dEnergyTrainerConfig,
        has_wandb_writer: bool = False,
    ) -> None:
        self.do_random_crop = args.do_random_crop
        self.random_crop_atom_num = args.random_crop_atom_num
        self.use_sparse_dataset = args.use_sparse_dataset
        self.random_crop_interval = args.random_crop_interval
        self.use_full_coverage_sparse_dataset = args.use_full_coverage_sparse_dataset
        self.full_coverage_chunk_size = args.full_coverage_chunk_size
        self.pkl_filter = args.pkl_filter

        if (
            self.use_full_coverage_sparse_dataset
            and not args.train_on_potential_map
            and args.eval_batch_size != accelerator.num_processes
        ):
            raise ValueError(
                "When using full coverage sparse dataset for evaluation, eval_batch_size must be equal to the number of processes."
            )

        super().__init__(accelerator, model, seed, args=args, has_wandb_writer=has_wandb_writer)

    @torch.inference_mode()
    def __eval_full_converate_on_one_smaple(
        self,
        level_set: torch.Tensor,
        atom_charge: torch.Tensor,
        atom_type: torch.Tensor,
        atom_mask: torch.Tensor,
        potential: torch.Tensor,
        grid_space: torch.Tensor,
    ):
        level_set = level_set.to(self.device, non_blocking=True)
        atom_charge = atom_charge.to(self.device, non_blocking=True)
        atom_type = atom_type.to(self.device, non_blocking=True)
        atom_mask = atom_mask.to(self.device, non_blocking=True)
        potential = potential.to(self.device, non_blocking=True)
        grid_space = grid_space.to(self.device, non_blocking=True)

        # model
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # squeeze 0 dimension and then split the data
        chunk_size = self.full_coverage_chunk_size
        list_of_level_set = torch.split(level_set.squeeze(0), chunk_size)
        list_of_atom_charge = torch.split(atom_charge.squeeze(0), chunk_size)
        list_of_atom_type = torch.split(atom_type.squeeze(0), chunk_size)
        list_of_atom_mask = torch.split(atom_mask.squeeze(0), chunk_size)
        list_of_potential = torch.split(potential.squeeze(0), chunk_size)
        list_of_grid_space = torch.split(grid_space.squeeze(0), chunk_size)

        pred_epb_list = []
        epb_label_list = []
        start_time = time.time()
        atom_num = 0
        for (
            level_set,
            atom_charge,
            atom_type,
            atom_mask,
            potential,
            grid_space,
        ) in zip(
            list_of_level_set,
            list_of_atom_charge,
            list_of_atom_type,
            list_of_atom_mask,
            list_of_potential,
            list_of_grid_space,
        ):
            # print(grid_space.shape)
            with self.accelerator.autocast():
                pred_epb = self.model(level_set, atom_charge, atom_type, atom_mask, grid_space)
            epb_label = unwrapped_model.compute_patch_epb(atom_charge, atom_mask, potential)
            pred_epb_list.append(pred_epb)
            epb_label_list.append(epb_label)
            atom_num += atom_mask.sum().item()

        pred_epb = torch.sum(torch.cat(pred_epb_list, dim=0)).unsqueeze(
            0
        )  # make sure the shape is (1, )
        end_time = time.time()
        epb_label = torch.sum(torch.cat(epb_label_list, dim=0)).unsqueeze(0)
        execution_time = torch.ones_like(pred_epb) * (end_time - start_time)
        total_atom_num = torch.ones_like(pred_epb, dtype=torch.long) * atom_num

        return pred_epb, epb_label, execution_time, total_atom_num

    @torch.inference_mode()
    def eval_full_converage(self, dataloader: DataLoader):
        self.accelerator.wait_for_everyone()
        self.set_model_state(False)

        samples_seen = 0
        sample_num = len(dataloader.dataset)
        y_pred_epb_list = []
        y_true_epb_list = []
        elapsed_time_list = []
        total_atom_num_list = []
        for step, (
            level_set,
            atom_charge,
            atom_type,
            atom_mask,
            potential,
            grid_space,
        ) in enumerate(
            tqdm(
                dataloader,
                total=len(dataloader),
                disable=not self.accelerator.is_main_process,
            )
        ):
            if self.eval_early_stop != -1 and step == self.eval_early_stop:
                break

            pred_epb, epb_label, execution_time, total_atom_num = (
                self.__eval_full_converate_on_one_smaple(
                    level_set, atom_charge, atom_type, atom_mask, potential, grid_space
                )
            )

            if self.accelerator.use_distributed:
                # Synchronize predictions across processes
                # need to truncate with the sample number, since `gather` will Accelerate will add samples to make sure each
                # process gets the same batch size. See: https://github.com/huggingface/accelerate/blob/main/examples/by_feature/multi_process_metrics.py
                pred_epb, epb_label, execution_time, total_atom_num = self.accelerator.gather(
                    (pred_epb, epb_label, execution_time, total_atom_num)
                )

                # Then see if we're on the last batch of our eval dataloader
                if step == len(dataloader) - 1:
                    # Last batch needs to be truncated on distributed systems as it contains additional samples
                    pred_epb = pred_epb[: sample_num - samples_seen]
                    epb_label = epb_label[: sample_num - samples_seen]
                    execution_time = execution_time[: sample_num - samples_seen]
                    total_atom_num = total_atom_num[: sample_num - samples_seen]
                else:
                    # Otherwise we add the number of samples seen
                    samples_seen += epb_label.shape[0]

            y_pred_epb_list.append(pred_epb)
            y_true_epb_list.append(epb_label)
            elapsed_time_list.append(execution_time)
            total_atom_num_list.append(total_atom_num)

        y_pred_epb = torch.cat(y_pred_epb_list, dim=0)
        y_true_epb = torch.cat(y_true_epb_list, dim=0)
        elapsed_time = torch.cat(elapsed_time_list, dim=0)
        total_atom_num = torch.cat(total_atom_num_list, dim=0)
        epb_score = self.get_metrics(y_pred_epb, y_true_epb, total_atom_num)
        epb_loss = torch.mean(torch.abs(y_pred_epb.float() - y_true_epb.float())).item()

        eval_score = {}
        eval_score.update({f"whole_epb_{k}": v for k, v in epb_score.items()})
        eval_score["whole_epb_loss"] = epb_loss
        eval_score["loss"] = epb_loss
        output = {
            "pred_epb": y_pred_epb.numpy(force=True),
            "true_epb": y_true_epb.numpy(force=True),
            "elapsed_time": elapsed_time.numpy(force=True),
            "atom_num": total_atom_num.numpy(force=True),
        }
        return eval_score, output

    @torch.no_grad()
    def eval(self, dataloader: DataLoader):
        if self.args.train_on_potential_map:
            return super().eval(dataloader)
        if self.use_full_coverage_sparse_dataset:
            return self.eval_full_converage(dataloader)
        else:
            return super().eval(dataloader)

    def split_or_read_pkls(self, dataset_path: str, split_proportions: Tuple[float, float, float]):
        if self.args.data_split_json_path is not None:
            with open(self.args.data_split_json_path, "r") as f:
                splits = json.load(f)
                train_pkls = splits["splits"]["train"]
                eval_pkls = splits["splits"]["eval"]
                test_pkls = splits["splits"]["test"]
        else:
            split_path = f"{self.output_folder}/data_split.json"
            if self.args.dataset_group_pattern is not None:
                pattern = self.args.dataset_group_pattern
            elif self.use_sparse_dataset:
                pattern = "_sparse.pkl.gz"
            elif ".npz" in dataset_path:
                pattern = ".npz"
            else:
                pattern = "_raw.pkl.gz"
            with self.accelerator.main_process_first():
                if not os.path.exists(split_path):
                    generator = torch.Generator().manual_seed(self.seed)

                    # get all pkl paths
                    pkl_paths = glob.glob(dataset_path)

                    # group pkl paths by the same molecule
                    pkl_by_mol = group_pkl_by_mol(pkl_paths, pattern)
                    pkl_keys = list(pkl_by_mol.keys())

                    # get random indices
                    random_indices = torch.randperm(len(pkl_keys), generator=generator).tolist()

                    # split training, developing, and testing datasets
                    lengths = [int(p * len(pkl_keys)) for p in split_proportions]
                    lengths[-1] = len(pkl_keys) - sum(lengths[:-1])
                    cumu_lengths = np.cumsum(lengths)

                    train_pkls = sum(
                        [pkl_by_mol[pkl_keys[i]] for i in random_indices[: cumu_lengths[0]]],
                        [],
                    )
                    eval_pkls = sum(
                        [
                            pkl_by_mol[pkl_keys[i]]
                            for i in random_indices[cumu_lengths[0] : cumu_lengths[1]]
                        ],
                        [],
                    )
                    test_pkls = sum(
                        [
                            pkl_by_mol[pkl_keys[i]]
                            for i in random_indices[cumu_lengths[1] : cumu_lengths[2]]
                        ],
                        [],
                    )

                    # filter pkl paths
                    if self.pkl_filter:
                        train_pkls = [p for p in train_pkls if self.pkl_filter in p]
                        eval_pkls = [p for p in eval_pkls if self.pkl_filter in p]
                        test_pkls = [p for p in test_pkls if self.pkl_filter in p]

                    print(
                        "mol statistics:",
                        f"total: {len(pkl_by_mol)}, split: {lengths}",
                    )
                    print(
                        "pkl statistics:",
                        f"train: {len(train_pkls)}, eval: {len(eval_pkls)}, test: {len(test_pkls)}",
                    )

                    # save the split information
                    with open(split_path, "w") as f:
                        json.dump(
                            {
                                "splits": {
                                    "train": train_pkls,
                                    "eval": eval_pkls,
                                    "test": test_pkls,
                                },
                                "meta_data": {
                                    "total": len(pkl_by_mol),
                                    "split": lengths,
                                    "seed": self.seed,
                                    "train_num": len(train_pkls),
                                    "eval_num": len(eval_pkls),
                                    "test_num": len(test_pkls),
                                },
                            },
                            f,
                            indent=4,
                        )
                else:
                    with open(split_path, "r") as f:
                        splits = json.load(f)
                        train_pkls = splits["splits"]["train"]
                        eval_pkls = splits["splits"]["eval"]
                        test_pkls = splits["splits"]["test"]

        return train_pkls, eval_pkls, test_pkls

    def build_dataset(
        self,
        dataset_path: str,
        patch_size: int,
        split_proportions: Tuple[float, float, float],
    ):
        train_pkls, eval_pkls, test_pkls = self.split_or_read_pkls(dataset_path, split_proportions)
        dataset_class = (
            VoxelImage3dEnergyDataset
            if not self.use_sparse_dataset
            else VoxelImage3dEnergySparseDataset
        )
        eval_dataset_class = (
            VoxelImage3dEnergyDataset
            if not self.use_sparse_dataset
            else (
                VoxelImage3dEnergySparseDataset
                if not self.use_full_coverage_sparse_dataset
                else VoxelImage3dEnergySparseDatasetForFullyConverage
            )
        )

        train_dataset = dataset_class(
            train_pkls,
            patch_size=patch_size,
            epb_mean=self.epb_mean,
            epb_std=self.epb_std,
            do_random_crop=self.do_random_crop,
            random_crop_atom_num=self.random_crop_atom_num,
            random_crop_interval=self.random_crop_interval,
            do_random_rotate=self.args.do_random_rotate,
            random_rotate_interval=self.args.random_rotate_interval,
            do_fixed_rotate=self.args.do_fixed_rotate,
            rotate_k=self.args.rotate_k,
            rotate_axis=self.args.rotate_axis,
            do_voxel_grids_shrinking=self.args.do_voxel_grids_shrinking,
            do_random_grid_scaling=self.args.do_random_grid_scaling,
            random_grid_scaling_left=self.args.random_grid_scaling_left,
            random_grid_scaling_right=self.args.random_grid_scaling_right,
            random_grid_scaling_interval=self.args.random_grid_scaling_interval,
            extra_config=self.args.train_dataset_extra_config,
        )

        test_dataset_kwargs = dict(
            patch_size=patch_size,
            epb_mean=self.epb_mean,
            epb_std=self.epb_std,
            do_random_crop=False,
            do_random_rotate=self.args.do_random_rotate_in_eval,
            given_rotate_angle=self.args.given_rotate_angle,
            given_rotate_axis=self.args.given_rotate_axis,
            do_fixed_rotate=self.args.do_fixed_rotate_in_eval,
            rotate_k=self.args.rotate_k,
            rotate_axis=self.args.rotate_axis,
            do_voxel_grids_shrinking=self.args.do_voxel_grids_shrinking,
            do_random_grid_scaling=self.args.do_random_grid_scaling_in_eval,
            given_grid_scaling_size=self.args.given_grid_scaling_size,
            extra_config=self.args.eval_dataset_extra_config,
        )
        eval_dataset = eval_dataset_class(eval_pkls, **test_dataset_kwargs)
        test_dataset = eval_dataset_class(test_pkls, **test_dataset_kwargs)

        collate_fn = None

        return train_dataset, eval_dataset, test_dataset, collate_fn


class AccelerateAtomicEPB3dEnergyTrainer(AccelerateVoxelEPB3dEnergyTrainer):
    def build_dataset(
        self,
        dataset_path: str,
        patch_size: int,
        split_proportions: Tuple[float, float, float],
    ):
        train_pkls, eval_pkls, test_pkls = self.split_or_read_pkls(dataset_path, split_proportions)
        dataset_class = VoxelImage3dEnergySparseAtomicDataset
        eval_dataset_class = VoxelImage3dEnergySparseAtomicDatasetForFullyConverage

        train_dataset = dataset_class(
            train_pkls,
            patch_size=patch_size,
            epb_mean=self.epb_mean,
            epb_std=self.epb_std,
            do_random_crop=self.do_random_crop,
            random_crop_atom_num=self.random_crop_atom_num,
            random_crop_interval=self.random_crop_interval,
            do_random_rotate=self.args.do_random_rotate,
            random_rotate_interval=self.args.random_rotate_interval,
            do_fixed_rotate=self.args.do_fixed_rotate,
            rotate_k=self.args.rotate_k,
            rotate_axis=self.args.rotate_axis,
            do_voxel_grids_shrinking=self.args.do_voxel_grids_shrinking,
            do_random_grid_scaling=self.args.do_random_grid_scaling,
            random_grid_scaling_left=self.args.random_grid_scaling_left,
            random_grid_scaling_right=self.args.random_grid_scaling_right,
            random_grid_scaling_interval=self.args.random_grid_scaling_interval,
            extra_config=self.args.train_dataset_extra_config,
        )

        test_dataset_kwargs = dict(
            patch_size=patch_size,
            epb_mean=self.epb_mean,
            epb_std=self.epb_std,
            do_random_crop=False,
            do_random_rotate=self.args.do_random_rotate_in_eval,
            given_rotate_angle=self.args.given_rotate_angle,
            given_rotate_axis=self.args.given_rotate_axis,
            do_fixed_rotate=self.args.do_fixed_rotate_in_eval,
            rotate_k=self.args.rotate_k,
            rotate_axis=self.args.rotate_axis,
            do_voxel_grids_shrinking=self.args.do_voxel_grids_shrinking,
            do_random_grid_scaling=self.args.do_random_grid_scaling_in_eval,
            given_grid_scaling_size=self.args.given_grid_scaling_size,
            extra_config=self.args.eval_dataset_extra_config,
        )
        eval_dataset = eval_dataset_class(eval_pkls, **test_dataset_kwargs)
        test_dataset = eval_dataset_class(test_pkls, **test_dataset_kwargs)

        if (
            self.args.train_dataset_extra_config.create_boundary_features
            != self.args.eval_dataset_extra_config.create_boundary_features
        ):
            raise ValueError(
                "The boundary feature creation should be the same in both training and evaluation datasets."
            )
        create_boundary_feature = self.args.train_dataset_extra_config.create_boundary_features
        collate_fn = AtomicDataCollator(create_boundary_feature)

        return train_dataset, eval_dataset, test_dataset, collate_fn

    @torch.no_grad()
    def __eval_full_converate_on_one_smaple(
        self,
        batch_dict: dict[torch.Tensor],
    ):
        batch_dict = move_dict_to_device(batch_dict, self.device)

        # model
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        start_time = time.time()
        atom_num = 0
        # print(grid_space.shape)
        with self.accelerator.autocast():
            pred_epb = self.model(
                batch_of_atom_indices=batch_dict["batch_of_atom_indices"],
                level_set=batch_dict["level_set"],
                atom_charge=batch_dict["atom_charge"],
                atom_type=batch_dict["atom_type"],
                atom_mask=batch_dict["atom_mask"],
                atom_xyz=batch_dict["atom_xyz"],
                grid_space=batch_dict["grid_space"],
                batch_dict=batch_dict,
            )
        epb_label = unwrapped_model.compute_patch_epb(
            batch_dict["atom_charge"],
            batch_dict["atom_mask"],
            batch_dict["atom_potential"],
            batch_dict["batch_of_atom_indices"],
        )

        atom_num += batch_dict["atom_mask"].sum().item()
        pred_epb = pred_epb.sum().unsqueeze(0)
        end_time = time.time()
        epb_label = epb_label.sum().unsqueeze(0)
        execution_time = torch.ones_like(pred_epb) * (end_time - start_time)
        total_atom_num = torch.ones_like(pred_epb, dtype=torch.long) * atom_num

        return pred_epb, epb_label, execution_time, total_atom_num

    @torch.no_grad()
    def eval_full_converage(self, dataloader: DataLoader):
        self.accelerator.wait_for_everyone()
        self.set_model_state(False)

        samples_seen = 0
        sample_num = len(dataloader.dataset)
        y_pred_epb_list = []
        y_true_epb_list = []
        elapsed_time_list = []
        total_atom_num_list = []
        for step, batch_dict in enumerate(
            tqdm(
                dataloader,
                total=len(dataloader),
                disable=not self.accelerator.is_main_process,
            )
        ):
            if self.eval_early_stop != -1 and step == self.eval_early_stop:
                break

            pred_epb, epb_label, execution_time, total_atom_num = (
                self.__eval_full_converate_on_one_smaple(batch_dict)
            )

            if self.accelerator.use_distributed:
                # Synchronize predictions across processes
                # need to truncate with the sample number, since `gather` will Accelerate will add samples to make sure each
                # process gets the same batch size. See: https://github.com/huggingface/accelerate/blob/main/examples/by_feature/multi_process_metrics.py
                pred_epb, epb_label, execution_time, total_atom_num = self.accelerator.gather(
                    (pred_epb, epb_label, execution_time, total_atom_num)
                )

                # Then see if we're on the last batch of our eval dataloader
                if step == len(dataloader) - 1:
                    # Last batch needs to be truncated on distributed systems as it contains additional samples
                    pred_epb = pred_epb[: sample_num - samples_seen]
                    epb_label = epb_label[: sample_num - samples_seen]
                    execution_time = execution_time[: sample_num - samples_seen]
                    total_atom_num = total_atom_num[: sample_num - samples_seen]
                else:
                    # Otherwise we add the number of samples seen
                    samples_seen += epb_label.shape[0]

            y_pred_epb_list.append(pred_epb)
            y_true_epb_list.append(epb_label)
            elapsed_time_list.append(execution_time)
            total_atom_num_list.append(total_atom_num)

        y_pred_epb = torch.cat(y_pred_epb_list, dim=0)
        y_true_epb = torch.cat(y_true_epb_list, dim=0)
        elapsed_time = torch.cat(elapsed_time_list, dim=0)
        total_atom_num = torch.cat(total_atom_num_list, dim=0)
        epb_score = self.get_metrics(y_pred_epb, y_true_epb, total_atom_num)
        epb_loss = torch.mean(torch.abs(y_pred_epb.float() - y_true_epb.float())).item()

        eval_score = {}
        eval_score.update({f"whole_epb_{k}": v for k, v in epb_score.items()})
        eval_score["whole_epb_loss"] = epb_loss
        eval_score["loss"] = epb_loss
        output = {
            "pred_epb": y_pred_epb.numpy(force=True),
            "true_epb": y_true_epb.numpy(force=True),
            "elapsed_time": elapsed_time.numpy(force=True),
            "atom_num": total_atom_num.numpy(force=True),
        }
        return eval_score, output

    def compute_loss(
        self,
        pred_epb,
        atom_charge,
        atom_mask,
        potential,
        batch_of_atom_indices,
        pred_per_epb=None,
    ):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        if unwrapped_model.is_atom_wise_potential_trained:
            if pred_per_epb is None:
                raise ValueError(
                    "When is_atom_wise_potential_trained is True, pred_per_epb must be provided."
                )
            epb_criterion = nn.SmoothL1Loss(reduction="none", beta=self.args.smooth_l1_loss_beta)
            per_epb_loss = epb_criterion(pred_per_epb, potential)
            atom_bool_mask = atom_mask > 0
            epb_loss = (
                torch.sum(per_epb_loss * torch.abs(atom_charge) * atom_bool_mask.float())
                / atom_mask.sum()
            )
        else:
            epb_criterion = nn.L1Loss()
            epb_label = unwrapped_model.compute_patch_epb(
                atom_charge, atom_mask, potential, batch_of_atom_indices
            )
            epb_loss = epb_criterion(pred_epb, epb_label)

        loss = epb_loss
        loss_dict = {
            "loss": loss.item(),
            "epb_loss": epb_loss.item(),
        }

        # observe the learned covariances
        if hasattr(unwrapped_model, "diffusion_sigma"):
            with torch.no_grad():
                covariances = softplus(unwrapped_model.diffusion_sigma).tolist()
            loss_dict.update({f"covariance_{i}": v for i, v in enumerate(covariances)})
        return loss, loss_dict

    def train(self):
        self.set_model_state(True)

        dl = cycle(self.dl)
        accum_step = 0
        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not self.accelerator.is_main_process,
        ) as pbar:
            while self.step < self.train_num_steps:
                batch_dict = next(dl)

                batch_dict = move_dict_to_device(batch_dict, self.device)
                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        pred_epb, pred_per_epb = self.model(
                            batch_of_atom_indices=batch_dict["batch_of_atom_indices"],
                            level_set=batch_dict["level_set"],
                            atom_charge=batch_dict["atom_charge"],
                            atom_type=batch_dict["atom_type"],
                            atom_mask=batch_dict["atom_mask"],
                            atom_xyz=batch_dict["atom_xyz"],
                            grid_space=batch_dict["grid_space"],
                            return_per_epb=True,
                            batch_dict=batch_dict,
                        )
                        loss, loss_dict = self.compute_loss(
                            pred_epb,
                            batch_dict["atom_charge"],
                            batch_dict["atom_mask"],
                            batch_dict["atom_potential"],
                            batch_dict["batch_of_atom_indices"],
                            pred_per_epb=pred_per_epb,
                        )

                    # visualize learning rate
                    loss_dict.update({"lr": self.lr_scheduler.get_last_lr()[0]})

                    self.accelerator.backward(loss)
                    self.accelerator.wait_for_everyone()
                    self.opt.step()
                    self.opt.zero_grad()
                    self.lr_scheduler.step()

                if (accum_step + 1) % self.gradient_accumulation_steps == 0:
                    pbar.set_description(f"loss: {loss.item():.4f}")
                    self.log(loss_dict, section="train")
                    if self.step != 0 and divisible_by(self.step, self.save_and_eval_every):
                        scores, _ = self.eval_during_training()
                        self.log(scores, section="eval")
                        self.accelerator.print("eval score:", scores)

                        # test score
                        scores, _ = self.eval(self.test_dl)
                        self.log(scores, section="test")
                        self.accelerator.print("test score:", scores)
                        self.set_model_state(True)

                        milestone = self.step // self.save_and_eval_every
                        self.save(milestone)

                    self.accelerator.wait_for_everyone()
                    self.step += 1
                    pbar.update(1)

                accum_step += 1

        # eval at the final step
        scores, _ = self.eval(self.eval_dl)
        self.log(scores, section="eval")
        self.accelerator.print("eval score:", scores)
        scores, _ = self.eval(self.test_dl)
        self.log(scores, section="test")
        self.accelerator.print("test score:", scores)
        self.save("final")
        self.accelerator.print("Training done!")
