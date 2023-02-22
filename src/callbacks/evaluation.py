# from pytorch_lightning import Callback, LightningModule
import os
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from lightning_fabric.utilities.apply_func import move_data_to_device
from torch.utils.data import DataLoader
from tqdm import tqdm

# from faiss.indexer import FaissIndexer

from utils.logging import get_console_logger

logger = get_console_logger()


class EvaluationCallback(pl.Callback):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def __call__(
        self,
        dataloaders: List[DataLoader],
        pl_module: pl.LightningModule,
        *args,
        **kwargs,
    ) -> dict:
        # It should return a dictionary of metrics
        # E.g. {"recall": 0.5, "precision": 0.5}
        # metrics = {}
        # return metrics
        raise NotImplementedError

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = self(trainer.val_dataloaders, pl_module)
        metrics = {f"val_{k}": v for k, v in metrics.items()}
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self, trainer, pl_module):
        metrics = self(trainer.test_dataloaders, pl_module)
        metrics = {f"test_{k}": v for k, v in metrics.items()}
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
