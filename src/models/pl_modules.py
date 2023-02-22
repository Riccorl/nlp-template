from typing import Any, Union

import hydra
import pytorch_lightning as pl
import torch
import transformers as tr
from omegaconf import DictConfig
from torch.optim import RAdam, AdamW

from data.labels import Labels


class BasePLModule(pl.LightningModule):
    def __init__(
        self,
        model: Union[torch.nn.Module, DictConfig],
        labels: Labels = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.labels = labels
        if isinstance(model, DictConfig):
            self.model = hydra.utils.instantiate(model, labels=labels)
        else:
            self.model = model

    def forward(self, **kwargs) -> dict:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.

        """
        return self.model(**kwargs)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        forward_output = self.forward(**{**batch, "return_loss": True})
        self.log("loss", forward_output["loss"])
        return forward_output["loss"]

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        forward_output = self.forward(**{**batch, "return_loss": True})
        self.log("val_loss", forward_output["loss"])

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        forward_output = self.forward(**{**batch, "return_loss": True})
        self.log("test_loss", forward_output["loss"])

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.parameters(), _convert_="partial"
        )
        if "lr_scheduler" not in self.hparams:
            return [optimizer]

        lr_scheduler = hydra.utils.instantiate(
            self.hparams.lr_scheduler, optimizer=optimizer
        )
        return [optimizer], [lr_scheduler]
