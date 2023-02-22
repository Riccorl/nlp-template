from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from data.labels import Labels
import transformers as tr


class BaseModel(torch.nn.Module):
    def __init__(
        self,
        loss_fn: torch.nn.Module,
        labels: Labels,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.labels: Optional[Labels] = None
        if labels is not None:
            self.labels = labels
        self.loss_fn = loss_fn

    def forward(
        self,
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        return_predictions: bool = False,
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            labels (`torch.Tensor`):
                The labels of the sentences.
            return_predictions (`bool`):
                Whether to compute the predictions.
            return_loss (`bool`):
                Whether to compute the loss.

        Returns:
            obj:`torch.Tensor`: The outputs of the model.
        """

        logits: Optional[torch.Tensor] = None
        output = {"logits": logits}

        if return_predictions:
            predictions = logits.argmax(dim=-1)
            output["predictions"] = predictions

        if return_loss and labels is not None:
            output["loss"] = self.loss_fn(logits, labels)

        return output
