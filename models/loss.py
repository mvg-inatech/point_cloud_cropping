import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from models.lovasz_loss import LovaszSoftmax


def get_loss_function(name: str, **kwargs) -> nn.Module:

    if "weight" in kwargs:
        # has to be tensor
        if kwargs["weight"] is not None:
            kwargs["weight"] = torch.tensor(kwargs["weight"])
        else:
            del kwargs["weight"]

    if name == "cross_entropy":
        return CrossEntropyLoss(**kwargs)
    elif name == "cross_entropy_probability_weighted":
        return CrossEntropyProbabilityWeighted(**kwargs)
    elif name == "cross_entropy_lovasz":
        return CrossEntropyLovasz(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {name}")


@dataclass
class LossConfig:
    probabilities: Optional[torch.Tensor] = None
    factor: float = 1.0


class BaseLoss(ABC, nn.Module):
    @abstractmethod
    def compute_loss(
        self, pred: torch.Tensor, label: torch.Tensor, config: LossConfig
    ) -> torch.Tensor:
        pass

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        config: Optional[LossConfig] = None,
    ) -> torch.Tensor:
        if config is None:
            config = LossConfig()
        return self.compute_loss(pred, label, config)


class CrossEntropyLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(**kwargs)

    def compute_loss(
        self, pred: torch.Tensor, label: torch.Tensor, config: LossConfig
    ) -> torch.Tensor:
        return self.ce(pred, label)


class CrossEntropyProbabilityWeighted(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="none", **kwargs)

    def compute_loss(
        self, pred: torch.Tensor, label: torch.Tensor, config: LossConfig
    ) -> torch.Tensor:
        if config.probabilities is None:
            raise ValueError("probabilities required for weighted loss")
        ce = self.ce(pred, label)
        return (ce * config.probabilities).mean()


class CrossEntropyLovasz(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__()
        self.ls = LovaszSoftmax(ignore=kwargs.get("ignore_index", None))
        self.ce = nn.CrossEntropyLoss(**kwargs)

    def compute_loss(
        self, pred: torch.Tensor, label: torch.Tensor, config: LossConfig
    ) -> torch.Tensor:
        ls = self.ls(pred.softmax(dim=1), label)
        ce = self.ce(pred, label)
        return ce + config.factor * ls
