from typing import Optional

import torch

Tensor = torch.Tensor
import numpy as np


def tversky_loss(
        input: Tensor,
        target: Tensor,
        smooth: torch.float,
        alpha: torch.float,
        beta: torch.float,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
) -> Tensor:

    input = torch.sigmoid(input)
    target = torch.sigmoid(target)
    tp = torch.sum(input*target)
    fp = torch.sum(input*(torch.tensor(1.0) - target))
    fn = torch.sum((torch.tensor(1.0) - input)*target)

    up = tp + smooth
    down = tp + alpha*fn + beta*fp + smooth

    # assert fp >= 0, "FP negative equals {}".format(fp)
    # # assert fn >= 0, "FN negative equals {}".format(fn)
    # # assert tp >= 0, "TP negative equals {}".format(tp)
    # # assert fp < 0, "FP negative equals {}".format(fp)
    # # assert fn < 0, "FN negative equals {}".format(fn)
    # # assert tp < 0, "TP negative equals {}".format(tp)
    # input = torch.sigmoid(input)
    #
    # # tversky = torch.relu((tp+smooth) / (tp + alpha*fp + beta*fn + smooth))
    # up = (2 * torch.sum(input*target) + smooth)
    # down = ((torch.sum(input + target)) + smooth)

    assert down != 0, "division by 0!"

    tversky = up / down
    # assert 0 <= tversky, "tversky wrong equals {}".format(tversky)
    return tversky


def dice_loss(
        input: Tensor,
        target: Tensor,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = "mean",
) -> Tensor:
    r"""Dice loss"""


if __name__ == "__main__":
    preds = torch.round(torch.rand(4, requires_grad=True))
    label = torch.round(torch.rand(4))
    print(tversky_loss(preds, label, 1e-5, 0.5, 0.5))
    print(label - torch.tensor(1.4))
