import numpy as np
import torch
from torch import einsum


class ConfusionMetric:
    def get_cm(self, pred, target):
        assert target.size() == pred.size(), 'make sure tensors are one hot encoded'
        target = target.float()
        pred = pred.float()

        # Determine the input dimensions
        if target.dim() == 4:  # 2D input
            tp = torch.einsum("bcxy, bcxy->c", pred, target)
            fp = torch.einsum("bcxy, bcxy->c", pred, (1 - target))
            fn = torch.einsum("bcxy, bcxy->c", (1 - pred), target)
            tn = torch.einsum("bcxy, bcxy->c", pred, (1 - target))
        elif target.dim() == 5:  # 3D input
            tp = torch.einsum("bchwd, bchwd->c", pred, target)
            fp = torch.einsum("bchwd, bchwd->c", pred, (1 - target))
            fn = torch.einsum("bchwd, bchwd->c", (1 - pred), target)
            tn = torch.einsum("bchwd, bchwd->c", pred, (1 - target))
        else:
            raise ValueError("Invalid input shape. Expected 4D (2D) or 5D (3D) tensors.")

        return tp, fp, fn, tn

    def __call__(self, pred, target):
        conf_mat = self.get_cm(pred, target)
        # calculate DICE score per channel
        metric = self.calc(conf_mat)
        # set nan to None
        metric = [m if m == m else None for m in metric]
        return metric

    def calc(self, conf_mat):
        tp, fp, fn, tn = conf_mat
        raise NotImplementedError


class DiceMetric(ConfusionMetric):
    def calc(self, conf_mat):
        tp, fp, fn, _ = conf_mat
        p = tp + fn
        return torch.where(p > 0, 2*tp / (2*tp+fp+fn), torch.tensor(float("nan"), device=tp.device))


class IoUMetric(ConfusionMetric):
    """
    Jaccard index / intersection over union
    """
    def calc(self, conf_mat):
        tp, fp, fn, _ = conf_mat
        p = tp + fn
        return torch.where(p > 0, tp / (tp+fp+fn), torch.tensor(float("nan"), device=tp.device))
