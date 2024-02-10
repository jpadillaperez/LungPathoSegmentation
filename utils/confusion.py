import numpy as np
import torch
from torch import einsum


class ConfusionMetric:
    def get_cm(self, pred, target):
        assert target.size() == pred.size(), 'make sure tensors are one hot encoded'
        target = target.float()
        pred = pred.float()

        # get conditions
        tp = einsum("bcxy, bcxy->c", pred, target)
        fp = einsum("bcxy, bcxy->c", pred, (1 - target))
        fn = einsum("bcxy, bcxy->c", (1 - pred), target)
        tn = einsum("bcxy, bcxy->c", pred, (1 - target))

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
