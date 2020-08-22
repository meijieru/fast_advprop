import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


def adjust_learning_rate(
    optimizer, epoch, args, state, batch=None, nBatch=None
):
    if args.lr_schedule == "cos":
        raise NotImplementedError()
    elif args.lr_schedule == "step":
        if epoch in args.schedule:
            state["lr"] *= args.gamma
            for param_group in optimizer.param_groups:
                param_group["lr"] *= args.gamma
    else:
        raise ValueError()


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler

    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1, start_lr=0.1):
        self.total_iters = total_iters
        self.start_lr = start_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        ret = [
            (
                self.start_lr
                + (base_lr - self.start_lr)
                * self.last_epoch
                / (self.total_iters + 1e-8)
                if base_lr > self.start_lr
                else base_lr
            )
            for base_lr in self.base_lrs
        ]
        return ret


def label_smoothing_cross_entropy(
    pred, target, classes, dim, reduction="batchmean", smoothing=0.1
):
    """Implement a cross_entropy with label smoothing.

    adopted from https://github.com/OpenNMT/OpenNMT-py/blob/e8622eb5c6117269bb3accd8eb6f66282b5e67d9/onmt/utils/loss.py#L186
    and https://github.com/pytorch/pytorch/issues/7455
    """
    confidence = 1.0 - smoothing
    pred = pred.log_softmax(dim=dim)
    with torch.no_grad():
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), confidence)
    return F.kl_div(pred, true_dist, reduction=reduction)
