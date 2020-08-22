"""Implement AdvProp Two BNs scheme."""

from functools import partial

import torch
import torch.nn as nn


def to_status(m: nn.Module, status: str) -> None:
    """Check `MixBatchNorm2d` for details."""
    if isinstance(m, MixBatchNorm2d):
        m.set_batch_type(status)


to_clean_status = partial(to_status, status="clean")
to_adv_status = partial(to_status, status="adv")
to_mix_status = partial(to_status, status="mix")


class MixBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d with original & auxiliary branches.

    `batch_type` must match the input type, and could be the following values:
        - `clean`: use the original BNs
        - `adv`: use the auxiliary BNs
        - `mix`: the first half of the inputs use the original BNs, while the others use the auxiliary BNs.
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.aux_bn = nn.BatchNorm2d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.set_batch_type("clean")

    def set_batch_type(self, batch_type: str):
        if batch_type not in ["adv", "clean", "mix"]:
            raise ValueError()
        self.batch_type = batch_type

    def forward(self, input):
        if self.batch_type == "adv":
            input = self.aux_bn(input)
        elif self.batch_type == "clean":
            input = super().forward(input)
        else:
            assert self.batch_type == "mix"
            batch_size = input.shape[0]
            input0 = super().forward(input[: batch_size // 2])
            input1 = self.aux_bn(input[batch_size // 2 :])
            input = torch.cat((input0, input1), 0)
        return input
