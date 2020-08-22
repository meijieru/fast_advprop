import abc
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SCALE = 2.0 / 255


def get_kernel(size, nsig, mode="gaussian", device="cuda:0"):
    if mode == "gaussian":
        # since we have to normlize all the numbers
        # there is no need to calculate the const number like \pi and \sigma.
        vec = torch.linspace(-nsig, nsig, steps=size).to(device)
        vec = torch.exp(-vec * vec / 2)
        res = vec.view(-1, 1) @ vec.view(1, -1)
        res = res / torch.sum(res)
    elif mode == "linear":
        # originally, res[i][j] = (1-|i|/(k+1)) * (1-|j|/(k+1))
        # since we have to normalize it
        # calculate res[i][j] = (k+1-|i|)*(k+1-|j|)
        vec = (size + 1) / 2 - torch.abs(
            torch.arange(-(size + 1) / 2, (size + 1) / 2 + 1, step=1)
        ).to(device)
        res = vec.view(-1, 1) @ vec.view(1, -1)
        res = res / torch.sum(res)
    else:
        raise ValueError("no such mode in get_kernel.")
    return res


class AttackerBase(abc.ABC):
    def __init__(
        self,
        num_iter: int = 0,
        num_classes: typing.Optional[int] = None,
        original_attack: bool = False,
    ):
        self.num_iter = num_iter
        self.num_classes = num_classes
        self.original_attack = original_attack

    def is_grad_reusable(self) -> bool:
        """Whether gradients are reuabled."""
        return self.original_attack

    @abc.abstractmethod
    def attack_init(
        self, image_clean: torch.Tensor, label: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Init the perturbation."""
        raise NotImplementedError()

    @abc.abstractmethod
    def attack_update(
        self,
        image_clean: torch.Tensor,
        adv: torch.Tensor,
        adv_grad: torch.Tensor,
    ) -> torch.Tensor:
        """Update the perturbation."""
        raise NotImplementedError()

    def _create_random_target(self, label: torch.Tensor) -> torch.Tensor:
        if self.num_classes is not None:
            label_offset = torch.randint_like(
                label, low=0, high=self.num_classes
            )
            return (label + label_offset) % self.num_classes
        raise RuntimeError("num_classes must be provided.")


class NoOpAttacker(AttackerBase):
    """Not impl."""

    def __init__(self):
        super().__init__(num_iter=0, num_classes=None, original_attack=True)

    def attack_init(
        self, image_clean: torch.Tensor, label: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return super().attack_init(image_clean, label)

    def attack_update(
        self,
        image_clean: torch.Tensor,
        adv: torch.Tensor,
        adv_grad: torch.Tensor,
    ) -> torch.Tensor:
        return super().attack_update(image_clean, adv, adv_grad)


class CopyAttacker(AttackerBase):
    """Always return the clean image itself.

    Mimick the AdvProp's behavior.
    """

    def __init__(self):
        super().__init__(num_iter=0, num_classes=None, original_attack=True)

    def attack_init(
        self, image_clean: torch.Tensor, label: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        return image_clean, label

    def attack_update(self, image_clean, adv, adv_grad):
        _, _ = adv, adv_grad
        return image_clean


class PGDAttacker(AttackerBase):
    """PGD Attacker.

    https://arxiv.org/pdf/1706.06083.pdf
    """

    def __init__(
        self,
        num_iter,
        epsilon,
        step_size,
        start_from_clean=False,
        kernel_size=15,
        prob_start_from_clean=0.0,
        translation=False,
        device="cuda:0",
        num_classes=1000,
        original_attack=False,
    ):
        super().__init__(num_iter, num_classes, original_attack)
        step_size = max(step_size, epsilon / num_iter)
        self.num_iter = num_iter
        self.epsilon = epsilon * IMAGE_SCALE
        self.step_size = step_size * IMAGE_SCALE
        self.start_from_clean = start_from_clean
        self.prob_start_from_clean = prob_start_from_clean
        self.device = device
        self.translation = translation
        if translation:
            # this is equivalent to deepth wise convolution
            # details can be found in the docs of Conv2d.
            # "When groups == in_channels and out_channels == K * in_channels, where K is a positive integer, this operation is also termed in literature as depthwise convolution."
            self.conv = nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=kernel_size,
                stride=(kernel_size - 1) // 2,
                bias=False,
                groups=3,
            ).to(self.device)
            self.gkernel = get_kernel(
                kernel_size, nsig=3, device=self.device
            ).to(self.device)
            self.conv.weight = self.gkernel

    def attack_init(self, image_clean, label):
        if self.original_attack:
            target_label = label
        else:
            target_label = self._create_random_target(label)

        if not self.start_from_clean:
            init_start = torch.empty_like(image_clean).uniform_(
                -self.epsilon, self.epsilon
            )

            # NOTE(meijieru): the prob_start_from_clean use normal distribution,
            # so prob_start_from_clean==1 doesn't mean clean images
            start_from_noise_index = (
                torch.randn([]) > self.prob_start_from_clean
            ).float()
            start_adv = image_clean + start_from_noise_index * init_start
        else:
            start_adv = image_clean.clone().detach()
        return start_adv, target_label

    def attack_update(
        self,
        image_clean: torch.Tensor,
        adv: torch.Tensor,
        adv_grad: torch.Tensor,
    ) -> torch.Tensor:
        lower_bound = torch.clamp(image_clean - self.epsilon, min=-1.0, max=1.0)
        upper_bound = torch.clamp(image_clean + self.epsilon, min=-1.0, max=1.0)

        if self.translation:
            adv_grad = self.conv(adv_grad)
        if self.original_attack:
            adv = adv + torch.sign(adv_grad) * self.step_size
        else:
            adv = adv - torch.sign(adv_grad) * self.step_size
        adv = torch.where(adv > lower_bound, adv, lower_bound)
        adv = torch.where(adv < upper_bound, adv, upper_bound).detach()
        return adv

    def attack(
        self,
        image_clean: torch.Tensor,
        label: torch.Tensor,
        model: typing.Callable,
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Full attack process.

        It use standard cross entropy loss, so we won't use it.
        """
        adv, target_label = self.attack_init(image_clean, label)
        for _ in range(self.num_iter):
            adv.requires_grad_(True)
            logits = model(adv)
            losses = F.cross_entropy(logits, target_label)
            g = torch.autograd.grad(
                losses, adv, retain_graph=False, create_graph=False
            )[0]
            adv = self.attack_update(image_clean, adv, g)
        return adv, target_label
