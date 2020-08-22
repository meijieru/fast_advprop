"""Utilities for Fast AdvProp."""

from typing import Any, MutableMapping, Optional, Sequence, Union

import argparse
import itertools

import torch
import torch.nn as nn

from normalization import MixBatchNorm2d


def get_param2name(model: nn.Module) -> MutableMapping[torch.Tensor, str]:
    return {param: name for name, param in model.named_parameters()}


def get_module2name(model: nn.Module) -> MutableMapping[nn.Module, str]:
    return {m: name for name, m in model.named_modules()}


def get_adv_lr_strategy(
    args: argparse.Namespace,
) -> Sequence[MutableMapping[str, Any]]:
    """Parse learning rate strategy.

    See `Synchronizing parameter updating speed.` section in the paper.

    NOTE:
        - `get_adv_params_split` should be called before this function.
        - `shared:1,clean:1,adv:1` sets equal weights for all parameters.
    """
    res = []
    for per_split in args.lr_strategy.split(","):
        key, multi = per_split.split(":")
        params = args._params_split[key]
        if len(params):
            res.append({"params": params, "lr": args.lr * float(multi)})
    return res


def get_adv_loss_strategy(
    args: argparse.Namespace, res: Optional[dict] = None
) -> MutableMapping[str, float]:
    """Parse `loss_strategy` to get the weights for each param split.

    See `Re-balancing training samples.` section in the paper.
    """
    res = res or {}
    for per_split in args.loss_strategy.split(","):
        key, multi = per_split.split(":")
        res[key] = float(multi)
    return res


def get_exact_same_budget(args: argparse.Namespace) -> None:
    """Ensure same training budget with vanilla training.

    We adjust the `schedule` & `epochs` accordingly.
    It modifies the `args` in place.
    """
    if args.exact_same_training_budget:
        if args.multi_clean_strategy == "none":
            budget_scale = 1.0
        elif args.multi_clean_strategy == "copy":
            budget_scale = args.attack_iter + 1 + 1
        elif ":" in args.multi_clean_strategy:
            # TODO(meijieru): update formula with attack_iter != 1
            assert args.attack_iter == 1
            ratios = args.multi_clean_strategy.split(":")
            num_normal, num_adv = [int(val) for val in ratios]
            adv_ratio = num_adv / (num_normal + num_adv)
            normal_ratio = num_normal / (num_normal + num_adv)
            budget_scale = normal_ratio + adv_ratio * 2
        else:
            raise ValueError(
                "Unknown multi_clean_strategy: {}".format(
                    args.multi_clean_strategy
                )
            )

        new_schedule = [int(round(ep / budget_scale)) for ep in args.schedule]
        new_epochs = int(round(args.epochs / budget_scale))
        print("Budget scaling factor: {}".format(budget_scale))
        print("Schedule: {} => {}".format(args.schedule, new_schedule))
        print("Epochs: {} => {}".format(args.epochs, new_epochs))
        args.epochs = new_epochs
        args.schedule = new_schedule


def get_adv_training_params(
    args: argparse.Namespace, res: Optional[dict] = None
) -> MutableMapping[str, Any]:
    """Setup Fast AdvProp training data config."""
    res = res or {}
    if args.multi_clean_strategy == "none":
        res["input_split_fun"] = lambda inputs, targets: (
            [inputs, None],
            [targets, None],
        )
        res["train_batch_scale"] = 1
    elif args.multi_clean_strategy == "copy":
        res["input_split_fun"] = lambda inputs, targets: (
            [inputs, inputs],
            [targets, targets],
        )
        res["train_batch_scale"] = 1
    elif ":" in args.multi_clean_strategy:
        ratios = args.multi_clean_strategy.split(":")
        ratios = [int(val) for val in ratios]

        def _input_split_fun(inputs, targets):
            splits = [args.train_batch]
            splits.append(inputs.size(0) - splits[0])
            _inputs = torch.split(inputs, splits)
            _targets = torch.split(targets, splits)
            return _inputs, _targets

        res["input_split_fun"] = _input_split_fun
        res["train_batch_scale"] = sum(ratios) / ratios[0]
    else:
        raise ValueError()

    return res


def get_adv_params_split(
    model: nn.Module,
    res: Optional[dict] = None,
    verbose: bool = False,
) -> MutableMapping[str, Sequence[torch.Tensor]]:
    """Organize model parameters into corresponding groups.

    Parameters are divided into 3 categories:
        adv: parameters within auxiliary BNs.
        clean: parameters within original BNs.
        shared: shared parameters by two branches.
    """
    module2name = get_module2name(model)

    clean_specific_params = []
    adv_specific_params = []
    shared_params = []

    def aux(m: nn.Module):
        if isinstance(m, (MixBatchNorm2d,)):
            adv_specific_params.extend([m.aux_bn.weight, m.aux_bn.bias])
            clean_specific_params.extend([m.weight, m.bias])
        elif isinstance(m, (nn.BatchNorm2d,)):
            if "aux" not in module2name[m]:
                shared_params.extend(m.parameters())
        elif isinstance(m, (nn.Conv2d, nn.Linear)):
            assert not isinstance(m, MixBatchNorm2d)
            shared_params.extend(m.parameters())
        else:
            # either no param or composed with other modules
            pass

    model.apply(aux)

    _all_params = [clean_specific_params, adv_specific_params, shared_params]
    if set(model.parameters()) != set(itertools.chain(*_all_params)):
        param2name = get_param2name(model)
        remain_list = [
            param2name[param]
            for param in set(model.parameters())
            - set(itertools.chain(*_all_params))
        ]
        raise RuntimeError(
            "The following params are not classified: {}".format(remain_list)
        )
    if len(list(model.parameters())) != sum(len(v) for v in _all_params):
        raise RuntimeError("Param split has overlap")

    res = res or {}
    res["clean"] = clean_specific_params
    res["adv"] = adv_specific_params
    res["shared"] = shared_params

    if verbose:
        param2name = get_param2name(model)
        names = {
            name: sorted([param2name[param] for param in plist])
            for name, plist in res.items()
        }
        print("Params split: {}".format(names))
    return res


def reinit_adv_params_count(
    model: nn.Module,
    res: Optional[dict] = None,
) -> MutableMapping[torch.Tensor, int]:
    """Reset counts for parameters.

    We track influences from all related samples for each parameters.
    This function need to be called before counting.
    """
    res = res or {}
    for param in model.parameters():
        res[param] = 0
    return res


def use_param_sets(
    args: argparse.Namespace,
    used_keys: Sequence[str],
    count_inc: Union[int, float] = 1,
    update_count: bool = True,
):
    """Update status for param.

    We track influences from all related samples for each parameters.
    Should be called every grad calculation.
    """
    _used_keys = set(used_keys)
    all_keys = set(args._params_split.keys())
    for key in all_keys:
        if key in _used_keys:
            requires_grad = True
        else:
            requires_grad = False
        for param in args._params_split[key]:
            param.requires_grad_(requires_grad)
            if update_count and requires_grad:
                args._params_count[param] += count_inc
