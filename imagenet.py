"""
Training script for ImageNet
Copyright (c) Wei YANG, 2017
"""
from __future__ import print_function

from typing import Optional, Tuple, Union

import numpy as np
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import argparse
import math
import os
import pprint
import random
import shutil
import sys
import time
from functools import partial

import tensorboardX
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

import net
from attacker import CopyAttacker, NoOpAttacker, PGDAttacker
from augmentations import autoaug
from augmentations.cutmix import cutmix_data
from augmentations.mixup import mixup_criterion, mixup_data
from fast_advprop import (
    get_adv_loss_strategy,
    get_adv_lr_strategy,
    get_adv_params_split,
    get_adv_training_params,
    get_exact_same_budget,
    reinit_adv_params_count,
    use_param_sets,
)
from normalization import (
    MixBatchNorm2d,
    to_adv_status,
    to_clean_status,
)
from utils import Bar, Logger, accuracy
from utils import distributed as ud
from utils import savefig
from utils.distributed import AverageMeter
from utils.fastaug.fastaug import FastAugmentation
from utils.training import (
    WarmUpLR,
    adjust_learning_rate,
    label_smoothing_cross_entropy,
)

# Models
default_model_names = sorted(
    name
    for name in net.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(net.__dict__[name])
    and not name.startswith("to_")
    and not name.startswith("partial")
)

model_names = default_model_names

# Parse arguments
parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

# Datasets
parser.add_argument("-d", "--data", default="path to dataset", type=str)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 4)",
)
# Optimization options
parser.add_argument(
    "--epochs",
    default=90,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--train-batch",
    default=256,
    type=int,
    metavar="N",
    help="total train batchsize (default: 256)",
)
parser.add_argument(
    "--test-batch",
    default=200,
    type=int,
    metavar="N",
    help="total test batchsize (default: 200)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.1,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
parser.add_argument(
    "--schedule",
    type=int,
    nargs="+",
    default=[150, 225],
    help="Decrease learning rate at these epochs.",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.1,
    help="LR is multiplied by gamma on schedule.",
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum"
)
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=1e-4,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-4)",
)
# Checkpoints
parser.add_argument(
    "-c",
    "--checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="path to save checkpoint.",
)

# The learning rate of the setting 'step' cannot be handled automatically,
# so you should change --lr as you wanted,
# but you don't need to change other settings.
# more information can be referred in the function adjust_learning_rate
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument("--load", default="", type=str)
# Architecture
parser.add_argument(
    "--arch",
    "-a",
    metavar="ARCH",
    default="resnet18",
    choices=model_names,
    help="model architecture: "
    + " | ".join(model_names)
    + " (default: resnet18)",
)
# Miscs
parser.add_argument("--manualSeed", type=int, help="manual seed")
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--evaluate_imagenet_c", action="store_true", help="for evaluate Imagenet-C"
)
# Device options
parser.add_argument(
    "--gpu-id", default="0", type=str, help="id(s) for CUDA_VISIBLE_DEVICES"
)

parser.add_argument("--warm", default=5, type=int, help="warm up epochs")
parser.add_argument(
    "--warm_lr", default=0.1, type=float, help="warm up start lr"
)
parser.add_argument(
    "--num_classes", default=1000, type=int, help="number of classes"
)
parser.add_argument(
    "--norm_layer",
    type=str,
    default="bn",
    choices=["bn", "mixbn", "gn", "mixgn"],
)
parser.add_argument(
    "--lr_schedule", type=str, default="step", choices=["step", "cos"]
)
parser.add_argument("--fastaug", action="store_true")
parser.add_argument("--autoaug", type=str, default=None)
parser.add_argument("--already224", action="store_true")
parser.add_argument("--nesterov", action="store_true")
parser.add_argument("--smoothing", type=float, default=0)

# attacker options
parser.add_argument(
    "--attack-iter", help="Adversarial attack iteration", type=int, default=0
)
parser.add_argument(
    "--attack-epsilon",
    help="Adversarial attack maximal perturbation",
    type=float,
    default=1.0,
)
parser.add_argument(
    "--attack-step-size",
    help="Adversarial attack step size",
    type=float,
    default=1.0,
)
parser.add_argument(
    "--prob_start_from_clean",
    help="PGD initial type, only for training"
    ", NOTE: 0.0 doesn't complete disabled, use `start_from_clean` instead",
    type=float,
    default=0.2,
)
parser.add_argument(
    "--start_from_clean",
    action="store_true",
    help="PGD initial type, only for training",
)

parser.add_argument(
    "--attacker_type",
    help="to determine which intermediate attacker to use",
    type=str,
    default="none",
    choices=["pgd", "none", "copy"],
)

parser.add_argument(
    "--multi_clean_strategy",
    help="how to split the data for clean/adv examples",
    type=str,
    default="none",
)
parser.add_argument(
    "--reuse_mid_grad", action="store_true", help="reuse grad in PGD attack"
)
parser.add_argument(
    "--original_attack",
    action="store_true",
    help="do not use target attack in pgd, must be set for grad reuse",
)
parser.add_argument(
    "--lr_strategy",
    type=str,
    default="shared:1,clean:1,adv:1",
    help="synchronize the parameter updating speed",
)
parser.add_argument(
    "--loss_strategy",
    type=str,
    default="clean:1,attack:0.5,adv:0.5",
    help="Re-balancing training samples",
)
parser.add_argument(
    "--attack_in_train", action="store_true", help="attack in train mode"
)
parser.add_argument(
    "--shuffle_before_train_adv",
    action="store_true",
    help="to avoid possible information leakage",
)
parser.add_argument(
    "--exact_same_training_budget",
    action="store_true",
    help="use the same training budget",
)
parser.add_argument(
    "--mixup", default=0.0, type=float, help="mixup hyper-parameter"
)
parser.add_argument(
    "--cutmix", default=0.0, type=float, help="cutmix hyper-parameter"
)
parser.add_argument(
    "--other_aug_on_attack",
    action="store_true",
    help="whether to apply data aug on noisy imgs",
)
parser.add_argument(
    "--other_aug_on_adv",
    action="store_true",
    help="whether to apply data aug on adv imgs",
)

# distributed training parameters
parser.add_argument("--distributed", action="store_true", help="use ddp")
parser.add_argument(
    "--world-size", default=1, type=int, help="number of distributed processes"
)
parser.add_argument(
    "--dist-url",
    default="env://",
    help="url used to set up distributed training",
)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


def main():
    global best_acc, state
    start_epoch = (
        args.start_epoch
    )  # start from epoch 0 or last checkpoint epoch

    ud.init_distributed_mode(args)

    if args.attacker_type == "none":
        attacker = NoOpAttacker()
    elif args.attacker_type == "copy":
        attacker = CopyAttacker()
    elif args.attacker_type == "pgd":
        attacker = PGDAttacker(
            args.attack_iter,
            args.attack_epsilon,
            args.attack_step_size,
            start_from_clean=args.start_from_clean,
            prob_start_from_clean=args.prob_start_from_clean
            if not args.evaluate
            else 0.0,
            num_classes=args.num_classes,
            original_attack=args.original_attack,
        )
    else:
        raise ValueError("Unknown attacker: {}".format(args.attacker_type))

    if args.checkpoint and ud.is_main_process():
        os.makedirs(args.checkpoint, exist_ok=(args.load != ""))
    if (
        not args.evaluate
        and not args.evaluate_imagenet_c
        and ud.is_main_process()
    ):
        with open(os.path.join(args.checkpoint, "args.txt"), "w") as f:
            pprint.pprint(args, f)

    # Data loading code
    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transform_train = transforms.Compose(
        [
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
    )
    if args.fastaug:
        assert args.autoaug is None
        transform_train.transforms.insert(0, FastAugmentation())
    elif args.autoaug is not None:
        transform_train.transforms.extend(
            [
                autoaug.AutoAugment(
                    policy=autoaug.AutoAugmentPolicy(args.autoaug)
                )
            ]
        )
    transform_train.transforms.extend(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    args._adv_training_params = get_adv_training_params(args)
    get_exact_same_budget(args)

    if not args.evaluate and not args.evaluate_imagenet_c:
        train_dataset = datasets.ImageFolder(traindir, transform_train)
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset
            )
        else:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            (train_dataset),
            batch_size=int(
                args.train_batch
                * args._adv_training_params["train_batch_scale"]
            ),
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True,
            # NOTE(meijieru): add drop_last here to make sure batch could be divided
            # correctly with our split strategy
            drop_last=True,
        )

    val_transforms = [
        transforms.ToTensor(),
        normalize,
    ]
    if not args.already224:
        val_transforms = [
            transforms.Scale(256),
            transforms.CenterCrop(224),
        ] + val_transforms
    if not args.evaluate_imagenet_c:
        test_dataset = datasets.ImageFolder(
            valdir, transforms.Compose(val_transforms)
        )
        if args.distributed:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset
            )
        else:
            test_sampler = torch.utils.data.SequentialSampler(test_dataset)
        val_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.test_batch,
            sampler=test_sampler,
            num_workers=args.workers,
            pin_memory=True,
        )

    # create model
    print("=> creating model '{}'".format(args.arch))
    norm_layer = {
        "bn": nn.BatchNorm2d,
        "mixbn": MixBatchNorm2d,
    }[args.norm_layer]
    model = net.__dict__[args.arch](
        num_classes=args.num_classes, norm_layer=norm_layer
    )
    args._params_split = get_adv_params_split(model)
    args._params_count = reinit_adv_params_count(model)

    if args.distributed:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
    else:
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True
    print(
        "    Total params: %.2fM"
        % (sum(p.numel() for p in model.parameters()) / 1000000.0)
    )

    # define loss function (criterion) and optimizer
    if args.smoothing == 0:
        criterion = nn.CrossEntropyLoss(reduction="none").cuda()
    else:
        criterion = partial(
            label_smoothing_cross_entropy, classes=args.num_classes, dim=-1
        )
    optimizer = optim.SGD(
        get_adv_lr_strategy(args),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )

    args._loss_strategy = get_adv_loss_strategy(args)

    # Resume
    title = "ImageNet-" + args.arch
    if args.resume:
        # Load checkpoint.
        print("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume
        ), "Error: no checkpoint directory found!"
        if not args.checkpoint:
            args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume, map_location="cpu")
        best_acc = checkpoint["best_acc"]
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if ud.is_main_process():
            logger = Logger(
                os.path.join(args.checkpoint, "log.txt"),
                title=title,
                resume=args.checkpoint == os.path.dirname(args.resume),
            )
        else:
            logger = None
    else:
        if args.load:
            checkpoint = torch.load(args.load, map_location="cpu")
            if not args.checkpoint:
                args.checkpoint = os.path.dirname(args.load)
            model.load_state_dict(checkpoint["state_dict"])
        if ud.is_main_process():
            logger = Logger(
                os.path.join(args.checkpoint, "log.txt"), title=title
            )
        else:
            logger = None
    if ud.is_main_process():
        logger.set_names(
            [
                "Learning Rate",
                "Train Loss",
                "Valid Loss",
                "Train Acc.",
                "Valid Acc.",
            ]
        )

    if args.evaluate:
        print("\nEvaluation only")
        test_loss, test_acc = test(
            val_loader, model, criterion, start_epoch, use_cuda
        )
        print(" Test Loss:  %.8f, Test Acc:  %.2f" % (test_loss, test_acc))
        return

    if args.evaluate_imagenet_c:
        print("Evaluate ImageNet C")
        distortions = [
            "gaussian_noise",
            "shot_noise",
            "impulse_noise",
            "defocus_blur",
            "glass_blur",
            "motion_blur",
            "zoom_blur",
            "snow",
            "frost",
            "fog",
            "brightness",
            "contrast",
            "elastic_transform",
            "pixelate",
            "jpeg_compression",
            "speckle_noise",
            "gaussian_blur",
            "spatter",
            "saturate",
        ]

        error_rates = []
        assert args.load
        checkpoint_dir = os.path.dirname(args.load)
        with open(
            os.path.join(checkpoint_dir, "eval_imagenet_c.txt"), "w"
        ) as f:
            for distortion_name in tqdm(distortions):
                rate = show_performance(
                    distortion_name,
                    model,
                    criterion,
                    start_epoch,
                    use_cuda,
                    file=f,
                )
                error_rates.append(rate)
                print(
                    "Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}".format(
                        distortion_name, 100 * rate
                    ),
                    file=f,
                    flush=True,
                )
            print(distortions, file=f)
            print(error_rates, file=f)
            print(np.mean(error_rates), file=f, flush=True)
        return

    # Train and val
    writer = None
    if ud.is_main_process():
        writer = tensorboardX.SummaryWriter(log_dir=args.checkpoint)
    warmup_scheduler = (
        WarmUpLR(
            optimizer, len(train_loader) * args.warm, start_lr=args.warm_lr
        )
        if args.warm > 0
        else None
    )
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if epoch >= args.warm and args.lr_schedule == "step":
            adjust_learning_rate(optimizer, epoch, args, state)

        print(
            "\nEpoch: [%d | %d] LR: %f"
            % (epoch, args.epochs, optimizer.param_groups[-1]["lr"])
        )

        if args.mixup:
            assert not args.cutmix
            extra_data_aug_fun = partial(
                mixup_data, alpha=args.mixup, half=False
            )
        elif args.cutmix:
            extra_data_aug_fun = partial(
                cutmix_data, beta=args.cutmix, half=False
            )
        else:
            extra_data_aug_fun = None
        train_func = partial(
            train,
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            epoch=epoch,
            use_cuda=use_cuda,
            warmup_scheduler=warmup_scheduler,
            extra_data_aug_fun=extra_data_aug_fun,
            mixbn=args.norm_layer in ["mixbn", "mixgn"],
            attacker=attacker,
        )
        if args.norm_layer in ["mixbn", "mixgn"]:
            (
                train_loss,
                train_acc,
                loss_attack,
                loss_adv,
                top1_attack,
                top1_adv,
            ) = train_func()
        else:
            train_loss, train_acc = train_func()
        if ud.is_main_process():
            writer.add_scalar("Train/loss", train_loss, epoch)
            writer.add_scalar("Train/acc", train_acc, epoch)
            writer.add_scalar(
                "Train/lr", optimizer.param_groups[-1]["lr"], epoch
            )

        if args.norm_layer in ["mixbn", "mixgn"]:
            if ud.is_main_process():
                writer.add_scalar("Train/loss_attack", loss_attack, epoch)
                writer.add_scalar("Train/loss_adv", loss_adv, epoch)
                writer.add_scalar("Train/acc_attack", top1_attack, epoch)
                writer.add_scalar("Train/acc_adv", top1_adv, epoch)
            model.apply(to_clean_status)
        test_loss, test_acc = test(
            val_loader, model, criterion, epoch, use_cuda
        )
        if ud.is_main_process():
            writer.add_scalar("Test/loss", test_loss, epoch)
            writer.add_scalar("Test/acc", test_acc, epoch)

            # append logger file
            logger.append(
                [state["lr"], train_loss, test_loss, train_acc, test_acc]
            )

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        state_ckpt = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "acc": test_acc,
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
        }
        if epoch + 1 in args.schedule:
            save_checkpoint(
                state_ckpt,
                is_best,
                checkpoint=args.checkpoint,
                filename="epoch_{}.pth.tar".format(epoch),
            )
        save_checkpoint(
            state_ckpt,
            is_best,
            checkpoint=args.checkpoint,
            filename="checkpoint.pth.tar",
        )

    print("Best acc:")
    print(best_acc)
    if ud.is_main_process():
        writer.close()
        logger.close()
        logger.plot()
        savefig(os.path.join(args.checkpoint, "log.eps"))


Tensor_or_TripleTensor = Union[
    torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
]


def forward_backward_one_step(
    inputs: torch.Tensor,
    targets: Tensor_or_TripleTensor,
    model: nn.Module,
    criterion: nn.Module,
    losses: Optional[AverageMeter] = None,
    top1: Optional[AverageMeter] = None,
    top5: Optional[AverageMeter] = None,
    loss_scalar: float = 1.0,
) -> None:
    loss_i, _, _ = forward_one_step(
        inputs, targets, model, criterion, losses, top1, top5
    )
    (loss_i * loss_scalar).sum().backward()


def forward_one_step(
    inputs: torch.Tensor,
    targets: Tensor_or_TripleTensor,
    model: nn.Module,
    criterion: nn.Module,
    losses: Optional[AverageMeter] = None,
    top1: Optional[AverageMeter] = None,
    top5: Optional[AverageMeter] = None,
):
    outputs = model(inputs)
    # NOTE(meijieru): we should not average loss here.
    if isinstance(targets, torch.Tensor):
        loss = criterion(outputs, targets)
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
    elif isinstance(targets, tuple) and len(targets) == 3:
        loss = mixup_criterion(
            criterion, outputs, targets[0], targets[1], targets[2]
        )
        prec1, prec5 = accuracy(outputs.data, targets[0].data, topk=(1, 5))
    else:
        raise RuntimeError()

    if losses:
        losses.update(loss.mean().item(), outputs.size(0))
    if top1:
        top1.update(prec1.item(), outputs.size(0))
    if top5:
        top5.update(prec5.item(), outputs.size(0))
    return loss, prec1, prec5


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    use_cuda,
    warmup_scheduler,
    extra_data_aug_fun=None,
    mixbn=False,
    attacker=NoOpAttacker(),
):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    if mixbn:
        losses_attack = AverageMeter()
        losses_adv = AverageMeter()
        top1_attack = AverageMeter()
        top1_adv = AverageMeter()
    else:
        losses_attack, losses_adv, top1_attack, top1_adv = [
            None for _ in range(4)
        ]
    end = time.time()

    bar = Bar("Processing", max=len(train_loader))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if epoch < args.warm:
            warmup_scheduler.step()
        elif args.lr_schedule == "cos":
            adjust_learning_rate(
                optimizer,
                epoch,
                args,
                state,
                batch=batch_idx,
                nBatch=len(train_loader),
            )

        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)

        # NOTE(meijieru): if use args.multi_clean, split the input here
        _inputs, _targets = args._adv_training_params["input_split_fun"](
            inputs, targets
        )

        args._params_count = reinit_adv_params_count(model)
        optimizer.zero_grad()

        # clean forward/backward
        model.train()
        model.apply(to_clean_status)
        use_param_sets(
            args,
            ["shared", "clean"],
            count_inc=_inputs[0].size(0) * args._loss_strategy["clean"],
        )
        inputs_clean, targets_clean = _inputs[0], _targets[0]
        if extra_data_aug_fun is not None:
            inputs_clean, targets_clean = extra_data_aug_fun(
                inputs_clean, targets_clean
            )
        forward_backward_one_step(
            inputs_clean,
            targets_clean,
            model,
            criterion,
            losses,
            top1,
            top5,
            loss_scalar=args._loss_strategy["clean"],
        )

        if _inputs[-1] is not None:
            # generate adv images

            if args.attack_in_train:
                model.train()
            else:
                model.eval()
            model.apply(to_adv_status)
            if not args.reuse_mid_grad:
                params_set = []
            else:
                if attacker.num_iter > 1:  # actual grad compute
                    assert attacker.is_grad_reusable()
                params_set = ["shared", "adv"]

            inputs_attack, targets_attack = attacker.attack_init(
                _inputs[-1], _targets[-1]
            )
            for _ in range(attacker.num_iter):
                inputs_attack.requires_grad_(True)
                use_param_sets(
                    args,
                    params_set,
                    count_inc=_inputs[-1].size(0)
                    * args._loss_strategy["attack"],
                )
                if args.other_aug_on_attack and extra_data_aug_fun is not None:
                    _inputs_attack, _targets_attack = extra_data_aug_fun(
                        inputs_attack, targets_attack
                    )
                else:
                    _inputs_attack, _targets_attack = (
                        inputs_attack,
                        targets_attack,
                    )
                forward_backward_one_step(
                    _inputs_attack,
                    _targets_attack,
                    model,
                    criterion,
                    losses_attack,
                    top1_attack,
                    loss_scalar=args._loss_strategy["attack"],
                )
                inputs_attack = attacker.attack_update(
                    _inputs[-1], inputs_attack, inputs_attack.grad
                )

            if mixbn:
                model.train()
                # if isinstance(attacker, (NoOpAttacker, CopyAttacker)):
                if isinstance(attacker, (NoOpAttacker)):
                    raise NotImplementedError(
                        "Clean image should not reach here!!!"
                    )
                model.apply(to_adv_status)
                use_param_sets(
                    args,
                    ["shared", "adv"],
                    count_inc=inputs_attack.size(0)
                    * args._loss_strategy["adv"],
                )
                if args.shuffle_before_train_adv:
                    if not args.distributed:
                        shuffle_index = torch.randperm(inputs_attack.size(0))
                        inputs_adv = inputs_attack[shuffle_index].detach()
                        targets_adv = _targets[-1][shuffle_index]
                    else:
                        inputs_adv_all = torch.cat(
                            ud.all_gather(inputs_attack.clone().detach()), dim=0
                        )
                        targets_adv_all = torch.cat(
                            ud.all_gather(_targets[-1]), dim=0
                        )
                        shuffle_index = torch.randperm(inputs_adv_all.size(0))
                        shuffle_index = shuffle_index[
                            ud.get_rank() : inputs_adv_all.size(
                                0
                            ) : ud.get_world_size()
                        ]
                        assert shuffle_index.numel() == inputs_attack.size(0)
                        inputs_adv = inputs_adv_all[shuffle_index]
                        targets_adv = targets_adv_all[shuffle_index]
                else:
                    inputs_adv = inputs_attack.clone().detach()
                    targets_adv = _targets[-1]
                if args.other_aug_on_adv and extra_data_aug_fun is not None:
                    inputs_adv, targets_adv = extra_data_aug_fun(
                        inputs_adv, targets_adv
                    )
                forward_backward_one_step(
                    inputs_adv,
                    targets_adv,
                    model,
                    criterion,
                    losses_adv,
                    top1_adv,
                    loss_scalar=args._loss_strategy["adv"],
                )
            else:
                # NOTE(meijieru): Sanity check with CopyAttacker & multi_clean_strategy=='1:1'
                raise NotImplementedError()

        # # NOTE(meijieru): normalize the gradient
        # # Here we divide use a constant, so the learning rate is correct for
        # # each set automatically
        # # Consider the extreme situation, where the noise is always zero, we
        # # have 2x batch. According to the linear scale rule, lr should be 2x,
        # # which is the case here.
        # for param in model.parameters():
        #     param.grad.div_(args.train_batch)

        # NOTE(meijieru): use args.lr_strategy instead, here we just average
        # across example
        for param, count in args._params_count.items():
            if param.grad is not None:
                param.grad.div_(count)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if not mixbn:
            loss_str = "{:.4f}".format(losses.avg)
            top1_str = "{:.4f}".format(top1.avg)
        else:
            loss_str = "{:.2f}/{:.2f}/{:.2f}".format(
                losses.avg, losses_attack.avg, losses_adv.avg
            )
            top1_str = "{:.2f}/{:.2f}/{:.2f}".format(
                top1.avg, top1_attack.avg, top1_adv.avg
            )
        bar.suffix = "({batch}/{size}) Data: {data:.2f}s | Batch: {bt:.2f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:s} | top1: {top1:s} | top5: {top5: .1f}".format(
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=loss_str,
            top1=top1_str,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    if mixbn:
        for meter in [
            losses,
            losses_adv,
            losses_attack,
            top1,
            top1_attack,
            top1_adv,
        ]:
            meter.synchronize_between_processes()
        return (
            losses.avg,
            top1.avg,
            losses_attack.avg,
            losses_adv.avg,
            top1_attack.avg,
            top1_adv.avg,
        )
    else:
        for meter in [losses, top1]:
            meter.synchronize_between_processes()
        return (losses.avg, top1.avg)


def test(val_loader, model, criterion, epoch, use_cuda):
    del epoch
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model.apply(to_clean_status)

    end = time.time()
    bar = Bar("Processing", max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(
            inputs, volatile=True
        ), torch.autograd.Variable(targets)

        # compute output
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, targets).mean()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}".format(
            batch=batch_idx + 1,
            size=len(val_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    for meter in [losses, top1, top5]:
        meter.synchronize_between_processes()
    return (losses.avg, top1.avg)


def show_performance(
    distortion_name,
    model,
    criterion,
    start_epoch,
    use_cuda,
    severities=list(range(1, 6)),
    file=sys.stdout,
):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    errs = []

    for severity in severities:
        if args.distributed:
            raise NotImplementedError("TODO(meijieru)")
        valdir = os.path.join(args.data, distortion_name, str(severity))
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                valdir,
                transforms.Compose(
                    [
                        # transforms.Scale(256),
                        # transforms.CenterCrop(224), # already 224 x 224
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            ),
            batch_size=args.test_batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

        _, test_acc = test(val_loader, model, criterion, start_epoch, use_cuda)

        errs.append(1.0 - test_acc / 100.0)

    print("\n=Average", tuple(errs), file=file, flush=True)
    return np.mean(errs)


def save_checkpoint(
    state, is_best, checkpoint="checkpoint", filename="checkpoint.pth.tar"
):
    filepath = os.path.join(checkpoint, filename)
    ud.save_on_master(state, filepath)
    if is_best and ud.is_main_process():
        shutil.copyfile(
            filepath, os.path.join(checkpoint, "model_best.pth.tar")
        )


if __name__ == "__main__":
    main()
