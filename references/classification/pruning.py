from pathlib import Path
import os

import torch
from torch import nn
from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
import torchvision
import wandb

from asdfghjkl import SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_DIAG

import transforms
from train import train_one_epoch, evaluate, load_data
import utils

from models import Toy2
from obs import Scope
from obs import FullOBS, LayerOBS, KronOBS, NoneOBS, FullWoodOBS, BlockWoodOBS


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Pruning")

    parser.add_argument("--dataset",
                        default="IMAGENET",
                        type=str,
                        help="dataset name")
    parser.add_argument("--data-path",
                        default="/datasets01/imagenet_full_size/061417/",
                        type=str,
                        help="dataset path")
    parser.add_argument("--model",
                        default="resnet50",
                        type=str,
                        help="model name")
    parser.add_argument("--device",
                        default="cuda",
                        type=str,
                        help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b",
        "--batch-size",
        default=128,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument(
        "--val-batch-size",
        default=512,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument(
        "--fisher-batch-size",
        default=32,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size")

    parser.add_argument("--epochs",
                        default=90,
                        type=int,
                        metavar="N",
                        help="number of total epochs to run")
    parser.add_argument("-j",
                        "--workers",
                        default=4,
                        type=int,
                        metavar="N",
                        help="number of data loading workers (default: 16)")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr",
                        default=5e-3,
                        type=float,
                        help="initial learning rate")
    parser.add_argument("--momentum",
                        default=0.9,
                        type=float,
                        metavar="M",
                        help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help=
        "weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument("--label-smoothing",
                        default=0.0,
                        type=float,
                        help="label smoothing (default: 0.0)",
                        dest="label_smoothing")
    parser.add_argument("--mixup-alpha",
                        default=0.0,
                        type=float,
                        help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha",
                        default=0.0,
                        type=float,
                        help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler",
                        default="steplr",
                        type=str,
                        help="the lr scheduler (default: steplr)")
    #parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    #parser.add_argument("--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)")
    #parser.add_argument("--lr-warmup-decay", default=1., type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size",
                        default=6,
                        type=int,
                        help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma",
                        default=0.6,
                        type=float,
                        help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--print-freq",
                        default=300,
                        type=int,
                        help="print frequency")
    parser.add_argument("--output-dir",
                        default="./output",
                        type=str,
                        help="path to save outputs")
    parser.add_argument("--resume",
                        default="",
                        type=str,
                        help="path of checkpoint")
    parser.add_argument("--start-epoch",
                        default=0,
                        type=int,
                        metavar="N",
                        help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help=
        "Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    #parser.add_argument(
    #    "--test-only",
    #    dest="test_only",
    #    help="Only test the model",
    #    action="store_true",
    #)
    #parser.add_argument(
    #    "--pretrained",
    #    dest="pretrained",
    #    help="Use pre-trained models from the modelzoo",
    #    action="store_true",
    #)
    parser.add_argument("--auto-augment",
                        default=None,
                        type=str,
                        help="auto augment policy (default: None)")
    parser.add_argument("--random-erase",
                        default=0.0,
                        type=float,
                        help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    #parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size",
                        default=1,
                        type=int,
                        help="number of distributed processes")
    parser.add_argument("--dist-url",
                        default="env://",
                        type=str,
                        help="url used to set up distributed training")
    #parser.add_argument(
    #    "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    #)
    #parser.add_argument(
    #    "--model-ema-steps",
    #    type=int,
    #    default=32,
    #    help="the number of iterations that controls how often to update the EMA model (default: 32)",
    #)
    #parser.add_argument(
    #    "--model-ema-decay",
    #    type=float,
    #    default=0.99998,
    #    help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    #)
    parser.add_argument(
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only.")
    parser.add_argument("--interpolation",
                        default="bilinear",
                        type=str,
                        help="the interpolation method (default: bilinear)")
    parser.add_argument(
        "--val-resize-size",
        default=256,
        type=int,
        help="the resize size used for validation (default: 256)")
    parser.add_argument(
        "--val-crop-size",
        default=224,
        type=int,
        help="the central crop size used for validation (default: 224)")
    parser.add_argument(
        "--train-crop-size",
        default=224,
        type=int,
        help="the random crop size used for training (default: 224)")
    parser.add_argument("--clip-grad-norm",
                        default=None,
                        type=float,
                        help="the maximum gradient norm (default None)")
    parser.add_argument(
        "--ra-sampler",
        action="store_true",
        help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps",
        default=3,
        type=int,
        help="number of repetitions for Repeated Augmentation (default: 3)")

    # Prototype models only
    parser.add_argument("--weights",
                        default=None,
                        type=str,
                        help="the weights enum name to load")

    # Pruning
    parser.add_argument("--pruning_strategy",
                        type=str,
                        default="oneshot",
                        choices=["oneshot", "gradual"])
    parser.add_argument("--pruning_epochs", default=40, type=int, metavar="N")
    parser.add_argument("--pruning_intvl", default=5, type=int, metavar="N")
    parser.add_argument("--fisher_type",
                        type=str,
                        default="fisher_emp",
                        choices=["fisher_exact", "fisher_mc", "fisher_emp"])
    parser.add_argument("--fisher_shape",
                        type=str,
                        default="full",
                        choices=[
                            "full", "layer_wise", "kron", "unit_wise",
                            "full_wood", "block_wood", "none"
                        ])
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--block_batch", type=int, default=10000)
    parser.add_argument(
        "--layer_normalize",
        dest="layer_normalize",
        action="store_true",
    )
    parser.add_argument(
        "--kfac_fast_inv",
        dest="kfac_fast_inv",
        action="store_true",
    )

    parser.add_argument("--check", dest="check", action="store_true")

    parser.add_argument("--sparsity", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--n_recompute", type=int, default=16)
    parser.add_argument("--n_recompute_samples", type=int, default=64)

    return parser.parse_args()


def list_model(module, prefix="", condition=lambda _: True):
    modules = {}
    has_children = False
    for name, x in module.named_children():
        has_children = True
        new_prefix = prefix + ("" if prefix == "" else ".") + name
        modules.update(list_model(x, new_prefix, condition))
    if not has_children and condition(module):
        modules[prefix] = module
    return modules


def get_global_prnning_scope(model):
    modules = list_model(
        model,
        condition=lambda x:
        (not isinstance(x, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.
                            LayerNorm))) and hasattr(x, "weight"))
    return [Scope(k, v) for k, v in modules.items()]


def create_obs(model, scopes, args):
    obs = {
        SHAPE_FULL: FullOBS,
        SHAPE_LAYER_WISE: LayerOBS,
        SHAPE_KRON: KronOBS,
        "full_wood": FullWoodOBS,
        "block_wood": BlockWoodOBS,
        "none": NoneOBS
    }[args.fisher_shape](model, scopes, args.fisher_type, args.rank,
                         args.world_size)
    if args.fisher_shape == "block_wood":
        obs.set_block_size(args.block_size)
        obs.set_block_batch(args.block_batch)
    if args.fisher_shape in [SHAPE_KRON, SHAPE_LAYER_WISE]:
        obs.normalize = args.layer_normalize
    if args.fisher_shape == SHAPE_KRON:
        obs.fast_inv = args.kfac_fast_inv
    return obs


def wlog(args, kargs):
    if args.rank == 0:
        wandb.log(kargs)


def log(args, *v1, **v2):
    if args.rank == 0:
        print(*v1, **v2)


def polynomial_schedule(start, end, i, n):
    scale = end - start
    progress = min(float(i) / n, 1.0)
    remaining_progress = (1.0 - progress)**2
    return end - scale * remaining_progress


def one_shot_pruning(obs, model, data_loaders, criterion, args):
    acc = 0

    def _cb():
        nonlocal acc
        acc = evaluate(model,
                       criterion,
                       data_loaders["test"],
                       device=args.device,
                       log_suffix=f"[sparsity={obs.sparsity}]")
        wlog(args, {"best_acc": acc, "sparsity": obs.sparsity})

    for i in range(1, args.n_recompute + 1):
        sparsity = polynomial_schedule(0.0, args.sparsity, i, args.n_recompute)
        obs.prune(data_loaders["fisher"], sparsity, args.damping, 1,
                  args.n_recompute_samples, _cb, args.check)
    return acc


def gradual_pruning(obs, model, data_loaders, train_sampler, fisher_sampler,
                    opt, lr_scheduler, criterion, args):
    best_acc = 0.0
    for e in range(args.epochs):
        log(args, f"Epoch {e}/{args.epochs}")
        if e % args.pruning_intvl == 0 and e <= args.pruning_epochs:
            sparsity = polynomial_schedule(
                0.05, args.sparsity, e // args.pruning_intvl,
                args.pruning_epochs // args.pruning_intvl)
            obs.prune(data_loaders["fisher"],
                      sparsity,
                      args.damping,
                      args.n_recompute,
                      args.n_recompute_samples,
                      check=args.check)
            acc = evaluate(model,
                           criterion,
                           data_loaders["test"],
                           device=args.device,
                           log_suffix=f"[sparsity={obs.sparsity}]")
            wlog(args, {"acc": acc, "epoch": e})
            best_acc = acc

        if args.distributed:
            train_sampler.set_epoch(e)
            fisher_sampler.set_epoch(e)
        if e > args.pruning_epochs and (
                e - args.pruning_epochs) % args.lr_step_size == 0:
            lr_scheduler.step()
        train_one_epoch(model, criterion, opt, data_loaders["train"],
                        args.device, e, args)
        obs.parameters = obs.parameters * obs.mask

        acc = evaluate(model,
                       criterion,
                       data_loaders["test"],
                       device=args.device,
                       log_suffix=f"[sparsity={obs.sparsity}]")
        wlog(args, {"acc": acc, "epoch": e, "lr": opt.param_groups[0]["lr"]})
        best_acc = max(best_acc, acc)
        if (e % (args.pruning_intvl - 1) == 0
                and e <= args.pruning_epochs) or (e == args.epochs - 1):
            wlog(args, {"best_acc": best_acc, "sparsity": obs.sparsity})
    return best_acc


def main():
    args = parse_args()
    if args.resume == "":
        args.pretrained = True
    args.output_dir = f"{args.output_dir}/{args.dataset}/{args.model}/" \
               f"{args.pruning_strategy}/{args.sparsity}/" \
               f"{args.fisher_type}/{args.fisher_shape}/" \
               f"{args.n_recompute}-{args.n_recompute_samples}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    utils.init_distributed_mode(args)
    print(args)

    if args.rank == 0:
        wandb.init(project="pruning")
        wandb.run.name = f"{args.dataset}/{args.model}/" \
               f"{args.pruning_strategy}/{args.sparsity}/" \
               f"{args.fisher_type}/{args.fisher_shape}/" \
               f"{args.n_recompute}-{args.n_recompute_samples}"

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(
        train_dir, val_dir, args)
    collate_fn = None
    num_classes = len(dataset.classes)
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(
            transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(
            transforms.RandomCutmix(num_classes,
                                    p=1.0,
                                    alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
        collate_fn = lambda batch: mixupcutmix(*default_collate(batch))
    data_loaders = {}
    data_loaders["train"] = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loaders["test"] = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.val_batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True)
    if args.distributed:
        fisher_sampler = DistributedSampler(dataset)
    else:
        fisher_sampler = RandomSampler(dataset)
    data_loaders["fisher"] = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.fisher_batch_size,
        sampler=fisher_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    print("Creating model")
    if args.model == "toy2":
        model = Toy2(1, num_classes)
        model.load_state_dict(
            torch.load(f".data/toy2-mnist-best", map_location=args.device))
    else:
        model = torchvision.models.__dict__[args.model](
            pretrained=args.pretrained, num_classes=1000)
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)

    model.to(args.device)
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.norm_weight_decay is None:
        parameters = model.parameters()
    else:
        param_groups = torchvision.ops._utils.split_normalization_params(model)
        wd_groups = [args.norm_weight_decay, args.weight_decay]
        parameters = [{
            "params": p,
            "weight_decay": w
        } for p, w in zip(param_groups, wd_groups) if p]

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        opt = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        opt = torch.optim.RMSprop(parameters,
                                  lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay,
                                  eps=0.0316,
                                  alpha=0.9)
    elif opt_name == "adamw":
        opt = torch.optim.AdamW(parameters,
                                lr=args.lr,
                                weight_decay=args.weight_decay)
    else:
        raise RuntimeError(
            f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported."
        )

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(opt,
                                                       step_size=1,
                                                       gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs - args.pruning_epochs)
    elif args.lr_scheduler == "exponentiallr":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            opt, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported.")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    pretrained_acc = evaluate(model,
                              criterion,
                              data_loaders["test"],
                              device=args.device,
                              log_suffix="Pretrained")
    wlog(args, {"best_acc": pretrained_acc, "sparsity": 0.0})

    print("Pruning model")
    scopes = get_global_prnning_scope(model)
    obs = create_obs(model, scopes, args)

    if args.pruning_strategy == "oneshot":
        pruned_acc = one_shot_pruning(obs, model, data_loaders, criterion,
                                      args)
    else:
        pruned_acc = gradual_pruning(obs, model, data_loaders, train_sampler,
                                     fisher_sampler, opt, lr_scheduler,
                                     criterion, args)
    log(args, obs)
    wlog(
        args, {
            "pruned_acc": pruned_acc,
            "drop_rate": (pruned_acc - pretrained_acc) / pretrained_acc
        })


if __name__ == "__main__":
    main()
