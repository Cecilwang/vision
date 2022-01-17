from pathlib import Path
import os

import torch
from torch import nn
from torch.utils.data.dataloader import default_collate
import torchvision
import wandb

from asdfghjkl import SHAPE_FULL, SHAPE_LAYER_WISE, SHAPE_KRON, SHAPE_DIAG

import transforms
from train import train_one_epoch, evaluate, load_data
import utils
from obs import Scope
from obs import FullOBS, LayerOBS, KronOBS, NoneOBS, FullWoodOBS


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Pruning")

    parser.add_argument("--dataset",
                        default="imagenet",
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
        default=64,
        type=int,
        help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--fisher-gb", default=10, type=int)

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
                        default=1e-3,
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
    #parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    #parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    #parser.add_argument(
    #    "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    #)
    #parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    #parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    #parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
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
    parser.add_argument("--fisher_type",
                        type=str,
                        default="fisher_emp",
                        choices=["fisher_exact", "fisher_mc", "fisher_emp"])
    parser.add_argument("--fisher_shape",
                        type=str,
                        default="full",
                        choices=[
                            "full", "layer_wise", "kron", "unit_wise",
                            "full_wood", "none"
                        ])
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
    parser.add_argument("--n_recompute", type=int, default=10)
    parser.add_argument("--n_recompute_samples", type=int, default=4096)

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
        "none": NoneOBS
    }[args.fisher_shape](model, scopes, args.fisher_type, args.world_size)
    if args.fisher_shape in [SHAPE_KRON, SHAPE_LAYER_WISE]:
        obs.normalize = args.layer_normalize
    if args.fisher_shape == SHAPE_KRON:
        obs.fast_inv = args.kfac_fast_inv
    return obs


def wlog(a, e, n, s, args):
    if args.rank == 0:
        wandb.log({"acc": a, "epoch": e, "n_zero": n, "sparsity": s})


def log(args, *v1, **v2):
    if args.rank == 0:
        print(*v1, **v2)


def polynomial_schedule(start, end, i, n):
    scale = end - start
    progress = min(float(i) / n, 1.0)
    remaining_progress = (1.0 - progress)**2
    return end - scale * remaining_progress


def one_shot_pruning(obs, model, data_loaders, criterion, args):
    def _cb():
        acc = evaluate(model,
                       criterion,
                       data_loaders["test"],
                       device=args.device,
                       log_suffix=f"[sparsity={obs.sparsity}]")
        wlog(acc, 0, obs.n_zero, obs.sparsity, args)

    obs.prune(data_loaders["fisher"], args.sparsity, args.damping,
              args.n_recompute, args.n_recompute_samples, args.fisher_gb, _cb,
              args.check)


def gradual_pruning(obs, model, data_loaders, train_sampler, opt, criterion,
                    args):
    for e in range(1, args.epochs + 1):
        log(args, f"Epoch {e}/{args.epochs}")
        if e == 1 or ((e % 5 == 0) and e<=40):
            sparsity = 0.05 if e == 1 else polynomial_schedule(0, args.sparsity, e//5, 40//5)
            obs.prune(data_loaders["fisher"],
                      sparsity,
                      args.damping,
                      1,
                      args.n_recompute_samples,
                      args.fisher_gb,
                      check=args.check)
            acc = evaluate(model,
                           criterion,
                           data_loaders["test"],
                           device=args.device,
                           log_suffix=f"[sparsity={obs.sparsity}]")
            wlog(acc, e, obs.n_zero, obs.sparsity, args)

        if args.distributed:
            train_sampler.set_epoch(e)
        train_one_epoch(model, criterion, opt, data_loaders["train"],
                        args.device, e, args)
        obs.parameters = obs.parameters * obs.mask

        acc = evaluate(model,
                       criterion,
                       data_loaders["test"],
                       device=args.device,
                       log_suffix=f"[sparsity={obs.sparsity}]")
        wlog(acc, e, obs.n_zero, obs.sparsity, args)


def main():
    args = parse_args()
    if args.resume == "":
        args.pretrained = True
    args.output_dir = f"{args.output_dir}/{args.dataset}/{args.model}/" \
               f"{args.pruning_strategy}/" \
               f"{args.fisher_type}/{args.fisher_shape}/" \
               f"{args.n_recompute}-{args.n_recompute_samples}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    utils.init_distributed_mode(args)
    print(args)

    if args.rank == 0:
        wandb.init(project="pruning")
        wandb.run.name = args.output_dir

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
    data_loaders["fisher"] = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.fisher_batch_size,
        sampler=torch.utils.data.RandomSampler(dataset),
        num_workers=args.workers,
        pin_memory=True,
    )
    data_loaders["test"] = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.val_batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True)

    print("Creating model")
    model = torchvision.models.__dict__[args.model](pretrained=args.pretrained,
                                                    num_classes=num_classes)
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

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    acc = evaluate(model,
                   criterion,
                   data_loaders["test"],
                   device=args.device,
                   log_suffix="Pretrained")
    wlog(acc, 0, 0, 0.0, args)

    print("Pruning model")
    scopes = get_global_prnning_scope(model)
    obs = create_obs(model, scopes, args)

    if args.pruning_strategy == "oneshot":
        one_shot_pruning(obs, model, data_loaders, criterion, args)
    else:
        gradual_pruning(obs, model, data_loaders, train_sampler, opt,
                        criterion, args)

    log(args, obs)


if __name__ == "__main__":
    main()
