from pathlib import Path
import os

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.optim.lr_scheduler import LinearLR, MultiStepLR, SequentialLR
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision import transforms

import wandb

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='kfac')
    parser.add_argument('--dir', default='/tmp', type=str)
    parser.add_argument('--name', default='default', type=str)
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--dataset',
                        default='IMAGENET',
                        type=str,
                        choices=['IMAGENET'])
    parser.add_argument('--data-path',
                        default='/sqfs/work/jh210024/data/ILSVRC2012',
                        type=str)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--val-batch-size', default=2048, type=int)
    parser.add_argument('--label-smoothing', default=0.1, type=float)

    parser.add_argument('--model',
                        default='resnet50',
                        type=str,
                        choices=['IMAGENET'])

    parser.add_argument('--epochs', type=int, default=55)
    parser.add_argument('--opt',
                        default='sgd',
                        type=str,
                        choices=['sgd', 'kfac'])
    parser.add_argument('--lr', type=float, default=0.8)
    parser.add_argument('--lr-decay-epoch',
                        nargs='+',
                        type=int,
                        default=[20, 30, 35])
    parser.add_argument('--warmup-factor', type=float, default=0.125)
    parser.add_argument('--warmup-epochs', type=float, default=5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.00005)

    return parser.parse_args()


def setup_print_for_distributed(is_master):
    '''
    This function disables printing when not in master process
    '''
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def setup_wandb_for_distributed(is_master, args):
    if is_master:
        wandb.init(project='kfac')
        wandb.run.name = f'{args.dataset}/{args.model}/{args.name}'
    else:

        def log(*args, **kwargs):
            pass

        wandb.log = log


def init_distributed_mode(args):
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        setup_wandb_for_distributed(True, args)
        return

    args.distributed = True
    args.rank = int(os.environ['RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    args.dist_url = 'env://'
    print(f'distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend,
                                         init_method=args.dist_url,
                                         world_size=args.world_size,
                                         rank=args.rank)
    setup_print_for_distributed(args.rank == 0)
    setup_wandb_for_distributed(args.rank == 0, args)


class Dataset(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.val_batch_size = args.val_batch_size
        self.num_workers = 4
        self.pin_memory = True
        if args.distributed:
            self.train_sampler = DistributedSampler(self.train_dataset)
        else:
            self.train_sampler = RandomSampler(self.train_dataset)
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       sampler=self.train_sampler,
                                       num_workers=4,
                                       pin_memory=True)
        if args.distributed:
            self.val_sampler = DistributedSampler(self.val_dataset)
        else:
            self.val_sampler = RandomSampler(self.val_dataset)
        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=self.val_batch_size,
                                     sampler=self.val_sampler,
                                     num_workers=4,
                                     pin_memory=True)
        self.sampler = None
        self.loader = None

    def train(self):
        self.sampler = self.train_sampler
        self.loader = self.train_loader

    def eval(self):
        self.sampler = self.val_sampler
        self.loader = self.val_loader


class IMAGENET(Dataset):
    def __init__(self, args):
        self.num_classes = 1000
        self.train_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.data_path, 'train'),
            transform=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]))
        self.val_dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.data_path, 'val'),
            transform=transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]))
        super().__init__(args)


def train(epoch, dataset, model, criterion, opt, args):
    dataset.train()
    if args.distributed:
        dataset.sampler.set_epoch(epoch)
    model.train()

    info = f"Epoch {epoch} {{}}/{len(dataset.loader)} Train Loss: {{:.4f}} Acc: {{:.4f}}"

    n = torch.tensor([0.0]).to(args.device)
    loss = torch.tensor([0.0]).to(args.device)
    corrects = torch.tensor([0.0]).to(args.device)
    for i, (inputs, targets) in enumerate(dataset.loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        opt.step()

        with torch.inference_mode():
            n += inputs.shape[0]
            loss += loss * inputs.shape[0]
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == targets)

        if i % 100 == 0:
            print(info.format(i, (loss / n).item(), (corrects / n).item()))

    if args.distributed:
        dist.all_reduce(n, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(corrects, op=dist.ReduceOp.SUM)
    print(info.format(i, (loss / n).item(), (corrects / n).item()))
    wandb.log(
        {
            'train/loss': (loss / n).item(),
            'train/accuracy': (corrects / n).item(),
            'train/lr': opt.param_groups[0]['lr'],
        },
        step=epoch)


def test(epoch, dataset, model, criterion,args):
    info = f"{epoch} Test Loss: {{:.4f}} Acc: {{:.4f}}"
    dataset.eval()
    dataset.sampler.set_epoch(epoch)
    model.eval()

    with torch.inference_mode():
        n = torch.tensor([0.0]).to(args.device)
        loss = torch.tensor([0.0]).to(args.device)
        corrects = torch.tensor([0.0]).to(args.device)
        for i, (inputs, targets) in enumerate(dataset.loader):
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            with torch.inference_mode():
                n += inputs.shape[0]
                loss += loss * inputs.shape[0]
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == targets)

    if args.distributed:
        dist.all_reduce(n, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(corrects, op=dist.ReduceOp.SUM)
    print(info.format((loss / n).item(), (corrects / n).item()))
    wandb.log(
        {
            'test/loss': (loss / n).item(),
            'test/accuracy': (corrects / n).item(),
        },
        step=epoch)


if __name__ == '__main__':
    args = parse_args()
    args.dir = f'{args.dir}/{args.dataset}/{args.model}/{args.name}'
    Path(args.dir).mkdir(parents=True, exist_ok=True)
    init_distributed_mode(args)
    print(args)

    # ========== DATA ==========
    if args.dataset == 'IMAGENET':
        dataset = IMAGENET(args)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        raise ValueError(f'Unknown dataset {args.dataset}')

    # ========== MODEL ==========
    if args.model == 'resnet50':
        model = torchvision.models.resnet50(num_classes=dataset.num_classes)
    else:
        raise ValueError(f'Unknown model {args.model}')
    model.to(args.device)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])

    # ========== OPTIMIZER ==========
    if args.opt == 'sgd':
        opt = torch.optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.opt == 'kfac':
        pass
    else:
        raise ValueError(f'Unknown optimizer {args.opt}')

    # ========== LEARNING RATE SCHEDULER ==========
    lr_scheduler = SequentialLR(opt, [
        LinearLR(opt, args.warmup_factor, total_iters=args.warmup_epochs),
        MultiStepLR(opt, args.lr_decay_epoch, gamma=0.1),
    ], [args.warmup_epochs])

    # ========== TRAINING ==========
    for e in range(args.epochs):
        train(e, dataset,model, criterion, opt, args)
        test(e, dataset,model, criterion, args)
        lr_scheduler.step()
