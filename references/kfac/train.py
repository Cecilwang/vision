from pathlib import Path
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.optim.lr_scheduler import LinearLR, MultiStepLR, SequentialLR
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
from torchvision import transforms

import wandb

from kfac import KFAC


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='kfac')
    parser.add_argument('--dir', default='/tmp', type=str)
    parser.add_argument('--name', default='default', type=str)
    parser.add_argument('--device', default='cuda', type=str)

    parser.add_argument('--dataset',
                        default='IMAGENET',
                        type=str,
                        choices=['IMAGENET', 'MNIST'])
    parser.add_argument('--data-path',
                        default='/sqfs/work/jh210024/data/ILSVRC2012',
                        type=str)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--val-batch-size', default=2048, type=int)
    parser.add_argument('--label-smoothing', default=0.1, type=float)

    parser.add_argument('--model',
                        default='resnet50',
                        type=str,
                        choices=['resnet50', 'MNISTToy'])

    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--opt',
                        default='kfac',
                        type=str,
                        choices=['sgd', 'kfac'])
    parser.add_argument('--lr', type=float, default=0.8)
    parser.add_argument('--warmup-factor', type=float, default=0.125)
    parser.add_argument('--warmup-epochs', type=float, default=5)
    parser.add_argument('--lr-decay-epoch',
                        nargs='+',
                        type=int,
                        default=[20, 30, 35])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=0.00005)
    parser.add_argument('--damping', type=float, default=1e-3)

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


class MNIST(Dataset):
    def __init__(self, args):
        self.num_classes = 10
        transform = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
        ])
        self.train_dataset = torchvision.datasets.MNIST(args.data_path,
                                                        train=True,
                                                        download=True,
                                                        transform=transform)
        self.val_dataset = torchvision.datasets.MNIST(args.data_path,
                                                      train=False,
                                                      download=False,
                                                      transform=transform)
        super().__init__(args)


class MNISTToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Metric(object):
    def __init__(self, device):
        self._n = torch.tensor([0.0]).to(device)
        self._loss = torch.tensor([0.0]).to(device)
        self._corrects = torch.tensor([0.0]).to(device)

    def update(self, n, loss, outputs, targets):
        with torch.inference_mode():
            self._n += n
            self._loss += loss * n
            _, preds = torch.max(outputs, 1)
            self._corrects += torch.sum(preds == targets)

    def sync(self):
        dist.all_reduce(self._n, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._corrects, op=dist.ReduceOp.SUM)

    @property
    def loss(self):
        return (self._loss / self._n).item()

    @property
    def accuracy(self):
        return (self._corrects / self._n).item()

    def __str__(self):
        return f"Loss: {self.loss:.4f}, Acc: {self.accuracy:.4f}"


def train(epoch, dataset, model, criterion, opt, args):
    dataset.train()
    if args.distributed:
        dataset.sampler.set_epoch(epoch)
    model.train()

    metric = Metric(args.device)
    for i, (inputs, targets) in enumerate(dataset.loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        opt.zero_grad()

        if args.opt in ['kfac'] and opt.steps % opt.TCov == 0:
            opt.acc_stats = True
            with torch.no_grad():
                sampled_y = torch.multinomial(
                    torch.nn.functional.softmax(outputs.cpu().data, dim=1),
                    1).squeeze().to(args.device)
            loss_sample = criterion(outputs, sampled_y)
            loss_sample.backward(retain_graph=True)
            opt.acc_stats = False
            opt.zero_grad()

        loss.backward()
        opt.step()

        metric.update(inputs.shape[0], loss, outputs, targets)

        if i % 100 == 0:
            print(f"Epoch {epoch} {i}/{len(dataset.loader)} Train {metric}")

    if args.distributed:
        metric.sync()
    print(
        f"Epoch {epoch} {i}/{len(dataset.loader)} Train {metric} LR: {opt.param_groups[0]['lr']}"
    )
    wandb.log(
        {
            'train/loss': metric.loss,
            'train/accuracy': metric.accuracy,
            'train/lr': opt.param_groups[0]['lr']
        }, epoch)


def test(epoch, dataset, model, criterion, args):
    dataset.eval()
    if args.distributed:
        dataset.sampler.set_epoch(epoch)
    model.eval()

    metric = Metric(args.device)

    with torch.inference_mode():
        for i, (inputs, targets) in enumerate(dataset.loader):
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            metric.update(inputs.shape[0], loss, outputs, targets)

    if args.distributed:
        metric.sync()
    print(f"Epoch {epoch} Test {metric}")
    wandb.log({
        'test/loss': metric.loss,
        'test/accuracy': metric.accuracy
    }, epoch)


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
    elif args.dataset == 'MNIST':
        dataset = MNIST(args)
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        raise ValueError(f'Unknown dataset {args.dataset}')

    # ========== MODEL ==========
    if args.model == 'resnet50':
        model = torchvision.models.resnet50(num_classes=dataset.num_classes)
    elif args.model == 'MNISTToy':
        model = MNISTToy()
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
        opt = KFAC(model,
                   lr=args.lr,
                   momentum=args.momentum,
                   weight_decay=args.weight_decay,
                   damping=args.damping,
                   damping_warmup_steps=0,
                   n_distributed=args.world_size)
    else:
        raise ValueError(f'Unknown optimizer {args.opt}')

    # ========== LEARNING RATE SCHEDULER ==========
    if args.warmup_epochs > 0:
        lr_scheduler = SequentialLR(opt, [
            LinearLR(opt, args.warmup_factor, total_iters=args.warmup_epochs),
            MultiStepLR(opt, args.lr_decay_epoch, gamma=0.1),
        ], [args.warmup_epochs])
    else:
        lr_scheduler = MultiStepLR(opt, args.lr_decay_epoch, gamma=0.1)

    # ========== TRAINING ==========
    for e in range(args.epochs):
        train(e, dataset, model, criterion, opt, args)
        test(e, dataset, model, criterion, args)
        lr_scheduler.step()
