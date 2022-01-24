import torch
import torchvision


def get_dataset_metadata(dataset):
    return {
        "MNIST": {
            "img_shape": (1, 32, 32),
            "n_classes": 10,
            "mean": (0.1307, ),
            "std": (0.3081, )
        },
        "CIFAR10": {
            "img_shape": (3, 32, 32),
            "n_classes": 10,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }[dataset]


def get_data(input_size, dataset, data_dir, batch_size, args):
    metadata = get_dataset_metadata(dataset)

    dataset_train = getattr(torchvision.datasets, dataset)(
        data_dir,
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(input_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(metadata["mean"],
                                             metadata["std"]),
        ]))
    dataset_test = getattr(torchvision.datasets, dataset)(
        data_dir,
        train=False,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(int(input_size * (256 / 224))),
            torchvision.transforms.CenterCrop(input_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(metadata["mean"],
                                             metadata["std"]),
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset_train)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    return dataset_train, dataset_test, train_sampler, test_sampler
