import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split


def build_trainval_dataset():
    # In distribution dataset
    in_dataset = torchvision.datasets.CIFAR10(root='./data', download=True, transform=preprocess(True))
    train_in, val_in = random_split(in_dataset, train_val_size(len(in_dataset)))

    # Auxiliary dataset
    aux_dataset = torchvision.datasets.ImageFolder(root='./data/tiny-images/TinyImages', transform=preprocess(True),
                                                   target_transform=aux_dataset_label)
    train_out, val_out = random_split(aux_dataset, train_val_size(len(aux_dataset)))

    return train_in, train_out, val_in, val_out

def build_trainval_dataloader(train_in=None, train_out=None, val_in=None, val_out=None):
    if train_in is None:
        train_in, train_out, val_in, val_out = build_trainval_dataset()

    train_in_loader = torch.utils.data.DataLoader(train_in, 128, shuffle=True, num_workers=4)
    val_in_loader = torch.utils.data.DataLoader(val_in, 128, num_workers=4)

    train_out_loader = torch.utils.data.DataLoader(train_out, 4, shuffle=True, num_workers=4)
    val_out_loader = torch.utils.data.DataLoader(val_out, 4, num_workers=4)

    return train_in_loader, train_out_loader, val_in_loader, val_out_loader

def build_test_dataloader():
    in_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=preprocess(False))
    in_loader = torch.utils.data.DataLoader(in_dataset, 128, num_workers=4)
    ood_dataset = torchvision.datasets.SVHN(root='./data', download=True, transform=preprocess(False),
                                            target_transform=aux_dataset_label)
    out_loader = torch.utils.data.DataLoader(ood_dataset, 256, num_workers=4)
    return in_loader, out_loader

def preprocess(training=True):
    transform = [
        T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    if training:
        transform.insert(0, T.RandomHorizontalFlip())
    return T.Compose(transform)

def aux_dataset_label(label):
    return -1

def ood_dataset_label(label):
    return 1

def train_val_size(length):
    ratio = 0.9
    train_size = int(ratio * length)
    val_size = length - train_size
    return train_size, val_size


if __name__ == '__main__':
    build_trainval_dataloader()
    # dataset = torchvision.datasets.ImageNet('./data', download=True)