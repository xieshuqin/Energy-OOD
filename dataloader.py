import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split


def build_trainval_dataloader():
    # In distribution dataset
    in_dataset = torchvision.datasets.CIFAR10(root='./data', download=True, transform=preprocess())
    train_in, val_in = random_split(in_dataset, train_val_size(len(in_dataset)))
    train_in_loader = torch.utils.data.DataLoader(train_in, 128, shuffle=True, num_workers=2)
    val_in_loader = torch.utils.data.DataLoader(val_in, 128, num_workers=2)

    # Auxiliary dataset
    aux_dataset = torchvision.datasets.ImageFolder(root='./data/tiny-images/TinyImages', transform=preprocess(), target_transform=aux_dataset_label)
    train_out, val_out = random_split(aux_dataset, train_val_size(len(aux_dataset)))
    train_out_loader = torch.utils.data.DataLoader(train_out, 4, shuffle=True, num_workers=2)
    val_out_loader = torch.utils.data.DataLoader(val_out, 4, num_workers=2)

    return train_in_loader, train_out_loader, val_in_loader, val_out_loader

def build_test_dataloader():
    ood_dataset = torchvision.datasets.SVHN(root='./data', download=True, transform=preprocess(),
                                            target_transform=ood_dataset_label)
    dataloader = torch.utils.data.DataLoader(ood_dataset, 256, num_workers=2)
    return dataloader

def preprocess():
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

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