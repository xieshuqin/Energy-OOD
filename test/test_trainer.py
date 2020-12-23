import torch
import torchvision
from torch.utils.data import Subset

from dataloader import build_trainval_dataset, build_trainval_dataloader, build_test_dataloader
from model import build_model
from trainer import Trainer


def test_trainer():
    train_in, train_out, val_in, val_out = build_trainval_dataset()

    # sample subset for fast failure
    train_in = Subset(train_in, torch.arange(512))
    train_out = Subset(train_out, torch.arange(16))
    val_in = Subset(val_in, torch.arange(512))
    val_out = Subset(val_out, torch.arange(16))

    train_in_loader, train_out_loader, val_in_loader, val_out_loader = \
        build_trainval_dataloader(train_in, train_out, val_in, val_out)
    test_in_loader, test_out_loader = build_test_dataloader()
    model = build_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(model, train_in_loader, train_out_loader, val_in_loader, val_out_loader,
                      test_in_loader, test_out_loader, device)
    trainer.run()


if __name__ == '__main__':
    test_trainer()