import torch
import torchvision

from dataloader import build_trainval_dataloader, build_test_dataloader
from model import build_model
from trainer import Trainer


def main():
    train_in_loader, train_out_loader, val_in_loader, val_out_loader = \
        build_trainval_dataloader()
    test_in_loader, test_out_loader = build_test_dataloader()
    model = build_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(model, train_in_loader, train_out_loader, val_in_loader, val_out_loader,
                      test_in_loader, test_out_loader, device)
    trainer.run()

if __name__ == '__main__':
    main()
