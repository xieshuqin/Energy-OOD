import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from ood_metrics import calc_metrics

class Trainer(object):
    def __init__(self, model, train_in_loader, train_out_loader,
                 val_in_loader, val_out_loader, test_loader, device):
        # dataloaders
        self.train_in_loader = train_in_loader
        self.train_out_loader = train_out_loader
        self.val_in_loader = val_in_loader
        self.val_out_loader = val_out_loader
        self.test_loader = test_loader

        # training parameters
        self.m_in = -23
        self.m_out = -5
        self.energy_weight = 0.1
        self.n_epochs = 10
        self.device = device

        # model
        self.model = model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4, nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.n_epochs)

        # logger
        self.writter = SummaryWriter()
        self.train_iters = 0
        self.val_iters = 0

    def run(self):
        for epoch in range(1, self.n_epochs+1):
            self.train(epoch)
            self.val(epoch)
            self.scheduler.step()
        self.test()

        return self.model

    def train(self, epoch):
        self.model.train()
        for (x_in, y_in), (x_out, y_out) in zip(self.train_in_loader, self.train_out_loader):
            self.optimizer.zero_grad()
            self.train_iters += 1

            # reshape x_out and y_out
            x_out = torch.reshape(x_out, (4, 3, 64, 32, 32)).permute(0, 2, 1, 3, 4).reshape(-1, 3, 32, 32)
            y_out = -1 * torch.ones(x_out.size(0), dtype=x_in.dtype)

            x = torch.cat((x_in, x_out), dim=0).to(self.device)
            y = torch.cat((y_in, y_out), dim=0).to(self.device)

            logits = self.model(x)
            nll_loss = F.cross_entropy(x[:len(x_in)], y[:len(y_in)])

            energy = -torch.log(torch.sum(torch.exp(logits), dim=1))
            enerygy_loss = (((energy[:len(x_in)]-self.m_in).clamp_min(0))**2).mean() \
                            + (((self.m_out - energy[len(x_in):]).clamp_min(0))**2).mean()
            loss = nll_loss + enerygy_loss*self.energy_weight

            self.writter.add_scalar('Train/Loss', loss.item(), self.train_iters)
            self.writter.add_scalar('Train/NLL', nll_loss.item(), self.train_iters)
            self.writter.add_scalar('Train/EL', enerygy_loss.item(), self.train_iters)

            loss.backward()
            self.optimizer.step()

    def val(self, epoch):
        self.model.eval()
        with torch.no_grad():
            scores, labels = [], []

            for (x_in, y_in), (x_out, y_out) in zip(self.val_in_loader, self.val_out_loader):
                self.val_iters += 1

                # reshape x_out and y_out
                x_out = torch.reshape(x_out, (4, 3, 64, 32, 32)).permute(0, 2, 1, 3, 4).reshape(-1, 3, 32, 32)
                y_out = -1 * torch.ones(x_out.size(0), dtype=x_in.dtype)

                x = torch.cat((x_in, x_out), dim=0).to(self.device)
                y = torch.cat((y_in, y_out), dim=0).to(self.device)

                logits = self.model(x)
                nll_loss = F.cross_entropy(x[:len(x_in)], y[:len(y_in)])

                energy = -torch.log(torch.sum(torch.exp(logits), dim=1))
                enerygy_loss = (((energy[:len(x_in)] - self.m_in).clamp_min(0)) ** 2).mean() \
                               + (((self.m_out - energy[len(x_in):]).clamp_min(0)) ** 2).mean()
                loss = nll_loss + enerygy_loss*self.energy_weight

                self.writter.add_scalar('Val/Loss', loss.item(), self.val_iters)
                self.writter.add_scalar('Val/NLL', nll_loss.item(), self.val_iters)
                self.writter.add_scalar('Val/EL', enerygy_loss.item(), self.val_iters)

                scores.append(-energy)
                labels.append(torch.eq(y, -1))

            scores = torch.cat(scores).cpu().numpy()
            labels = torch.cat(labels).cpu().numpy()
            metric = calc_metrics(scores, labels)

            for name, value in metric.items():
                self.writter.add_scalar('Val/{}'.format(name), value, epoch)

    def test(self):
        scores, labels = [], []

        self.model.eval()
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)

                logits = self.model(x)
                energy = -torch.log(torch.sum(torch.exp(logits), dim=1))
                score = -energy

                scores.append(score)
                labels.append(y)

        scores = torch.cat(scores).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()
        metric = calc_metrics(scores, labels)

        for name, value in metric.items():
            self.writter.add_scalar('Test/{}'.format(name), value)
