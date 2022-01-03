import os
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as torchdata
from torch import nn

from tqdm import tqdm
import wandb
import paths


class Trainer(object):
    def __init__(self, source_ds, target_ds, test_ds, cfg, device, exp_name, project_name):
        print(f"len source_ds is {len(source_ds)}")
        print(f"len test_ds is {len(test_ds)}")
        if target_ds:
            print(f'target_ds len {len(target_ds)}')
        self.ckpt_dir = os.path.join(paths.out_path, exp_name)
        self.res_dir = os.path.join(paths.out_path, exp_name)
        Path(self.ckpt_dir).mkdir(parents=True, exist_ok=True)
        Path(self.res_dir).mkdir(parents=True, exist_ok=True)
        self.step = 0
        self.epoch = 0
        wandb.init(
            project=project_name,
            id=wandb.util.generate_id(),
            name=exp_name,
        )
        self.device = device
        self.source_ds = source_ds
        self.target_ds = target_ds
        self.cfg = cfg
        if cfg.train_only_source:
            self.source_dl = torchdata.DataLoader(source_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=True,
                                                  drop_last=True)
            self.target_dl = None
        elif cfg.train_only_target:
            self.source_dl = None
            self.target_dl = torchdata.DataLoader(target_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=True,
                                                  drop_last=True)
        else:
            self.create_data_loaders()
        self.test_dl = torchdata.DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=True,
                                            drop_last=False)
        self.model = cfg.model
        if getattr(cfg, 'freeze_func', False):
            cfg.freeze_func(self.model)

        if cfg.train_only_source:
            assert not getattr(cfg, 'continue_optimizer', False)
            param_group = [{'params': [], 'lr': cfg.lr}, {'params': [], 'lr': cfg.lr*10}]
            for k, v in self.model.named_parameters():
                if not k.__contains__('fc'):
                    param_group[0]['params'].append(v)
                else:
                    param_group[1]['params'].append(v)
        else:
            self.model.load_state_dict(torch.load(os.path.join(paths.pretrained_models_path, cfg.base_model_path)))
            param_group = [{'params': [], 'lr': cfg.lr}, {'params': [], 'lr': cfg.lr}]
            for k, v in self.model.named_parameters():
                if not k.__contains__('fc'):
                    param_group[0]['params'].append(v)
                else:
                    param_group[1]['params'].append(v)
        self.model = self.model.to(device)
        self.optimizer = cfg.optimizer(param_group, momentum=cfg.momentum, lr=cfg.lr,
                                       weight_decay=getattr(cfg, 'weight_decay', 0.15))
        continue_optimizer = getattr(cfg, 'continue_optimizer', False)
        if continue_optimizer:
            self.optimizer.load_state_dict(torch.load(os.path.join(paths.pretrained_models_path, cfg.base_optim_path)))
            for k, v in self.optimizer.defaults.items():
                for i in range(len(self.optimizer.param_groups)):
                    self.optimizer.param_groups[i][k] = v
        self.criterion = nn.CrossEntropyLoss()

    def create_data_loaders(self):

        target_amount = min((self.epoch // (self.cfg.num_epochs // self.cfg.batch_size))+1,self.cfg.batch_size -1)
        source_amount = self.cfg.batch_size - target_amount
        assert source_amount >= 1

        self.source_dl = torchdata.DataLoader(self.source_ds, batch_size=source_amount, shuffle=True,
                                                  pin_memory=True,
                                                  drop_last=True)

        self.target_dl = torchdata.DataLoader(self.target_ds, batch_size=self.cfg.batch_size - source_amount,
                                              shuffle=True, pin_memory=True,
                                              drop_last=True)
        wandb.log({'source amount': source_amount}, step=self.step)

    def run_val(self, dl, val_or_test):
        self.model.eval()
        bar = tqdm(enumerate(dl), total=len(dl))
        accs = 0
        num_examples = 0
        for i, (inputs, labels) in bar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            num_examples += inputs.size(0)
            with torch.no_grad():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

            accs += torch.sum(preds == labels.data).item()
            if i == len(dl) - 1:
                acc = float(accs / num_examples)
                print(f'{val_or_test} accuracy: {acc} iter: {i}')
                wandb.log({f'{val_or_test} accuracy': acc}, step=self.step)

        return float(accs / num_examples)

    def get_iter_for_step(self):
        if self.cfg.train_only_source:
            while True:
                for x in self.source_dl:
                    yield x
        elif self.cfg.train_only_target or self.source_dl is None:
            while True:
                for x in self.target_dl:
                    yield x
        else:
            while True:
                iter1 = iter(self.source_dl)
                iter2 = iter(self.target_dl)
                try:
                    while True:
                        inputs1, labels1 = next(iter1)
                        inputs2, labels2 = next(iter2)
                        inputs = torch.cat([inputs1, inputs2], dim=0)
                        labels = torch.cat([labels1, labels2], dim=0)
                        yield inputs, labels
                except StopIteration:
                    pass

    def run_train(self):
        losses = []
        accs = 0
        num_examples = 0
        self.model.train()  # Set model to training mode
        data_iter = self.get_iter_for_step()
        bar = tqdm(range(self.cfg.steps_per_epoch))
        for i in bar:
            inputs, labels = next(data_iter)

            self.optimizer.zero_grad()

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            num_examples += inputs.size(0)

            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)

                loss = self.criterion(outputs, labels)

                loss.backward()

                self.optimizer.step()
                # self.scheduler.step()
            self.step += 1
            losses.append(loss.item())

            accs += torch.sum(preds == labels.data).item()

            if i % 10 == 9:
                bar.set_description(
                    f'train loss: {np.mean(losses)} train accuracy: {accs / num_examples} iter: {i} lr is {0}')
                logs = {
                    f'train loss': float(np.mean(losses)),
                    f'train accuracy': float(accs / num_examples),
                    f'lr': 0 ,
                }
                wandb.log(logs, step=self.step)
        return float(accs / num_examples)

    def save_all(self, best=''):
        torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, f'model{best}.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(self.ckpt_dir, f'optim{best}.pth'))

    def train(self):
        best_acc = 0.0

        num_epochs = self.cfg.num_epochs
        for epoch in range(num_epochs):

            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            self.run_train()
            self.save_all()
            if not self.cfg.train_only_source and not self.cfg.train_only_target:
                self.create_data_loaders()
            epoch_acc_val = self.run_val(self.test_dl, 'test')
            if epoch_acc_val > best_acc:
                best_acc = epoch_acc_val
            self.epoch+=1
                # self.save_all(best='best')
        # self.run_val(self.test_dl, 'test')
        self.save_all(best='_final')
