import argparse
import random
from functools import partial

import numpy as np
import torch
import yaml
from torch import nn

import dataset
import trainer
from torch.optim import *
from torchvision.models import *
from torch.nn import Linear


class Config:
    def parse(self, raw):
        for k, v in raw.items():
            if type(v) == dict:
                curr_func = v.pop('func')
                return_as_class = v.pop('as_class', False)
                assert curr_func in globals()
                for key, val in v.items():
                    if type(val) == str and val in globals():
                        v[key] = globals()[val]
                v = partial(globals()[curr_func], **v)
                if return_as_class:
                    v = v()
            elif v in globals():
                v = globals()[v]
            setattr(self, k, v)

    def __init__(self, raw):
        self._second_round = raw.pop('SECOND_ROUND') if 'SECOND_ROUND' in raw else {}
        self.parse(raw)

    def second_round(self):
        self.parse(self._second_round)


def fix_seed(seed=0xBadCafe):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args=None):
    cli = argparse.ArgumentParser()
    cli.add_argument("--exp_name", default='debug')
    cli.add_argument("--config", default='configs/amazon_target.yml')
    cli.add_argument("--device", default='cuda:0')
    cli.add_argument("--split_size")
    opts = cli.parse_args(args)
    fix_seed()
    cfg = Config(yaml.safe_load(open(opts.config, 'r')))
    cfg.second_round()
    cfg.model.fc = cfg.fc
    if cfg.train_only_source:
        test_ds, val_ds, source_ds = dataset.get_dataset(cfg.dataset_name)
        t = trainer.Trainer(source_ds, None, val_ds, test_ds, cfg, opts.device, opts.exp_name)
        t.train()
    else:
        _, _, source_ds = dataset.get_dataset(cfg.dataset_source_name)
        target_ds, val_ds, test_ds = dataset.get_dataset(cfg.dataset_target_name)
        t = trainer.Trainer(source_ds, target_ds, val_ds, test_ds, cfg, opts.device, opts.exp_name,
                            project_name=f'office_31_{opts.split_size}')
        t.train()


if __name__ == '__main__':
    main()
