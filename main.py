import argparse
import random
from functools import partial

import numpy as np
import torch
import yaml
from torch import nn
from densenet import DenseNet121
import dataset
import trainer
from torch.optim import *
from torchvision.models import *
from torch.nn import Linear
from utils import freeze_model
from spottune_densenet import StDenseNet

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

            elif type(v) != list and v in globals():
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
    cli.add_argument("--config", default='configs/l2sp.yml')
    cli.add_argument("--device", default='cuda:0')
    cli.add_argument("--target_size", default=10,type=int)
    cli.add_argument("--source_ds", default='chexpert')
    cli.add_argument("--target_ds", default='cxr14')
    cli.add_argument("--seed", default=11,type=int)
    opts = cli.parse_args(args)

    fix_seed(opts.seed)
    cfg = yaml.safe_load(open(opts.config, 'r'))
    if not torch.cuda.is_available():
        opts.device = 'cpu'
    base_cfg = yaml.safe_load(open('configs/base_config.yml', 'r'))
    for k, v in base_cfg.items():
        if k not in cfg:
            cfg[k] = v
    cfg['dataset_source_name'] = opts.source_ds
    cfg['dataset_target_name'] = opts.target_ds
    cfg['base_model_path'] = f'source_{opts.source_ds}/model_final.pth'
    cfg['base_optim_path'] = f'source_{opts.source_ds}/optim_final.pth'
    cfg = Config(cfg)
    cfg.second_round()
    # if 'VGG' in str(type(cfg.model)):
    #     cfg.model.classifier[-1] = Linear(4096, 14)
    # else:
    #     cfg.model.fc = cfg.fc
    if cfg.train_only_source:
        source_train, _, test = dataset.get_dataset(cfg.dataset_source_name,opts.target_size)
        t = trainer.Trainer(source_train, None, test, cfg, opts.device, opts.exp_name, project_name=f'xray_b_{opts.seed}')
        t.train()
    else:
        source_train, _, _ = dataset.get_dataset(cfg.dataset_source_name,opts.target_size)
        _, target_train, test = dataset.get_dataset(cfg.dataset_target_name,opts.target_size)
        t = trainer.Trainer(source_train, target_train, test, cfg, opts.device, opts.exp_name,
                            project_name=f'xray_b_{opts.seed}')
        t.train()


if __name__ == '__main__':
    main()
