import itertools
import os
import time
from multiprocessing import Process

from tqdm import tqdm

import paths
from main import fix_seed

fix_seed()
curr_device = 0
pp = []

exps = ['source.yml','target_base.yml', 'g_da_paper.yml','ftf.yml',
        'target_combined_paper.yml', 'target_continue_optimizer_paper.yml',]
datasets = ['amazon', 'webcam', 'dslr']
for source_size in [20]:
    for target_size in [3]:
        for source_ds, target_ds in itertools.permutations(datasets, 2):
            for i, config in enumerate(exps):
                exp_name = config.split('.')[0]
                exp_name = exp_name + f'_{source_ds}'
                config = os.path.join('configs', config)
                if i < 1:
                    if not os.path.exists(os.path.join(paths.out_path, f'{exp_name}_{source_size}/model_final.pth')):
                        print(f'running source {exp_name}_{source_size}')
                        os.system(
                            f'python main.py --exp_name {exp_name}_{source_size} --config {config}  --source_size {source_size} --source_ds {source_ds}')
                    else:
                        continue
                exp_name = exp_name + f'_{target_ds}'
                if not os.path.exists(
                        os.path.join(paths.out_path, f'{exp_name}_{source_size}_{target_size}/model_final.pth')):
                    if curr_device > 3:
                        for p1 in tqdm(pp, desc='pp wait'):
                            p1.join()
                        curr_device = 0
                        pp = []

                    print(f'run training {exp_name}_{source_size}_{target_size} with config {config}')
                    cmd = f'python main.py --exp_name {exp_name}_{source_size}_{target_size} --config {config}  --device cuda:{curr_device} --source_size {source_size} --target_size {target_size} --source_ds {source_ds} --target_ds {target_ds}'
                    curr_device += 1
                    print(cmd)
                    p = Process(target=lambda cmd1: os.system(cmd1), args=(cmd,))
                    p.start()
                    time.sleep(4)
                    pp.append(p)
                else:
                    print(f'skipping {exp_name}')
