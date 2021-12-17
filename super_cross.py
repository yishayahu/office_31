import os
import time
from multiprocessing import Process

from tqdm import tqdm

import paths
from main import fix_seed

fix_seed()
curr_device = 0
pp = []

exps = ['webcam_source.yml','amazon_source.yml', 'webcam_target.yml',
        'webcam_target_base.yml', 'webcam_target_combined.yml',
        'webcam_target_continue_optimizer_09.yml', 'webcam_target_keep_source.yml',
        'webcam_target_combined_keep_source.yml'
        , 'amazon_target.yml', 'amazon_target_base.yml',
         'amazon_target_continue_optimizer_09.yml','amazon_target_combined.yml',
        'amazon_target_keep_source.yml','amazon_target_combined_keep_source.yml']
for source_size in [20,-1]:
    for target_size in [3,10]:
        for i,config in enumerate(exps):
            exp_name = config.split('.')[0]
            config = os.path.join('configs', config)
            if i < 2:
                if not os.path.exists(os.path.join(paths.out_path, f'{exp_name}_{source_size}/model_final.pth')):
                    print(f'running source {exp_name}_{source_size}')
                    os.system(f'python main.py --exp_name {exp_name}_{source_size} --config {config}  --source_size {source_size}')
                else:
                    continue
            if not os.path.exists(os.path.join(paths.out_path, f'{exp_name}_{source_size}_{target_size}/model_final.pth')):
                if curr_device>7:
                    for p1 in tqdm(pp,desc='pp wait'):
                        p1.join()
                    curr_device = 0
                    pp = []

                print(f'run training {exp_name}_{source_size}_{target_size} with config {config}')
                cmd = f'python main.py --exp_name {exp_name}_{source_size}_{target_size} --config {config}  --device cuda:{curr_device} --source_size {source_size} --target_size {target_size}'
                curr_device += 1
                print(cmd)
                p = Process(target=lambda cmd1: os.system(cmd1), args=(cmd,))
                p.start()
                time.sleep(4)
                pp.append(p)
            else:
                print(f'skipping {exp_name}')
