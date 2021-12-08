import os
from multiprocessing import Process

from tqdm import tqdm

import paths
from main import fix_seed
from split_dataset_folder import split

fix_seed()
curr_device = 0
pp = []
for part_ratio in [0.05, 0.1, 0.2]:
    print('spliting_data')
    part_ratio = str(part_ratio).split('.')[1]
    exps = ['webcam_source.yml', 'webcam_target.yml',
            'webcam_target_base.yml', 'webcam_target_continue_optimizer.yml',
            'webcam_target_continue_optimizer_09.yml', 'webcam_target_keep_source.yml',
            'amazon_source.yml', 'amazon_target.yml', 'amazon_target_base.yml',
            'amazon_target_continue_optimizer.yml', 'amazon_target_continue_optimizer_09.yml',
            'amazon_target_keep_source.yml']
    for config in exps:
        exp_name = config.split('.')[0]
        config = os.path.join('configs', config)
        if exp_name == 'webcam_source' or exp_name == 'amazon_source':
            if part_ratio == '05' and not os.path.exists(os.path.join(paths.out_path, f'{exp_name}/model_final.pth')):
                print('running source')
                os.system(f'python main.py --exp_name {exp_name} --config {config}')
            else:
                continue
        if not os.path.exists(os.path.join(paths.out_path, f'{exp_name}_{part_ratio}/model_final.pth')):
            if curr_device>3:
                for p1 in tqdm(pp,desc='pp wait'):
                    p1.join()
                curr_device = 0
                pp = []

            print(f'run training {exp_name}_{part_ratio} with config {config}')
            cmd = f'python main.py --exp_name {exp_name}_{part_ratio} --config {config} --split_size {int(part_ratio)} --device cuda:{curr_device}'
            curr_device += 1
            print(cmd)
            p = Process(target=lambda cmd1: os.system(cmd1[0]), args=(cmd, None))
            pp.append(p)
        else:
            print(f'skipping {exp_name}')
