import os

import paths
from main import main, fix_seed
from split_dataset_folder import split

fix_seed()
for part_ratio in [0.05, 0.1, 0.2]:
    print('spliting_data')
    split(part_ratio=part_ratio)
    part_ratio = str(part_ratio).split('.')[1]
    for config in sorted(os.listdir('configs'), key=lambda x: int('source' in x), reverse=True):
        exp_name = config.split('.')[0]
        config = os.path.join('configs', config)
        if 'source' in exp_name:
            if part_ratio == '05' and not os.path.exists(os.path.join(paths.out_path, f'{exp_name}/model_final.pth')):
                print('running source')
                os.system(f'python main.py --exp_name {exp_name} --config {config}')
            else:
                continue
        if not os.path.exists(os.path.join(paths.out_path, f'{exp_name}_{part_ratio}/model_final.pth')):
            print(f'run training {exp_name}_{part_ratio} with config {config}')
            os.system(f'python main.py --exp_name {exp_name}_{part_ratio} --config {config}')
        else:
            print(f'skipping {exp_name}')
