import itertools
import os
import random
import time
from multiprocessing import Process

import torch
from tqdm import tqdm

import paths

from wandb.vendor.pynvml.pynvml import nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, \
    nvmlDeviceGetUtilizationRates, nvmlInit
def find_available_device(my_devices,pp):
    if torch.cuda.is_available():
        wanted_free_mem = 18 * 2 ** 30
        while True:
            for device_num in range(nvmlDeviceGetCount()):
                if device_num in my_devices:
                    continue
                h = nvmlDeviceGetHandleByIndex(device_num)
                info = nvmlDeviceGetMemoryInfo(h)
                gpu_utilize = nvmlDeviceGetUtilizationRates(h)
                if info.free > wanted_free_mem and gpu_utilize.gpu < 3:
                    return device_num
            time.sleep(10)
            for p1,d in pp:
                p1.join(timeout=0)
                if not p1.is_alive():
                    my_devices.remove(d)
                    pp.remove((p1,d))
                    break
    else:
        return 'cpu'
def super_cross():
    if torch.cuda.is_available():
        nvmlInit()
    pp = []
    my_devices = []
    # exps = ['target_base.yml','ftf.yml', 'g_da_paper.yml','target_continue_optimizer_paper.yml']
    exps = ['spottune.yml']
    # exps = ['ftf.yml']
    datasets = ['chexpert', 'cxr14']

    for target_size in [128,256,512,1024]:
        for source_ds, target_ds in itertools.permutations(datasets, 2):
            for i, config in enumerate(exps):
                exp_name = config.split('.')[0]
                exp_name = exp_name + f'_{source_ds}'
                config = os.path.join('configs', config)
                # if i < 1:
                #     if not os.path.exists(os.path.join(paths.out_path, f'{exp_name}/model_final.pth')):
                #         print(f'running source {exp_name}')
                #         curr_device = find_available_device()
                #         my_devices.append(curr_device)
                #         os.system(
                #             f'python main.py --exp_name {exp_name} --config {config}  --source_ds {source_ds} --device cuda:{curr_device}')
                #         my_devices.remove(curr_device)
                #     else:
                #         continue
                exp_name = exp_name + f'_{target_ds}'
                if not os.path.exists(os.path.join(paths.out_path, f'{exp_name}_{target_size}/model_final.pth')):
                    print(f'run training {exp_name}_{target_size} with config {config}')

                    curr_device = find_available_device(my_devices,pp)
                    my_devices.append(curr_device)

                    cmd = f'python main.py --exp_name {exp_name}_{target_size} --config {config}  --device cuda:{curr_device}  --target_size {target_size} --source_ds {source_ds} --target_ds {target_ds} > /home/dsi/shaya/xray/exps/{exp_name}_{target_size}_logs_out.txt 2> /home/dsi/shaya/xray/exps/{exp_name}_{target_size}_logs_errs.txt'

                    print(cmd)
                    p = Process(target=lambda cmd1: os.system(cmd1), args=(cmd,))
                    p.start()
                    time.sleep(4)
                    pp.append((p,curr_device))
                else:
                    print(f'skipping {exp_name}')

    for p2,_ in tqdm(pp, desc='pp wait'):
        p2.join()

if __name__ == '__main__':
    super_cross()