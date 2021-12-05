import os

from main import main

for config in sorted(os.listdir('configs'),key=lambda x:int('source' in  x),reverse=True):
    exp_name = config.split('.')[0]
    config = os.path.join('configs',config)
    print(f'run training {exp_name} with config {config}')
    main(f'--exp_name {exp_name} --config {config}'.split())
