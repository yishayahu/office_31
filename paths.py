import sys

if sys.platform == 'win32':
    data_path = r'C:\Users\Y\PycharmProjects\domain_adaptation_images'
    pretrained_models_path = r'C:\Users\Y\PycharmProjects\domain_adaptation_out'
    out_path = r'C:\Users\Y\PycharmProjects\domain_adaptation_out'
else:
    data_path ='/home/dsi/shaya/domain_adaptation_images2/'
    pretrained_models_path = '/home/dsi/shaya/domain_adaptation_images/outs/'
    out_path ='/home/dsi/shaya/domain_adaptation_images/outs/'