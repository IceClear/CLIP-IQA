import mmcv
import numpy as np
import pytest
from mmedit.core import tensor2img, srocc, plcc
import pandas as pd
import os
import scipy.io
from PIL import Image

from mmedit.core.evaluation.metrics import (connectivity, gradient_error, mse,
                                            niqe, psnr, reorder_image, sad,
                                            ssim)
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import random
from tqdm import tqdm

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

def copy_img_to_folder(img_path, resize=None):
    img = mmcv.imread(img_path)
    if resize:
        img = np_to_pil(img)
        transform_func = _transform(resize)
        img = transform_func(img)
        img = pil_to_np(img)
    return img

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
    ])

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    ar = ar.transpose(1, 0, 2)

    return ar

def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np, 0, 255).astype(np.uint8)

    ar = ar.transpose(1, 0, 2)

    return Image.fromarray(ar)

def main():
    csv_path = '/mnt/lustre/jywang/code/SPAQ/Annotations/'
    file_path = '/mnt/lustre/jywang/code/SPAQ/TestImage/'
    save_path = '../testpic_spaq'
    mkdirs(save_path)

    csv_list = pd.read_excel(os.path.join(csv_path, 'MOS and Image attribute scores.xlsx'))
    img_test = csv_list['Image name'].values
    mos_list = csv_list['MOS'].values
    bright_list = csv_list['Brightness'].values
    colorful_list = csv_list['Colorfulness'].values
    contrast_list = csv_list['Contrast'].values
    noise_list = 100 - csv_list['Noisiness'].values
    sharp_list = csv_list['Sharpness'].values

    total_number = 11125//5 # 11125
    random.seed(0)
    random.shuffle(img_test)
    random.seed(0)
    random.shuffle(mos_list)

    pred_score = []
    for i in tqdm(range(total_number)):
        output = copy_img_to_folder(os.path.join(file_path, img_test[i]), resize=512)
        mmcv.imwrite(output, os.path.join(save_path,img_test[i].split('.')[0]+'.png'))

if __name__ == '__main__':
    main()
