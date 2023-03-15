# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
import torch

from mmedit.apis import init_model, restoration_inference, init_coop_model
from mmedit.core import tensor2img, srocc, plcc

import pandas as pd
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA demo')
    parser.add_argument('--config', default='configs/clipiqa/clipiqa_attribute_test.py', help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--file_path', default='/root/4T/dataset/koniq10k/1024x768/', help='path to input image file')
    parser.add_argument('--csv_path', default='/root/4T/dataset/koniq10k/koniq10k_distributions_sets.csv', help='path to input image file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    csv_list = pd.read_csv(args.csv_path, on_bad_lines='skip')
    img_test = csv_list[csv_list.set=='test'].reset_index()

    txt_path = './koniq_resize.txt'
    y_true = csv_list[csv_list.set=='test'].MOS.values

    pred_score = []
    for i in tqdm(range(len(img_test))):
        output, attributes = restoration_inference(model, os.path.join(args.file_path, img_test['image_name'][i]), return_attributes=True)
        output = output.float().detach().cpu().numpy()
        pred_score.append(attributes[0])

    pred_score = np.squeeze(np.array(pred_score))*100


    p_srocc = srocc(pred_score, y_true)
    p_plcc = plcc(pred_score, y_true)

    print(args.checkpoint)
    print('SRCC: {} | PLCC: {}'.\
          format(p_srocc, p_plcc))


if __name__ == '__main__':
    main()
