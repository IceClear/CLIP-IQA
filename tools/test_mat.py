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

import scipy.io
import matplotlib.pyplot as plt

import random

def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA demo')
    parser.add_argument('--config', default='configs/restorers/clipiqa/clipiqa_attribute_test.py', help='test config file path')
    parser.add_argument('--checkpoint', default='/mnt/lustre/jywang/code/CLIP-IQA/work_dirs/clipiqa_koniq_RN50_ECCV_nolinear/iter_80000.pth', help='checkpoint file') # work_dirs/clipiqa_koniq_fixed_RN50_debug/iter_80000.pth
    parser.add_argument('--file_path', default='/mnt/lustre/jywang/code/SPAQ/TestImage/', help='path to input image file')
    parser.add_argument('--csv_path', default='/mnt/lustre/jywang/code/SPAQ/Annotations/', help='path to input image file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args

def metric_eva(pred_score, gt_score):
    p_srocc = srocc(pred_score, gt_score)
    p_plcc = plcc(pred_score, gt_score)

    return p_srocc, p_plcc

def main():
    args = parse_args()

    csv_list = pd.read_excel(os.path.join(args.csv_path, 'MOS and Image attribute scores.xlsx'))
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

    pred_score = scipy.io.loadmat('/mnt/lustre/jywang/code/your_scores.mat')['scores'][0]
    gt_list = []
    brisque_list = []
    biqi_list = []
    blind_list = []

    for i in range(total_number):
        gt_list.append(csv_list[csv_list['Image name']==pred_score[i][0][0].split('.')[0]+'.jpg']['MOS'].values[0])
        brisque_list.append(pred_score[i][1][0][0])
        biqi_list.append(pred_score[i][2][0][0])
        blind_list.append(pred_score[i][3][0][0])

    pred_score = np.squeeze(np.array(brisque_list))
    gt_score = np.squeeze(np.array(gt_list))
    p_srocc, p_plcc = metric_eva(pred_score, gt_score)
    print('{}--> SRCC: {} | PLCC: {}'.\
          format('brisque', p_srocc, p_plcc))

    pred_score = np.squeeze(np.array(biqi_list))
    p_srocc, p_plcc = metric_eva(pred_score, gt_score)
    print('{}--> SRCC: {} | PLCC: {}'.\
          format('biqi', p_srocc, p_plcc))

    pred_score = np.squeeze(np.array(blind_list))
    p_srocc, p_plcc = metric_eva(pred_score, gt_score)
    print('{}--> SRCC: {} | PLCC: {}'.\
          format('blind', p_srocc, p_plcc))


if __name__ == '__main__':
    main()
