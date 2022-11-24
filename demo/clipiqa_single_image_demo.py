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

import plotly.graph_objects as go
import plotly.offline as pyo


def parse_args():
    parser = argparse.ArgumentParser(description='ClipIQA demo')
    parser.add_argument('--config', default='configs/clipiqa/clipiqa_attribute_test.py', help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument('--file_path', default='/root/4T/dataset/AVA/images-ava/images/935405.jpg', help='path to input image file')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))


    # attribute_list = ['Quality', 'Brightness', 'Sharpness', 'Noisiness', 'Colorfulness', 'Contrast']
    attribute_list = ['Aesthetic', 'Happy', 'Natural', 'New', 'Scary', 'Complex']
    attribute_list = [*attribute_list, attribute_list[0]]

    angles = np.linspace(0, 2*np.pi, len(attribute_list), endpoint=False)
    output, attributes = restoration_inference(model, os.path.join(args.file_path), return_attributes=True)
    output = output.float().detach().cpu().numpy()
    attributes = attributes.float().detach().cpu().numpy()[0]
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print(attributes)

    attributes = [*attributes, attributes[0]]

    fig = go.Figure(
        data=[
            go.Scatterpolar(r=attributes, theta=attribute_list, fill='toself'),
        ],
        layout=go.Layout(
            title=go.layout.Title(text='Attributes'),
            polar={'radialaxis': {'visible': True}},
            showlegend=False,
        )
    )

    fig.update_xaxes(tickfont_family="Arial Black")

    fig.write_image('./test.svg', engine="kaleido")



if __name__ == '__main__':
    main()
