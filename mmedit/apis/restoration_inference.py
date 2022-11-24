# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.parallel import collate, scatter

from mmedit.datasets.pipelines import Compose
from PIL import ImageEnhance, Image, ImageFilter
from torchvision.transforms import Compose as torchcompose
from torchvision.transforms import Resize, CenterCrop, ToTensor, Normalize
import numpy as np
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def restoration_inference(model, img, return_attributes=False, process_type=0):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): File path of input image.

    Returns:
        Tensor: The predicted restoration result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if process_type == 1:
        # remove gt from test_pipeline_8x
        keys_to_remove = ['gt', 'gt_path']
        for key in keys_to_remove:
            for pipeline in list(cfg.test_pipeline_8x):
                if 'key' in pipeline and key == pipeline['key']:
                    cfg.test_pipeline_8x.remove(pipeline)
                if 'keys' in pipeline and key in pipeline['keys']:
                    pipeline['keys'].remove(key)
                    if len(pipeline['keys']) == 0:
                        cfg.test_pipeline_8x.remove(pipeline)
                if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                    pipeline['meta_keys'].remove(key)
        test_pipeline = Compose(cfg.test_pipeline_8x)
    elif process_type == 0:
        # remove gt from test_pipeline
        keys_to_remove = ['gt', 'gt_path']
        for key in keys_to_remove:
            for pipeline in list(cfg.test_pipeline):
                if 'key' in pipeline and key == pipeline['key']:
                    cfg.test_pipeline.remove(pipeline)
                if 'keys' in pipeline and key in pipeline['keys']:
                    pipeline['keys'].remove(key)
                    if len(pipeline['keys']) == 0:
                        cfg.test_pipeline.remove(pipeline)
                if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                    pipeline['meta_keys'].remove(key)
        test_pipeline = Compose(cfg.test_pipeline)
    else:
        # remove gt from test_pipeline
        keys_to_remove = ['gt', 'gt_path']
        for key in keys_to_remove:
            for pipeline in list(cfg.test_pipeline_crop):
                if 'key' in pipeline and key == pipeline['key']:
                    cfg.test_pipeline_crop.remove(pipeline)
                if 'keys' in pipeline and key in pipeline['keys']:
                    pipeline['keys'].remove(key)
                    if len(pipeline['keys']) == 0:
                        cfg.test_pipeline_crop.remove(pipeline)
                if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                    pipeline['meta_keys'].remove(key)
        test_pipeline = Compose(cfg.test_pipeline_crop)
    # prepare data
    data = dict(lq_path=img)
    data = test_pipeline(data)

    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(test_mode=True, **data)

    if return_attributes:
        return result['output'], result['attributes']
    else:
        return result['output']

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    ar = ar.transpose(1, 0, 2)

    return ar.astype(np.float32) / 255.

def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    ar = ar.transpose(1, 0, 2)

    return Image.fromarray(ar)

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def resize_process(img, size=224):
    '''
    224
    288
    384
    '''
    process = torchcompose([
        # Resize(size, interpolation=BICUBIC),
        # Resize((size, size), interpolation=BICUBIC),
        # CenterCrop(size),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    return process(img)

def transform_process(img, _transform_dict, noise_std=None):
    _color_jitter = ColorJitter(_transform_dict, noise_std)
    return _color_jitter(img)

def gaussian_noise(image, std):
    ## Give PIL, return the noisy PIL
    mean=0
    gauss=np.random.normal(loc=mean,scale=std,size=image.shape)
    noisy=image+gauss
    noisy=np.clip(noisy,0,1).astype(np.float32)

    return noisy

def noise_process(img, std):
    img = pil_to_np(img)
    img = gaussian_noise(img, std)
    return np_to_pil(img)

class ColorJitter(object):
    def __init__(self, transform_dict, noise_std=None):
        transform_type_dict = dict(
            brightness=ImageEnhance.Brightness, contrast=ImageEnhance.Contrast,
            sharpness=ImageEnhance.Sharpness,   color=ImageEnhance.Color
        )
        self.transforms = [(transform_type_dict[k], transform_dict[k]) for k in transform_dict]
        self.noise_std = noise_std

    def __call__(self, img):
        out = img
        # rand_num = np.random.uniform(0, 1, len(self.transforms))
        for i, (transformer, alpha) in enumerate(self.transforms):
            # r = alpha * (rand_num[i]*2.0 - 1.0) + 1   # r in [1-alpha, 1+alpha)
            r = 1+alpha
            out = transformer(out).enhance(r)
        if self.noise_std:
            out = noise_process(out, self.noise_std)

        return out

def restoration_demo_inference(model, _transform_dict, img, noise_std=None, return_attributes=False):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): File path of input image.

    Returns:
        Tensor: The predicted restoration result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # remove gt from test_pipeline
    keys_to_remove = ['gt', 'gt_path']
    for key in keys_to_remove:
        for pipeline in list(cfg.test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                cfg.test_pipeline.remove(pipeline)
            if 'keys' in pipeline and key in pipeline['keys']:
                pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0:
                    cfg.test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                pipeline['meta_keys'].remove(key)
    # build the data pipeline
    # test_pipeline = Compose(cfg.test_pipeline)
    # prepare data
    pic = Image.open(img)
    # print(os.path.join(data_dir,test_info['image_name'][k*batch_size]), flush=True)
    pic = transform_process(pic, _transform_dict=_transform_dict, noise_std=noise_std)
    pic_to_save = pic
    pic = resize_process(pic, size=224)
    # data = dict(lq_path=img)
    # data = test_pipeline(data)
    data = {'meta': {}, 'lq': pic}
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(test_mode=True, **data)

    if return_attributes:
        return result['output'], result['attributes'], pic_to_save
    else:
        return result['output']

def restoration_noisy_inference(model, std, img, return_attributes=False):
    """Inference image with the model.

    Args:
        model (nn.Module): The loaded model.
        img (str): File path of input image.

    Returns:
        Tensor: The predicted restoration result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # remove gt from test_pipeline
    keys_to_remove = ['gt', 'gt_path']
    for key in keys_to_remove:
        for pipeline in list(cfg.test_pipeline):
            if 'key' in pipeline and key == pipeline['key']:
                cfg.test_pipeline.remove(pipeline)
            if 'keys' in pipeline and key in pipeline['keys']:
                pipeline['keys'].remove(key)
                if len(pipeline['keys']) == 0:
                    cfg.test_pipeline.remove(pipeline)
            if 'meta_keys' in pipeline and key in pipeline['meta_keys']:
                pipeline['meta_keys'].remove(key)
    # build the data pipeline
    # test_pipeline = Compose(cfg.test_pipeline)
    # prepare data
    pic = Image.open(img)
    # print(os.path.join(data_dir,test_info['image_name'][k*batch_size]), flush=True)
    pic = noise_process(pic, std)
    pic = resize_process(pic, size=224)
    # data = dict(lq_path=img)
    # data = test_pipeline(data)
    data = {'meta': {}, 'lq': pic}
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(test_mode=True, **data)

    if return_attributes:
        return result['output'], result['attributes']
    else:
        return result['output']
