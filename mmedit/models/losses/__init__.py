# Copyright (c) OpenMMLab. All rights reserved.
from .composition_loss import (CharbonnierCompLoss, L1CompositionLoss,
                               MSECompositionLoss)
from .feature_loss import LightCNNFeatureLoss
from .gan_loss import DiscShiftLoss, GANLoss, GradientPenaltyLoss
from .gradient_loss import GradientLoss
from .perceptual_loss import (PerceptualLoss, PerceptualVGG,
                              TransferalPerceptualLoss)
from .pixelwise_loss import CharbonnierLoss, L1Loss, MaskedTVLoss, MSELoss
from .utils import mask_reduce_loss, reduce_loss
from .clip_klloss import CLIPKLLoss
from .iqa_loss import EMDLoss, PLCCLoss, SRCCLoss, NiNLoss

__all__ = [
    'L1Loss', 'MSELoss', 'CharbonnierLoss', 'L1CompositionLoss',
    'MSECompositionLoss', 'CharbonnierCompLoss', 'GANLoss',
    'TransferalPerceptualLoss', 'GradientPenaltyLoss', 'PerceptualLoss',
    'PerceptualVGG', 'reduce_loss', 'mask_reduce_loss', 'DiscShiftLoss',
    'MaskedTVLoss', 'GradientLoss', 'LightCNNFeatureLoss', 'CLIPKLLoss',
    'EMDLoss', 'PLCCLoss', 'SRCCLoss', 'NiNLoss'
]
