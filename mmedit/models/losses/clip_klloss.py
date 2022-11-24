import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from ..registry import LOSSES
from mmedit.models.components.clip import clip


def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class KLDLoss_Pytorch(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(KLDLoss_Pytorch, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
    def forward(self, pred, target):
        return F.kl_div(pred.softmax(dim=-1).log(), target.softmax(dim=-1), reduction=self.reduction)*self.loss_weight


@LOSSES.register_module()
class CLIPKLLoss(nn.Module):
    def __init__(self,
                 classnames,
                 backbone_name='RN50',
                 loss_weight=1.0,
                 reduction='mean'
                 ):
        super().__init__()

        self.num_clip = len(classnames)
        self.kl_loss = KLDLoss_Pytorch(loss_weight=loss_weight, reduction=reduction)
        self.clip_model = load_clip_to_cpu(backbone_name)

        for v in self.clip_model.parameters():
            v.requires_grad = False

        self.tokenized_prompts = []
        for i in range(self.num_clip):
            self.tokenized_prompts.append(clip.tokenize(classnames[i]))

    def forward(self, x, image):
        if self.clip_model.training:
            self.clip_model.eval()

        logits_list_gt = []
        for i in range(self.num_clip):
            logits_per_image, logits_per_text = self.clip_model(image.detach(), self.tokenized_prompts[i].to(image.device))
            probs = logits_per_image.softmax(dim=-1)
            logits_list_gt.append(probs[:, 0].unsqueeze(1))
        logits_list_gt = torch.cat(logits_list_gt, dim=1).float()

        return self.kl_loss(x, logits_list_gt)
