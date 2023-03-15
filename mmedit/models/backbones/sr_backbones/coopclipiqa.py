# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint
from mmcv.runner.checkpoint import _load_checkpoint_with_prefix
import mmcv
import os

from mmedit.models.backbones.sr_backbones.rrdb_net import RRDB
from mmedit.models.builder import build_component
from mmedit.models.common import PixelShufflePack, make_layer
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger

from torch.nn import functional as F

from mmedit.models.components.clip import clip
from mmedit.models.components.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import copy
from torchvision import models

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

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, n_ctx=16, ctx_init="", cfg_imsize=224, class_specify=False, class_token_position='middle'):
        super().__init__()
        n_cls = len(classnames)
        _tokenizer = _Tokenizer()
        self.clip_model = clip_model
        self.clip_model.requires_grad_(False)

        dtype = self.clip_model.dtype
        ctx_dim = self.clip_model.ln_final.weight.shape[0]
        clip_imsize = self.clip_model.visual.input_resolution
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = self.clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if class_specify:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = class_token_position

    def forward(self):
        if self.clip_model.training:
            self.clip_model.eval()

        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, classnames, backbone_name='ViT-B/32', n_ctx=16, ctx_init="", cfg_imsize=224, class_specify=False, class_token_position='middle'):
        super().__init__()
        clip_model = load_clip_to_cpu(backbone_name)
        self.prompt_learner = PromptLearner(classnames, clip_model, n_ctx=n_ctx, ctx_init=ctx_init, cfg_imsize=cfg_imsize, class_specify=class_specify, class_token_position=class_token_position)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, pos_embedding=False, return_token=False):
        if self.image_encoder.training:
            self.image_encoder.eval()
            self.text_encoder.eval()

        if return_token:
            image_features, token_features = self.image_encoder(image.type(self.dtype), return_token=return_token, pos_embedding=pos_embedding)
        else:
            image_features = self.image_encoder(image.type(self.dtype), return_token=return_token, pos_embedding=pos_embedding)

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        if return_token:
            return image_features, token_features, text_features

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits.softmax(dim=-1)

class NonLinearRegressor(nn.Module):
    def __init__(self, n_input, n_output, n_hidden=128):
        super().__init__()

        self.linear_1 = nn.Linear(n_input, n_hidden)
        self.linear_2 = nn.Linear(n_hidden, n_hidden)
        self.linear_3 = nn.Linear(n_hidden, n_output)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, logits):
        x = self.lrelu(self.linear_1(logits))
        x = self.lrelu(self.linear_2(x))
        return self.linear_3(x)

@BACKBONES.register_module()
class CLIPIQAPredictor(nn.Module):
    def __init__(self, classnames, backbone_name='ViT-B/32', n_ctx=16, ctx_init="", cfg_imsize=224, class_specify=False, class_token_position='middle'):
        super().__init__()

        self.num_clip = len(classnames)

        for i in range(self.num_clip):
            disc = CustomCLIP(classnames[i],
                              backbone_name=backbone_name,
                              n_ctx=n_ctx, ctx_init=ctx_init,
                              cfg_imsize=cfg_imsize,
                              class_specify=class_specify,
                              class_token_position=class_token_position)

            for name, param in disc.named_parameters():
                if "prompt_learner" not in name:
                    param.requires_grad = False

            setattr(self, 'clipmodel_{}'.format(i), disc)
        if self.num_clip > 1:
            self.regressor = NonLinearRegressor(n_input=self.num_clip, n_output=1)

    def forward(self, image):
        logits_list = []
        for i in range(self.num_clip):
            disc = getattr(self, 'clipmodel_{}'.format(i))
            logits = disc(image)
            logits_list.append(logits[:, 0].unsqueeze(1))
        logits_list = torch.cat(logits_list, dim=1).float()
        if self.num_clip > 1:
            pred_score = self.regressor(logits_list)
            return pred_score, logits_list
        else:
            return logits_list, logits_list

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')

@BACKBONES.register_module()
class CLIPIQAFixed(nn.Module):
    def __init__(self, classnames, backbone_name='RN50', pos_embedding=False):
        super().__init__()

        self.num_clip = len(classnames)
        self.clip_model = load_clip_to_cpu(backbone_name)
        self.pos_embedding = pos_embedding
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.tokenized_prompts = []
        for i in range(self.num_clip):
            self.tokenized_prompts.append(clip.tokenize(classnames[i]))

    def forward(self, image):
        if self.clip_model.training:
            self.clip_model.eval()

        logits_list = []
        for i in range(self.num_clip):
            logits_per_image, logits_per_text = self.clip_model(image, self.tokenized_prompts[i].to(image.device), self.pos_embedding)
            probs = logits_per_image.softmax(dim=-1)
            logits_list.append(probs[:, 0].unsqueeze(1))
        logits_list = torch.cat(logits_list, dim=1).float()
        pred_score = logits_list
        return pred_score, logits_list

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults: None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
