# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn

from mmcls.registry import MODELS
from mmcls.structures import ClsDataSample
from .base import BaseClassifier


@MODELS.register_module()
class MARKDimage(BaseClassifier):
    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        if data_preprocessor is None:
            data_preprocessor = {}
        # The build process is in MMEngine, so we need to add scope here.
        data_preprocessor.setdefault('type', 'mmcls.ClsDataPreprocessor')

        if train_cfg is not None and 'augments' in train_cfg:
            # Set batch augmentations by `train_cfg`
            data_preprocessor['batch_augments'] = train_cfg

        super(MARKDimage, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        if not isinstance(backbone, nn.Module):
            backbone = MODELS.build(backbone)
        if neck is not None and not isinstance(neck, nn.Module):
            neck = MODELS.build(neck)
        if head is not None and not isinstance(head, nn.Module):
            head = MODELS.build(head)

        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.align = nn.Linear(384, 192, bias=True)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[ClsDataSample]] = None,
                mode: str = 'tensor'):

        if mode == 'tensor':
            feats = self.extract_feat(inputs)
            return self.head(feats) if self.with_head else feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, inputs, stage='neck'):
        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')

        x = self.backbone(inputs)

        if stage == 'backbone':
            return x

        if self.with_neck:
            x = self.neck(x)
        if stage == 'neck':
            return x

        assert self.with_head and hasattr(self.head, 'pre_logits'), \
            "No head or the head doesn't implement `pre_logits` method."
        return self.head.pre_logits(x)

    def loss(self, inputs: torch.Tensor,
             data_samples: List[ClsDataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[ClsDataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feats = self.extract_feat(inputs)
        return self.head.loss(feats, data_samples)

    def predict(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[ClsDataSample]] = None,
                **kwargs) -> List[ClsDataSample]:
        """Predict results from a batch of inputs.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[ClsDataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.
        """
        feats = self.extract_feat(inputs)
        return self.head.predict(feats, data_samples, **kwargs)

    def new_predict(self,
                feat_t_cls,
                inputs: torch.Tensor,
                data_samples: Optional[List[ClsDataSample]] = None,
                **kwargs) -> List[ClsDataSample]:

        tea_cls = feat_t_cls
        feats_s = self.extract_feat(inputs)

        y = self.align(tea_cls)
        feats_s[0][1] = y

        return self.head.predict(feats_s, data_samples, **kwargs)
