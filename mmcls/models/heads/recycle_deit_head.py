# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union
from collections import OrderedDict
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import build_activation_layer
from mmengine.model import Sequential
from mmengine.model.weight_init import trunc_normal_

from mmcls.evaluation.metrics import Accuracy
from mmcls.registry import MODELS
from mmcls.structures import ClsDataSample
from .base_head import BaseHead


@MODELS.register_module()
class RecycleDeiTClsHead(BaseHead):
    """Classification head.

    Args:
        loss (dict): Config of classification loss. Defaults to
            ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 hidden_dim: Optional[int] = None,
                 act_cfg: dict = dict(type='Tanh'),
                 init_cfg: dict = dict(type='Constant', layer='Linear', val=0),
                 loss: dict = dict(type='CrossEntropyLoss', loss_weight=1.0),
                 topk: Union[int, Tuple[int]] = (1, ),
                 cal_acc: bool = False,
                 **kwargs):
        super(RecycleDeiTClsHead, self).__init__(init_cfg=init_cfg, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.act_cfg = act_cfg

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.topk = topk
        if not isinstance(loss, nn.Module):
            loss = MODELS.build(loss)
        self.loss_module = loss
        self.cal_acc = cal_acc

        self._init_layers()

    def _init_layers(self):
        """"Init hidden layer if exists."""
        if self.hidden_dim is None:
            layers = [('head', nn.Linear(self.in_channels, self.num_classes))]
            head_dist = nn.Linear(self.in_channels, self.num_classes)
        else:
            layers = [
                ('pre_logits', nn.Linear(self.in_channels, self.hidden_dim)),
                ('act', build_activation_layer(self.act_cfg)),
                ('head', nn.Linear(self.hidden_dim, self.num_classes)),
            ]
            head_dist = nn.Linear(self.hidden_dim, self.num_classes)
        self.layers = Sequential(OrderedDict(layers))
        self.layers.add_module('head_dist', head_dist)

    def init_weights(self):
        """"Init weights of hidden layer if exists."""
        super(RecycleDeiTClsHead, self).init_weights()
        # Modified from ClassyVision
        if hasattr(self.layers, 'pre_logits'):
            # Lecun norm
            trunc_normal_(
                self.layers.pre_logits.weight,
                std=math.sqrt(1 / self.layers.pre_logits.in_features))
            nn.init.zeros_(self.layers.pre_logits.bias)

    def pre_logits(self,
                   feats: Tuple[List[torch.Tensor]]) -> Tuple[torch.Tensor]:
        """The process before the final classification head.

        The input ``feats`` is a tuple of list of tensor, and each tensor is
        the feature of a backbone stage. In ``DeiTClsHead``, we obtain the
        feature of the last stage and forward in hidden layer if exists.
        """
        _, cls_token, dist_token = feats[-1]
        if self.hidden_dim is None:
            return cls_token, dist_token
        else:
            cls_token = self.layers.act(self.layers.pre_logits(cls_token))
            dist_token = self.layers.act(self.layers.pre_logits(dist_token))
            return cls_token, dist_token

    def avg_logits(self, x):
        cls_token, dist_token = x
        avg_logits = (self.layers.head(cls_token) +
                     self.layers.head_dist(dist_token)) / 2
        return avg_logits


    def forward(self, feats: Tuple[List[torch.Tensor]]) -> torch.Tensor:
        """The forward process."""
        if self.training:
            warnings.warn('MMClassification cannot train the '
                          'distilled version DeiT.')
        cls_token, dist_token = self.pre_logits(feats)
        # The final classification head.
        cls_score = (self.layers.head(cls_token) +
                     self.layers.head_dist(dist_token)) / 2
        return cls_score

    def loss(self, feats: Tuple[torch.Tensor],
             data_samples: List[ClsDataSample], **kwargs) -> dict:
        # The part can be traced by torch.fx
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        return losses

    def _get_loss(self, cls_score: torch.Tensor,
                  data_samples: List[ClsDataSample], **kwargs):
        """Unpack data samples and compute loss."""
        # Unpack data samples and pack targets
        if 'score' in data_samples[0].gt_label:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack([i.gt_label.score for i in data_samples])
        else:
            target = torch.cat([i.gt_label.label for i in data_samples])

        # compute loss
        losses = dict()
        loss = self.loss_module(cls_score, target, avg_factor=cls_score.size(0), **kwargs)
        losses['loss'] = loss

        # compute accuracy
        if self.cal_acc:
            assert target.ndim == 1, 'If you enable batch augmentation ' \
                'like mixup during training, `cal_acc` is pointless.'
            acc = Accuracy.calculate(cls_score, target, topk=self.topk)
            losses.update(
                {f'accuracy_top-{k}': a
                 for k, a in zip(self.topk, acc)})

        return losses

    def predict(
        self,
        classifier_t,
        feats: Tuple[torch.Tensor],
        data_samples: List[Union[ClsDataSample, None]] = None
    ) -> List[ClsDataSample]:
        pred_class = classifier_t
        # The part can be traced by torch.fx
        cls_score = self(feats)
        #pred_class = self.classifier(cls_score)
        # The part can not be traced by torch.fx
        predictions = self._get_predictions(pred_class, cls_score, data_samples)
        return predictions

    def _get_predictions(self, pred_class, pred_scores, data_samples):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        pred_scores = F.softmax(pred_scores, dim=1)
        #pred_labels = pred_scores.argmax(dim=1, keepdim=True).detach()
        pred_labels = pred_class.argmax(dim=1, keepdim=True).detach()

        out_data_samples = []
        if data_samples is None:
            data_samples = [None for _ in range(pred_scores.size(0))]

        for data_sample, score, label in zip(data_samples, pred_scores,
                                             pred_labels):
            if data_sample is None:
                data_sample = ClsDataSample()

            data_sample.set_pred_score(score).set_pred_label(label)
            out_data_samples.append(data_sample)
        return out_data_samples
