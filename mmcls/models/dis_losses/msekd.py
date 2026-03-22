import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.registry import MODELS

@MODELS.register_module()
class MSEKDLoss(nn.Module):
    def __init__(self,
                 name,
                 use_this,
                 student_dims,
                 teacher_dims,
                 alpha_msekd=0.00002,
                 ):
        super(MSEKDLoss, self).__init__()
        self.alpha_msekd = alpha_msekd
    
        if student_dims != teacher_dims:
            self.align2 = nn.ModuleList([
                nn.Linear(student_dims, teacher_dims, bias=True)
                for i in range(2)])
            self.align = nn.Linear(student_dims, teacher_dims, bias=True)
        else:
            self.align2 = None
            self.align = None

    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(List): [B*2*N*D, B*N*D], student's feature map
            preds_T(List): [B*2*N*D, B*N*D], teacher's feature map
        """
        ##high_t=(256, 384)
        high_s = preds_S[0]
        high_t = preds_T[0]

        B = high_s.shape[0]
        loss_mse = nn.MSELoss(reduction='sum')

        if self.align is not None:
            x = self.align(high_s)
        else:
            x = high_s


        loss_fd = loss_mse(x, high_t) / B * self.alpha_msekd
        return loss_fd



