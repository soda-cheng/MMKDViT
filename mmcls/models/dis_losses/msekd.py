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
                 alpha_msekd=0.00003,
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
        low_s = preds_S[0]
        low_t = preds_T[0]

        B = low_s.shape[0]
        loss_mse = nn.MSELoss(reduction='sum')

        if self.align2 is not None:
            for i in range(2):
                if i == 0:
                    xc = self.align2[i](low_s[:,i]).unsqueeze(1)
                else:
                    xc = torch.cat((xc, self.align2[i](low_s[:,i]).unsqueeze(1)),dim=1)
        else:
            xc = low_s

        loss_lr = loss_mse(xc, low_t) / B * self.alpha_msekd
        return loss_lr



