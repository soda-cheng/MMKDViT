import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.registry import MODELS


@MODELS.register_module()
class DDVTFDLoss(nn.Module):
    def __init__(self,
                 name,
                 use_this,
                 student_channels,
                 teacher_channels,
                 lambda_ddvtfd=0.1,
                 temp_ddvtfd=0.000002
                 ):
        super(DDVTFDLoss, self).__init__()

        self.lambda_ddvtfd = lambda_ddvtfd
        self.temp_ddvtfd = temp_ddvtfd

        if student_channels != teacher_channels:
            self.align = nn.Linear(student_channels, teacher_channels, bias=True)
        else:
            self.align = None

    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(List): [B*N*D, B*Heads*N*N, B*D], student's feature map
            preds_T(List): [B*N*D, B*Heads*N*N, B*D], teacher's feature map
        """
        high_sa2 = preds_S[0]
        high_ta2 = preds_T[0]
        high_sa = preds_S[1]
        high_ta = preds_T[1]

        '''a Distillation'''
        #high_sa=(256, 3, 65, 65)
        #print(high_sa.shape)
        #print(high_ta.shape)
        #a1_resized = F.interpolate(high_sa.unsqueeze(1), size=(high_sa.size(1), high_ta.size(-1)), mode='nearest').squeeze(1)
        log_a1 = torch.log(high_sa + 1e-8)
        log_a2 = torch.log(high_ta + 1e-8)
        a1_fused = torch.exp(torch.sum(log_a1, dim=1))
        a2_fused = torch.exp(torch.sum(log_a2, dim=1))
        #print(a1_resized.shape)
        a1 = F.softmax(a1_fused / self.temp_ddvtfd, dim=-1)
        a2 = F.softmax(a2_fused / self.temp_ddvtfd, dim=-1)

        loss_a = (
                F.kl_div(torch.log(a1), a2, reduction='batchmean')
                * (self.temp_ddvtfd**2)
        )

        #log_a21 = torch.log(high_sa2 + 1e-8)
        #log_a22 = torch.log(high_ta2 + 1e-8)
        #a21_fused = torch.exp(torch.sum(log_a21, dim=1))
        #a22_fused = torch.exp(torch.sum(log_a22, dim=1))
        #print(a1_resized.shape)
        #a21 = F.softmax(a21_fused / self.temp_ddvtfd, dim=-1)
        #a22 = F.softmax(a22_fused / self.temp_ddvtfd, dim=-1)

        #loss_a2 = (
        #        F.kl_div(torch.log(a21), a22, reduction='batchmean')
        #        * (self.temp_ddvtfd ** 2)
        #)

        #'''BatchNorm Feature Distillation'''
        ##high_t=(256, 64, 384)
        #B = high_s.shape[0]
        #loss_mse = nn.MSELoss(reduction='sum')

        #if self.align is not None:
            #x = self.align(high_s)
        #else:
            #x = high_s

        #D = high_t.shape[-1]
        #batch_norm = CustomBatchNorm(D).cuda()

        #teacher_batch = batch_norm(high_t)
        #student_batch = batch_norm(x)

        #loss_fd = loss_mse(x, high_t) / B * self.lambda_ddvtfd

        loss_ddvtkd = loss_a

        return loss_ddvtkd * self.lambda_ddvtfd

