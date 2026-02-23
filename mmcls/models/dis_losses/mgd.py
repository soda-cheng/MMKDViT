import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.registry import MODELS



@MODELS.register_module()
class MGDLoss(nn.Module):
    def __init__(self,
                 name,
                 use_this,
                 student_channels,
                 teacher_channels,
                 alpha_mgd=0.00007,
                 lambda_mgd=0.15,
                 ):
        super(MGDLoss, self).__init__()
        self.alpha_mgd = alpha_mgd
        self.lambda_mgd = lambda_mgd

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.mask_token = nn.Parameter(torch.zeros(1, 1, teacher_channels))

        self.generation = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1))

    def forward(self,
                preds_S,
                preds_T):
        """Forward function.
        Args:
            preds_S(List): [B*2*N*D, B*N*D], student's feature map
            preds_T(List): [B*2*N*D, B*N*D], teacher's feature map
        """
        high_s = preds_S[1]
        high_t = preds_T[1]

        loss_mse = nn.MSELoss(reduction='sum')

        '''Generated Feature'''
        if self.align is not None:
            x = self.align(high_s)
        else:
            x = high_s

        # Mask tokens
        B, N, D = x.shape
        x, mat, ids, ids_masked = self.random_masking(x, self.lambda_mgd)
        mask_tokens = self.mask_token.repeat(B, N - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x_, dim=1, index=ids.unsqueeze(-1).repeat(1, 1, D))
        mask = mat.unsqueeze(-1)

        hw = int(N ** 0.5)
        x = x.reshape(B, hw, hw, D).permute(0, 3, 1, 2)
        x = self.generation(x).flatten(2).transpose(1, 2)

        loss_gen = loss_mse(torch.mul(x, mask), torch.mul(high_t, mask))
        loss_gen = loss_gen / B * self.alpha_mgd / self.lambda_mgd

        return loss_gen

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_masked = ids_shuffle[:, len_keep:L]

        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_keep, mask, ids_restore, ids_masked