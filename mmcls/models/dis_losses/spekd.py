import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.registry import MODELS
import torch.fft

@MODELS.register_module()
class SPEKDLoss(nn.Module):
    def __init__(self,
                 name,
                 use_this,
                 student_dims,
                 teacher_dims,
                 temp_spekd=1.0,
                 alpha_spekd=1.0):
        super(SPEKDLoss, self).__init__()
        self.temp = temp_spekd
        self.alpha = alpha_spekd

        if student_dims != teacher_dims:
            self.align = nn.Linear(student_dims, teacher_dims, bias=True)
        else:
            self.align = None

    def forward(self, feat_s, feat_t):
        # 对教师特征对齐到学生维度
        f_s = feat_s[2]
        f_t = feat_t[2]
        # f_s: [256, 64, 192]
        # f_t: [256, 64, 384]

        f_s_aligned = self.align(f_s)

        # 对特征沿 token 维度（N=64）做 FFT，获取频域表示
        freq_s = torch.fft.fft(f_s, dim=1)
        freq_t = torch.fft.fft(f_s_aligned, dim=1)

        # 取复数模长，作为频谱强度
        mag_s = torch.abs(freq_s) + 1e-9  # 避免log(0)
        mag_t = torch.abs(freq_t) + 1e-9

        # L2 归一化
        mag_s = F.normalize(mag_s, p=2, dim=-1)
        mag_t = F.normalize(mag_t, p=2, dim=-1)

        # 计算 KL 散度（按频率维度进行softmax）
        log_s = torch.log(F.softmax(mag_s / self.temp, dim=1) + 1e-9)
        p_t = F.softmax(mag_t / self.temp, dim=1) + 1e-9

        # KL 散度按 batch 取平均
        loss = F.kl_div(log_s, p_t, reduction='batchmean') * (self.temp ** 2)

        return self.alpha * loss

