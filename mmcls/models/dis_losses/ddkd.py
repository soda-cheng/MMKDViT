import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.registry import MODELS

@MODELS.register_module()
class DDKDLoss(nn.Module):
    def __init__(self,
                 name,
                 use_this,
                 temp=1.0,
                 alpha=0.8,
                 beta=0.4,
                 ):
        super(DDKDLoss, self).__init__()
        self.temp = temp
        self.alpha = alpha
        self.beta = beta

    def forward(self, logit_s, logit_t):
        pred_student = F.softmax(logit_s / self.temp, dim=1) + 1e-9
        pred_teacher = F.softmax(logit_t / self.temp, dim=1) + 1e-9

        # 获取每个样本的主类（概率最大的类别索引）
        max_idx_t = torch.argmax(pred_teacher, dim=1, keepdim=True)

        # 计算主类概率的 KL 散度
        student_main = pred_student.gather(1, max_idx_t)
        teacher_main = pred_teacher.gather(1, max_idx_t)

        log_student_main = torch.log(student_main)
        mc_loss = F.kl_div(log_student_main, teacher_main, reduction='batchmean') \
                  * (self.temp**2)

        # 掩盖主类，仅保留非主类
        mask = torch.ones_like(pred_teacher).scatter_(1, max_idx_t, 0).bool()

        # 计算非主类的概率分布
        student_non_main = pred_student[mask].view(logit_s.size(0), -1)
        teacher_non_main = pred_teacher[mask].view(logit_s.size(0), -1)

        log_student_non_main = torch.log(student_non_main)
        nmc_loss = F.kl_div(log_student_non_main, teacher_non_main, reduction='batchmean') \
                   * (self.temp**2)

        # 检查 loss 是否为 NaN
        if torch.isnan(mc_loss):
            print("Warning: NaN loss encountered")

        return self.alpha * mc_loss + self.beta * nmc_loss

