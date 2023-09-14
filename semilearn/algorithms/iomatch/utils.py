import torch
from torch import nn
import torch.nn.functional as F


# Reference: https://github.com/VisionLearningGroup/OP_Match/blob/main/utils/misc.py
def mb_sup_loss(logits_ova, label):
    batch_size = logits_ova.size(0)
    logits_ova = logits_ova.view(batch_size, 2, -1)
    num_classes = logits_ova.size(2)
    probs_ova = F.softmax(logits_ova, 1)
    label_s_sp = torch.zeros((batch_size, num_classes)).long().to(label.device)
    label_range = torch.arange(0, batch_size).long().to(label.device)
    label_s_sp[label_range[label < num_classes], label[label < num_classes]] = 1
    label_sp_neg = 1 - label_s_sp
    open_loss = torch.mean(torch.sum(-torch.log(probs_ova[:, 1, :] + 1e-8) * label_s_sp, 1))
    open_loss_neg = torch.mean(torch.max(-torch.log(probs_ova[:, 0, :] + 1e-8) * label_sp_neg, 1)[0])
    l_ova_sup = open_loss_neg + open_loss
    return l_ova_sup
