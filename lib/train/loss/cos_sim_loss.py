import torch
import torch.nn.functional as F


def cosine_similarity_loss(anchor, negative, margin):
    # 计算向量之间的余弦相似度
    # similarity_pos = F.cosine_similarity(anchor, positive)
    similarity_neg = F.cosine_similarity(anchor, negative)

    # 计算余弦相似度损失
    loss = torch.clamp(similarity_neg - 0 + margin, min=0)

    return loss.mean()
