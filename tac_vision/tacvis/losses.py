import torch
import torch.nn.functional as F


def batch_contrastive_loss(b1, b2, temp):
    '''
    b1 is output features of 1 encoder, b2 is the other
    b1 and b2 are [B x D], B=batch size, D=feature dim size
    '''
    # step 1 compute dot prods between
    dot_prods = b1.mm(b2.t()) * torch.exp(torch.clip(temp, 0, 10))  # [B x B]
    # step 2 compute CE loss
    labels = torch.arange(b1.shape[0], device=b1.device)
    loss = (F.cross_entropy(dot_prods, labels) + F.cross_entropy(dot_prods.T, labels)) / 2
    return loss


def l2_norm_loss(b):
    '''
    penalizes l2 norm of each batch
    '''
    return torch.norm(b, dim=1).mean()


def rot_loss(pred, label):
    loss = F.cross_entropy(pred, label)
    return loss
