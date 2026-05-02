import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss("asymmetric_loss")
def asymmetric_loss(pred, true):
    if cfg.model.loss_fun != "asymmetric_loss":
        return None

    gamma_pos = cfg.otformer.finetune.asl_gamma_pos
    gamma_neg = cfg.otformer.finetune.asl_gamma_neg
    clip = cfg.otformer.finetune.asl_clip

    is_labeled = true == true
    pred_labeled = pred[is_labeled]
    true_labeled = true[is_labeled].float()

    if pred_labeled.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True), pred

    # ASL on probabilities converted from logits.
    p = torch.sigmoid(pred_labeled).clamp(min=1e-8, max=1 - 1e-8)
    p_m = torch.clamp(p - clip, min=0.0, max=1.0).clamp(min=1e-8, max=1 - 1e-8)

    pos_loss = -((1 - p) ** gamma_pos) * torch.log(p) * true_labeled
    neg_loss = -(p_m**gamma_neg) * torch.log(1 - p_m) * (1 - true_labeled)

    loss = (pos_loss + neg_loss).sum() / is_labeled.sum().clamp(min=1)
    return loss, torch.sigmoid(pred)


@register_loss("bce_with_logits_finetune")
def bce_with_logits_finetune(pred, true):
    if cfg.model.loss_fun != "bce_with_logits_finetune":
        return None

    is_labeled = true == true
    if not is_labeled.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True), pred

    if pred.dim() < is_labeled.dim():
        pred = pred.unsqueeze(-1).expand_as(true)
    pred_labeled = pred[is_labeled]
    true_labeled = true[is_labeled].float()

    loss = nn.BCEWithLogitsLoss()(pred_labeled, true_labeled)
    return loss, torch.sigmoid(pred)


@register_loss("mse_finetune")
def mse_finetune(pred, true):
    if cfg.model.loss_fun != "mse_finetune":
        return None

    is_labeled = true == true
    if not is_labeled.any():
        return torch.tensor(0.0, device=pred.device, requires_grad=True), pred

    pred_labeled = pred[is_labeled]
    true_labeled = true[is_labeled].float()

    loss = nn.MSELoss()(pred_labeled, true_labeled)
    return loss, pred
