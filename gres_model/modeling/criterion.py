import logging

import torch
import torch.nn.functional as F
from torch import nn

from ..utils.misc import nested_tensor_from_tensor_list

def sigmoid_focal_loss(inputs, targets, num_masks=None, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum()

def counting_loss_v0(
        inputs: torch.Tensor,
        targets: torch.Tensor,
):  
    num_classes = 12
    loss = F.smooth_l1_loss(inputs, targets, reduction="none")
    w :List[float] = [0.01, 0.5, 15.0, 0.65, 2.25, 13.0, 4.8, 0.55, 0.64, 1.73, 3.236, 100.0]
    weights = torch.tensor(w).to(inputs)
    loss = torch.mean(loss * weights)
    return loss

def refer_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: torch.Tensor):

    loss = F.cross_entropy(inputs, targets, weight=weight)

    return loss

refer_focal_loss_jit = torch.jit.script(
    sigmoid_focal_loss
)

refer_ce_loss_jit = torch.jit.script(
    refer_ce_loss
)  # type: torch.jit.ScriptModule

count_loss_jit = torch.jit.script(
    counting_loss_v0
)


class ReferringCriterion(nn.Module):
    def __init__(self, weight_dict, losses):
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses

    def get_loss(self, loss, outputs, targets):
        loss_map = {
            'masks': self.loss_masks_refer,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets)

    def loss_masks_refer(self, outputs, targets, aux_loss=False):
        src_masks = outputs["pred_masks"]
        src_nt_label = outputs["nt_label"]
        src_count = outputs["pred_count"]
        masks = [t["gt_mask_merged"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_nts = torch.stack([t["empty"] for t in targets])
        target_count = torch.stack([t["labels"] for t in targets])
        if aux_loss == False:
            h, w = target_masks.shape[-2:]
            src_masks = F.interpolate(src_masks, (h, w), mode='bilinear', align_corners=False)
        else:
            h, w = src_masks.shape[-2:]
            target_masks = F.interpolate(target_masks, (h, w), mode='bilinear', align_corners=False)

        weight = torch.FloatTensor([0.8, 1.2]).to(src_masks)
        nt_weight = torch.FloatTensor([0.9, 1.1]).to(src_masks)

        loss_mask = refer_ce_loss_jit(src_masks, target_masks.squeeze(1).long(), weight) * 2
        loss_label = refer_ce_loss_jit(src_nt_label, target_nts, nt_weight) * 0.5
        loss_count = count_loss_jit(src_count, target_count) * 0.1

        losses = {
            "loss_mask": loss_mask,
            "loss_label": loss_label,
            "loss_count": loss_count,
        }

        del src_masks
        del target_masks
        return losses

    def forward(self, outputs, targets):
        aux_outputs_list = outputs.pop('aux_outputs', None)
        losses = self.loss_masks_refer(outputs, targets) #compute the last decoder loss
        
        if aux_outputs_list is not None:
            for i, aux_outputs in enumerate(aux_outputs_list):
                losses_dict = self.loss_masks_refer(aux_outputs, targets, aux_loss=True)
                losses_dict = {k + f'_{i}': v for k, v in losses_dict.items()}
                losses.update(losses_dict)
        return losses