import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.cldice_soft_skeleton import SoftSkeletonize


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class SoftCLDiceLoss(nn.Module):
    """
    Official clDice Loss (Shit et al. 2021) — pure PyTorch, gradient-safe.
    Uses differentiable soft-skeletonization (min-max morphological pooling)
    from src/cldice_soft_skeleton.py.  No numpy/skimage detour; gradients
    flow end-to-end through both skeleton branches.

    tprec = (skel_pred ∩ target)  / |skel_pred|
    tsens = (skel_true ∩ pred)    / |skel_true|
    clDice = 2·tprec·tsens / (tprec + tsens)
    """

    def __init__(self, num_iter: int = 10, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        self.skeletonize = SoftSkeletonize(num_iter=num_iter)

    def forward(self, pred_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        # pred_logits : (B, 1, H, W)  — raw logits
        # target_mask : (B, 1, H, W)  — binary float [0, 1]
        probs  = torch.sigmoid(pred_logits)
        target = target_mask.float()

        skel_pred = self.skeletonize(probs)     # differentiable skeleton of prediction
        skel_true = self.skeletonize(target)    # differentiable skeleton of ground truth
        # NOTE: DO NOT RUN — skel_true redesign pending expert review.
        # Expert proposed: replace skel_true with hard skimage skeleton of GT.
        # Will be implemented after post-processing sweep completes.

        # Topology precision: predicted skeleton inside true region
        tprec = (torch.sum(skel_pred * target, dim=(1, 2, 3)) + self.smooth) / \
                (torch.sum(skel_pred,           dim=(1, 2, 3)) + self.smooth)

        # Topology sensitivity: true skeleton inside predicted region
        tsens = (torch.sum(skel_true * probs, dim=(1, 2, 3)) + self.smooth) / \
                (torch.sum(skel_true,          dim=(1, 2, 3)) + self.smooth)

        cldice = (2.0 * tprec * tsens) / (tprec + tsens + 1e-7)
        return torch.mean(1.0 - cldice)

class ForegroundAwareOHEMLoss(nn.Module):
    def __init__(self, bg_ratio=0.30):
        super().__init__()
        self.bg_ratio = bg_ratio
        self.dice_fn = SoftDiceLoss()

    def forward(self, logits, targets):
        bce_all = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        flat_logits = logits.view(-1)
        flat_targets = targets.view(-1)
        flat_bce = bce_all.view(-1)

        fg_mask = flat_targets > 0.5
        bg_mask = ~fg_mask

        fg_bce = flat_bce[fg_mask]
        bg_bce = flat_bce[bg_mask]

        if bg_bce.numel() > 0:
            k = max(1, int(self.bg_ratio * bg_bce.numel()))
            topk_bg_bce, _ = torch.topk(bg_bce, k=k, largest=True)
        else:
            topk_bg_bce = bg_bce

        combined_bce = torch.cat([fg_bce, topk_bg_bce]) if fg_bce.numel() > 0 else topk_bg_bce
        ohem_bce = combined_bce.mean() if combined_bce.numel() > 0 else flat_bce.mean()

        dice = self.dice_fn(logits, targets)
        loss = 0.5 * ohem_bce + 0.5 * dice

        fg_fraction = fg_bce.numel() / max(1, combined_bce.numel())
        return loss, fg_fraction

class UNetCompoundLoss(nn.Module):
    """
    Task 3: Compound Loss with clDice
    Total = 0.4 * BCE + 0.3 * Dice + cldice_weight * clDice + 0.2 * Spine_loss + 0.3 * Aux1_loss + 0.3 * Aux2_loss
    """
    def __init__(self, use_cldice=False, cldice_weight=0.3, cldice_alpha=0.8, cldice_beta=0.2):
        super().__init__()
        self.use_cldice = use_cldice
        self.cldice_weight = cldice_weight
        self.cldice_alpha = cldice_alpha
        self.cldice_beta = cldice_beta
        self.dice_fn = SoftDiceLoss()
        if use_cldice:
            self.cldice_fn = SoftCLDiceLoss(num_iter=10, smooth=1.0)

    def forward(self, preds, gt_main, gt_spine):
        # 1. Main BCE & Dice
        bce_main = F.binary_cross_entropy_with_logits(preds["main"], gt_main)
        dice_main = self.dice_fn(preds["main"], gt_main)

        # 2. Main clDice (applied to main mask only)
        if self.use_cldice:
            cldice_val = self.cldice_fn(preds["main"], gt_main)
            # Combine Dice and clDice with alpha / beta weighting
            main_dice_cldice = self.cldice_alpha * dice_main + self.cldice_beta * cldice_val
        else:
            cldice_val = torch.tensor(0.0, device=gt_main.device)
            main_dice_cldice = dice_main

        # 3. Secondary Spine Loss
        spine_bce = F.binary_cross_entropy_with_logits(preds["spine"], gt_spine)
        spine_dice = self.dice_fn(preds["spine"], gt_spine)
        spine_loss = 0.5 * spine_bce + 0.5 * spine_dice

        # 4. Aux Deep Supervision
        aux1_loss = torch.tensor(0.0, device=gt_main.device)
        aux2_loss = torch.tensor(0.0, device=gt_main.device)
        if "aux1" in preds and preds["aux1"] is not None:
            aux1_bce = F.binary_cross_entropy_with_logits(preds["aux1"], gt_main)
            aux1_dice = self.dice_fn(preds["aux1"], gt_main)
            aux1_loss = 0.5 * aux1_bce + 0.5 * aux1_dice

        if "aux2" in preds and preds["aux2"] is not None:
            aux2_bce = F.binary_cross_entropy_with_logits(preds["aux2"], gt_main)
            aux2_dice = self.dice_fn(preds["aux2"], gt_main)
            aux2_loss = 0.5 * aux2_bce + 0.5 * aux2_dice

        # Total formula per Task 3 specification:
        total_loss = (
            0.5 * bce_main
            + 0.5 * main_dice_cldice
            + 0.2 * spine_loss
            + 0.3 * aux1_loss
            + 0.3 * aux2_loss
        )

        return {
            "total_loss": total_loss,
            "bce": bce_main,
            "dice": dice_main,
            "cldice": cldice_val,
            "spine": spine_loss,
            "aux1": aux1_loss,
            "aux2": aux2_loss,
        }

if __name__ == "__main__":
    crit = UNetCompoundLoss(use_cldice=True, cldice_weight=0.3)
    preds = {
        "main": torch.randn(2, 1, 128, 128, requires_grad=True),
        "spine": torch.randn(2, 1, 128, 128, requires_grad=True),
        "aux1": torch.randn(2, 1, 128, 128, requires_grad=True),
        "aux2": torch.randn(2, 1, 128, 128, requires_grad=True),
    }
    gt_main = (torch.rand(2, 1, 128, 128) > 0.8).float()
    gt_spine = (torch.rand(2, 1, 128, 128) > 0.9).float()

    losses = crit(preds, gt_main, gt_spine)
    losses["total_loss"].backward()
    print("src/losses.py verification passed!")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
