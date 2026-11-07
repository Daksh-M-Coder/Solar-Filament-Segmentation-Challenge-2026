import torch
import torch.nn as nn
import torch.nn.functional as F

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

class ForegroundAwareOHEMLoss(nn.Module):
    def __init__(self, bg_ratio=0.30):
        super().__init__()
        self.bg_ratio = bg_ratio
        self.dice_fn = SoftDiceLoss()

    def forward(self, logits, targets):
        # logits: [B, 1, H, W], targets: [B, 1, H, W]
        bce_all = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        
        flat_logits = logits.view(-1)
        flat_targets = targets.view(-1)
        flat_bce = bce_all.view(-1)
        
        fg_mask = flat_targets > 0.5
        bg_mask = ~fg_mask
        
        fg_bce = flat_bce[fg_mask]
        bg_bce = flat_bce[bg_mask]
        
        # Keep top bg_ratio hardest background pixels
        if bg_bce.numel() > 0:
            k = max(1, int(self.bg_ratio * bg_bce.numel()))
            topk_bg_bce, _ = torch.topk(bg_bce, k=k, largest=True)
        else:
            topk_bg_bce = bg_bce

        # Combine all foreground + top hardest background
        combined_bce = torch.cat([fg_bce, topk_bg_bce]) if fg_bce.numel() > 0 else topk_bg_bce
        ohem_bce = combined_bce.mean() if combined_bce.numel() > 0 else flat_bce.mean()
        
        dice = self.dice_fn(logits, targets)
        loss = 0.5 * ohem_bce + 0.5 * dice
        
        fg_fraction = fg_bce.numel() / max(1, combined_bce.numel())
        return loss, fg_fraction

class UNetCompoundLoss(nn.Module):
    def __init__(self, main_ohem_bg_ratio=0.30):
        super().__init__()
        self.main_loss_fn = ForegroundAwareOHEMLoss(bg_ratio=main_ohem_bg_ratio)
        self.dice_fn = SoftDiceLoss()

    def forward(self, preds, gt_main, gt_spine):
        # 1. Main Loss: 0.5 * OHEM-BCE + 0.5 * Dice
        main_loss, fg_frac = self.main_loss_fn(preds["main"], gt_main)
        
        # 2. Secondary Loss (Spine): BCE + Dice, no OHEM (spine is sparse)
        spine_bce = F.binary_cross_entropy_with_logits(preds["spine"], gt_spine)
        spine_dice = self.dice_fn(preds["spine"], gt_spine)
        spine_loss = 0.5 * spine_bce + 0.5 * spine_dice
        
        total_loss = main_loss + 0.2 * spine_loss
        
        # 3. Deep supervision on aux1 and aux2 (if in training mode)
        aux1_loss = torch.tensor(0.0, device=gt_main.device)
        aux2_loss = torch.tensor(0.0, device=gt_main.device)
        
        if "aux1" in preds and preds["aux1"] is not None:
            aux1_bce = F.binary_cross_entropy_with_logits(preds["aux1"], gt_main)
            aux1_dice = self.dice_fn(preds["aux1"], gt_main)
            aux1_loss = 0.5 * aux1_bce + 0.5 * aux1_dice
            total_loss = total_loss + 0.3 * aux1_loss

        if "aux2" in preds and preds["aux2"] is not None:
            aux2_bce = F.binary_cross_entropy_with_logits(preds["aux2"], gt_main)
            aux2_dice = self.dice_fn(preds["aux2"], gt_main)
            aux2_loss = 0.5 * aux2_bce + 0.5 * aux2_dice
            total_loss = total_loss + 0.3 * aux2_loss

        return {
            "total_loss": total_loss,
            "main_loss": main_loss,
            "spine_loss": spine_loss,
            "aux1_loss": aux1_loss,
            "aux2_loss": aux2_loss,
            "fg_frac": fg_frac,
        }

if __name__ == "__main__":
    crit = UNetCompoundLoss()
    preds = {
        "main": torch.randn(4, 1, 512, 512, requires_grad=True),
        "spine": torch.randn(4, 1, 512, 512, requires_grad=True),
        "aux1": torch.randn(4, 1, 512, 512, requires_grad=True),
        "aux2": torch.randn(4, 1, 512, 512, requires_grad=True),
    }
    gt_main = (torch.rand(4, 1, 512, 512) > 0.9).float()
    gt_spine = (torch.rand(4, 1, 512, 512) > 0.98).float()
    
    losses = crit(preds, gt_main, gt_spine)
    losses["total_loss"].backward()
    print("Loss verification successful!")
    for k, v in losses.items():
        val = v.item() if isinstance(v, torch.Tensor) else v
        print(f"  {k}: {val:.4f}")
