"""
Official soft-skeleton implementation from:
  Shit et al. 2021 — clDice: A Novel Topology-Preserving Loss Function
  for Tubular Structure Segmentation
  https://github.com/jocpae/clDice

Uses iterative min/max morphological pooling to compute a differentiable
skeleton. Gradient flow is preserved throughout (no numpy/skimage detour).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSkeletonize(nn.Module):
    """
    Differentiable soft skeletonization via iterative morphological operations.
    Supports 2-D (B, C, H, W) and 3-D (B, C, D, H, W) inputs.
    """

    def __init__(self, num_iter: int = 10):
        super().__init__()
        self.num_iter = num_iter

    # ------------------------------------------------------------------
    # Morphological primitives
    # ------------------------------------------------------------------
    def soft_erode(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 4:                          # 2-D: (B, C, H, W)
            p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
            return torch.min(p1, p2)
        elif img.dim() == 5:                        # 3-D: (B, C, D, H, W)
            p1 = -F.max_pool3d(-img, (3, 1, 1), stride=1, padding=(1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), stride=1, padding=(0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), stride=1, padding=(0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)
        else:
            raise ValueError(f"soft_erode expects 4-D or 5-D tensor, got {img.dim()}-D")

    def soft_dilate(self, img: torch.Tensor) -> torch.Tensor:
        if img.dim() == 4:
            return F.max_pool2d(img, (3, 3), stride=1, padding=1)
        elif img.dim() == 5:
            return F.max_pool3d(img, (3, 3, 3), stride=1, padding=1)
        else:
            raise ValueError(f"soft_dilate expects 4-D or 5-D tensor, got {img.dim()}-D")

    def soft_open(self, img: torch.Tensor) -> torch.Tensor:
        return self.soft_dilate(self.soft_erode(img))

    # ------------------------------------------------------------------
    # Skeletonization loop  (Algorithm 1 from the paper)
    # ------------------------------------------------------------------
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img = torch.clamp(img, 0.0, 1.0)
        skel = F.relu(img - self.soft_open(img))
        for _ in range(self.num_iter):
            img = self.soft_erode(img)
            delta = F.relu(img - self.soft_open(img))
            # Union via inclusion-exclusion (keeps values in [0, 1])
            skel = skel + F.relu(delta - skel * delta)
        return torch.clamp(skel, 0.0, 1.0)


# Convenience alias so callers can do:  from src.cldice_soft_skeleton import soft_skel
def soft_skel(img: torch.Tensor, iter_: int = 10) -> torch.Tensor:
    """Functional wrapper — skeletonize a (B, C, H, W) probability map."""
    return SoftSkeletonize(num_iter=iter_)(img)


# ------------------------------------------------------------------
# Gradient-flow smoke test (run this file directly to verify)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    print("=== clDice soft-skeleton gradient-flow test ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    x = torch.rand(2, 1, 64, 64, requires_grad=True, device=device)
    skel = soft_skel(x, iter_=3)
    loss = skel.sum()
    loss.backward()

    ok = x.grad is not None and not torch.isnan(x.grad).any()
    print(f"x.grad is not None : {x.grad is not None}")
    print(f"grad has NaN       : {torch.isnan(x.grad).any().item()}")
    print(f"Gradient flows OK  : {ok}")

    if not ok:
        print("FAIL — gradient did not flow!", file=sys.stderr)
        sys.exit(1)
    else:
        print("PASS")
