import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftSkeletonize(nn.Module):
    """
    Differentiable morphological soft skeletonization based on min-max morphological pooling.
    Paper: Shit et al. 2021 (clDice: A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation)
    """
    def __init__(self, num_iter=10):
        super().__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        if len(img.shape) != 4:
            raise ValueError("Expected 4D tensor (B, C, H, W)")
        p1 = -F.max_pool2d(-img, (3, 1), stride=1, padding=(1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), stride=1, padding=(0, 1))
        return torch.min(p1, p2)

    def soft_dilate(self, img):
        if len(img.shape) != 4:
            raise ValueError("Expected 4D tensor (B, C, H, W)")
        return F.max_pool2d(img, (3, 3), stride=1, padding=1)

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def forward(self, img):
        img = torch.clamp(img, 0.0, 1.0)
        skel = F.relu(img - self.soft_open(img))
        for _ in range(self.num_iter):
            img = self.soft_erode(img)
            skel_step = F.relu(img - self.soft_open(img))
            skel = skel + F.relu(skel_step - skel * skel_step)
        return torch.clamp(skel, 0.0, 1.0)

class SoftCLDiceLoss(nn.Module):
    """
    clDice Loss function (Shit et al. 2021)
    Calculates skeleton intersection over union between prediction and ground truth.
    """
    def __init__(self, num_iter=10, smooth=1.0):
        super().__init__()
        self.num_iter = num_iter
        self.smooth = smooth
        self.skeletonizer = SoftSkeletonize(num_iter=num_iter)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        # Skeletonize
        skel_pred = self.skeletonizer(probs)
        skel_true = self.skeletonizer(targets)

        # Tprec: fraction of predicted skeleton inside true segmentation
        tprec = (torch.sum(skel_pred * targets, dim=(1, 2, 3)) + self.smooth) / (
            torch.sum(skel_pred, dim=(1, 2, 3)) + self.smooth
        )
        # Tsens: fraction of true skeleton inside predicted segmentation
        tsens = (torch.sum(skel_true * probs, dim=(1, 2, 3)) + self.smooth) / (
            torch.sum(skel_true, dim=(1, 2, 3)) + self.smooth
        )

        cldice = (2.0 * tprec * tsens) / (tprec + tsens + 1e-7)
        return torch.mean(1.0 - cldice)

if __name__ == "__main__":
    loss_fn = SoftCLDiceLoss(num_iter=5)
    logits = torch.randn(2, 1, 128, 128, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 128, 128)).float()
    loss = loss_fn(logits, targets)
    loss.backward()
    print("clDice test passed! Loss:", loss.item())
