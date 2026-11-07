import os
import json
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from src.unet_model import UNetResNet34
from src.unet_dataset import SolarFilamentUNetDataset
from src.unet_loss import UNetCompoundLoss
from src.unet_validate import validate_unet

def train_unet_pipeline(
    epochs: int = 60,
    batch_size: int = 4,
    enc_lr: float = 1e-4,
    dec_lr: float = 3e-4,
    weight_decay: float = 0.0005,
    val_interval: int = 5,
    output_dir: str = "models",
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    best_model_path = os.path.join(output_dir, "unet_best.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TRAINING U-NET PIPELINE (PLAN A) ===")
    print(f"Device: {device} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"Encoder LR: {enc_lr} | Decoder/Heads LR: {dec_lr}")

    json_p = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    imgs_dir = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images"

    with open(json_p, "r") as f:
        coco = json.load(f)

    all_img_ids = [img["id"] for img in coco["images"]]
    # Deterministic split 90% train / 10% val (consistent with prior split)
    np.random.seed(42)
    shuffled_ids = np.random.permutation(all_img_ids)
    n_val = max(20, int(len(all_img_ids) * 0.10))
    val_ids = list(shuffled_ids[:n_val])
    train_ids = list(shuffled_ids[n_val:])

    print(f"Train images: {len(train_ids)} | Validation images: {len(val_ids)}")

    train_ds = SolarFilamentUNetDataset(
        json_path=json_p,
        images_dir=imgs_dir,
        image_ids=train_ids,
        crop_size=512,
        is_train=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    model = UNetResNet34(pretrained=True).to(device)

    # Differential learning rates
    encoder_params = list(model.conv1.parameters()) + list(model.bn1.parameters()) + \
                     list(model.layer1.parameters()) + list(model.layer2.parameters()) + \
                     list(model.layer3.parameters()) + list(model.layer4.parameters())
    decoder_params = list(model.up4.parameters()) + list(model.up3.parameters()) + \
                     list(model.up2.parameters()) + list(model.up1.parameters()) + \
                     list(model.final_conv.parameters()) + list(model.spine_conv.parameters()) + \
                     list(model.aux1.parameters()) + list(model.aux2.parameters())

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": enc_lr},
        {"params": decoder_params, "lr": dec_lr},
    ], weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    criterion = UNetCompoundLoss()
    scaler = GradScaler()

    best_pq = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        epoch_fg_fracs = []
        start_t = time.time()

        for step, (imgs, mains, spines) in enumerate(train_loader):
            imgs = imgs.to(device)
            mains = mains.to(device)
            spines = spines.to(device)

            optimizer.zero_grad()

            with autocast():
                preds = model(imgs)
                loss_dict = criterion(preds, mains, spines)
                total_loss = loss_dict["total_loss"]

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(total_loss.item())
            epoch_fg_fracs.append(loss_dict["fg_frac"])

        scheduler.step()
        elapsed = time.time() - start_t
        mean_loss = float(np.mean(epoch_losses))
        mean_fg = float(np.mean(epoch_fg_fracs))
        curr_lr = optimizer.param_groups[1]["lr"]

        print(f"Epoch [{epoch:02d}/{epochs}] | Loss: {mean_loss:.4f} | FG OHEM Frac: {mean_fg:.3f} | LR: {curr_lr:.2e} | Time: {elapsed:.1f}s")

        # Validation every val_interval epochs or on final epoch
        if epoch % val_interval == 0 or epoch == epochs:
            print(f"--- Running Strict PQ Validation on 20 held-out images ---")
            val_res = validate_unet(
                model=model,
                json_path=json_p,
                images_dir=imgs_dir,
                val_image_ids=val_ids,
                device=device,
                thresh=0.50,
                limit=20,
            )
            val_pq = val_res["mean_pq"]
            print(f"Val PQ: {val_pq:.4f} (TP={val_res['total_tp']}, FP={val_res['total_fp']}, FN={val_res['total_fn']})")

            if val_pq > best_pq:
                best_pq = val_pq
                torch.save(model.state_dict(), best_model_path)
                print(f"--> Saved new best checkpoint to {best_model_path} (PQ: {best_pq:.4f})")

    print(f"\nTraining Complete. Best Validation Strict PQ: {best_pq:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--enc_lr", type=float, default=1e-4)
    parser.add_argument("--dec_lr", type=float, default=3e-4)
    parser.add_argument("--val_interval", type=int, default=5)
    args = parser.parse_args()

    train_unet_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        enc_lr=args.enc_lr,
        dec_lr=args.dec_lr,
        val_interval=args.val_interval,
    )
