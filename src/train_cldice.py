import os
import json
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pycocotools import mask as mask_utils

from src.unet_model import UNetResNet34
from src.unet_dataset import SolarFilamentUNetDataset
from src.losses import UNetCompoundLoss
from src.unet_inference import predict_single_image
from src.unet_validate import evaluate_strict_pq

def evaluate_val115(model, val_ids, images_dict, anns_dict, images_dir, device, postproc_kwargs):
    model.eval()
    pqs, tps, fps, fns = [], [], [], []
    for img_id in val_ids:
        info = images_dict[img_id]
        img_p = os.path.join(images_dir, info["file_name"])
        h, w = info["height"], info["width"]

        gt_instances = []
        for ann in anns_dict.get(img_id, []):
            seg = ann.get("segmentation", [])
            if isinstance(seg, list):
                rles = mask_utils.frPyObjects(seg, h, w)
                m = mask_utils.decode(mask_utils.merge(rles))
                gt_instances.append(m)
            elif isinstance(seg, dict):
                m = mask_utils.decode(seg)
                gt_instances.append(m)

        insts, _, _, _ = predict_single_image(
            model=model,
            img_path=img_p,
            device=device,
            thresh=0.45,
            use_tta=True,
            tta_mode="prob_avg",
            postproc_kwargs=postproc_kwargs,
        )
        res = evaluate_strict_pq(insts, gt_instances)
        pqs.append(res["pq"])
        tps.append(res["tp"])
        fps.append(res["fp"])
        fns.append(res["fn"])

    return float(np.mean(pqs)), int(np.sum(tps)), int(np.sum(fps)), int(np.sum(fns))

def main():
    parser = argparse.ArgumentParser(description="Task 4: Separate clDice Training Script")
    parser.add_argument("--use_cldice", action="store_true", default=False, help="Enable clDice loss")
    parser.add_argument("--cldice_weight", type=float, default=0.3, help="Weight for clDice loss")
    parser.add_argument("--cldice_alpha", type=float, default=0.8, help="Dice weight in clDice loss combination")
    parser.add_argument("--cldice_beta", type=float, default=0.2, help="clDice weight in clDice loss combination")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Global learning rate (if specified, overrides enc_lr and dec_lr)")
    parser.add_argument("--enc_lr", type=float, default=1e-5, help="Encoder learning rate")
    parser.add_argument("--dec_lr", type=float, default=5e-5, help="Decoder/heads learning rate")
    parser.add_argument("--val_interval", type=int, default=5, help="Validation interval (epochs)")
    parser.add_argument("--start_checkpoint", type=str, default="models/unet_best.pth", help="Starting weights")
    parser.add_argument("--save_path", type=str, default="models/unet_cldice_v2_best.pth", help="Best checkpoint path (never overwrites unet_best.pth)")
    parser.add_argument("--checkpoint_dir", type=str, default="models/cldice_v2_checkpoints", help="Directory to save per-epoch checkpoints")
    args = parser.parse_args()

    # If cldice_beta > 0 or explicitly requested, enable clDice
    if args.cldice_beta > 0:
        args.use_cldice = True

    # If --lr is provided, set enc_lr and dec_lr based on standard differential ratio
    if args.lr is not None:
        args.enc_lr = args.lr * 0.2
        args.dec_lr = args.lr


    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============================================================")
    print("=== TASK 4: STAGE 3 cLDICE RETRAINING ===")
    print("============================================================")
    print(f"Device           : {device}")
    print(f"Use clDice       : {args.use_cldice}")
    print(f"clDice Weight    : {args.cldice_weight}")
    print(f"Epochs           : {args.epochs}")
    print(f"Batch Size       : {args.batch_size}")
    print(f"Encoder LR       : {args.enc_lr} | Decoder LR: {args.dec_lr}")
    print(f"Val Interval     : {args.val_interval} epochs (on 115-pool)")
    print(f"Start Checkpoint : {args.start_checkpoint}")
    print(f"Best Save Path   : {args.save_path} (never overwrites unet_best.pth)")
    print(f"Checkpoint Dir   : {args.checkpoint_dir} (per-epoch files saved here)")
    print("============================================================", flush=True)

    json_p = "data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    imgs_dir = "data/MAGFiLO_1.0_Kaggle_2026/train/train_images"

    with open(json_p, "r") as f:
        coco = json.load(f)

    images_dict = {img["id"]: img for img in coco["images"]}
    anns_dict = {}
    for ann in coco["annotations"]:
        anns_dict.setdefault(ann["image_id"], []).append(ann)

    all_img_ids = [img["id"] for img in coco["images"]]
    np.random.seed(42)
    shuffled_ids = np.random.permutation(all_img_ids)
    n_val = max(20, int(len(all_img_ids) * 0.10))
    val_ids = list(shuffled_ids[:n_val])
    train_ids = list(shuffled_ids[n_val:])

    print(f"Train images: {len(train_ids)} | Validation images: {len(val_ids)}", flush=True)

    train_ds = SolarFilamentUNetDataset(
        json_path=json_p,
        images_dir=imgs_dir,
        image_ids=train_ids,
        crop_size=512,
        is_train=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    model = UNetResNet34(pretrained=False).to(device)
    if os.path.exists(args.start_checkpoint):
        print(f"Loading weights from start checkpoint: {args.start_checkpoint}", flush=True)
        model.load_state_dict(torch.load(args.start_checkpoint, map_location=device))
    else:
        print(f"Warning: {args.start_checkpoint} not found. Starting from scratch.", flush=True)

    encoder_params = list(model.conv1.parameters()) + list(model.bn1.parameters()) + \
                     list(model.layer1.parameters()) + list(model.layer2.parameters()) + \
                     list(model.layer3.parameters()) + list(model.layer4.parameters())
    decoder_params = list(model.up4.parameters()) + list(model.up3.parameters()) + \
                     list(model.up2.parameters()) + list(model.up1.parameters()) + \
                     list(model.final_conv.parameters()) + list(model.spine_conv.parameters()) + \
                     list(model.aux1.parameters()) + list(model.aux2.parameters())

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.enc_lr},
        {"params": decoder_params, "lr": args.dec_lr},
    ], weight_decay=0.0005)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = UNetCompoundLoss(
        use_cldice=args.use_cldice,
        cldice_weight=args.cldice_weight,
        cldice_alpha=args.cldice_alpha,
        cldice_beta=args.cldice_beta,
    )
    scaler = GradScaler()

    postproc_kwargs = {
        "peak_min_dist": 55,
        "skel_prune_len": 15,
        "morph_kernel_size": 3,
        "dist_transform_type": "DIST_L1",
        "min_area": 300,
    }

    best_val_pq = 0.4154
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, bces, dices, cldices = [], [], [], []
        start_t = time.time()

        for step, (imgs, mains, spines) in enumerate(train_loader):
            imgs = imgs.to(device)
            mains = mains.to(device)
            spines = spines.to(device)

            optimizer.zero_grad()
            with autocast():
                preds = model(imgs)
                loss_dict = criterion(preds, mains, spines)
                loss = loss_dict["total_loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())
            bces.append(loss_dict["bce"].item())
            dices.append(loss_dict["dice"].item())
            cld_val = loss_dict["cldice"].item() if isinstance(loss_dict["cldice"], torch.Tensor) else loss_dict["cldice"]
            cldices.append(cld_val)

        scheduler.step()
        epoch_dur = time.time() - start_t
        mean_tot = np.mean(losses)
        mean_bce = np.mean(bces)
        mean_dice = np.mean(dices)
        mean_cld = np.mean(cldices)
        lr_enc = optimizer.param_groups[0]["lr"]
        lr_dec = optimizer.param_groups[1]["lr"]

        if args.use_cldice:
            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} [{epoch_dur:.1f}s] "
                f"Total Loss: {mean_tot:.4f} | BCE: {mean_bce:.4f} | Dice: {mean_dice:.4f} | clDice: {mean_cld:.4f} | "
                f"LR: [enc={lr_enc:.2e}, dec={lr_dec:.2e}]",
                flush=True,
            )
        else:
            print(
                f"Epoch {epoch:02d}/{args.epochs:02d} [{epoch_dur:.1f}s] "
                f"Total Loss: {mean_tot:.4f} | BCE: {mean_bce:.4f} | Dice: {mean_dice:.4f} | "
                f"LR: [enc={lr_enc:.2e}, dec={lr_dec:.2e}]",
                flush=True,
            )

        # In 1-epoch smoke tests or final epoch, always save checkpoint
        if epoch == 1 and args.epochs == 1:
            torch.save(model.state_dict(), args.save_path)
            print(f"  --> Smoke test checkpoint saved to: {args.save_path}", flush=True)

        # Validation on 115-pool every val_interval epochs
        if (epoch % args.val_interval == 0 and args.epochs > 1) or (epoch == args.epochs and args.epochs > 1):
            # Always save epoch checkpoint regardless of PQ
            epoch_ckpt = os.path.join(args.checkpoint_dir, f"ep{epoch:03d}.pth")
            torch.save(model.state_dict(), epoch_ckpt)
            print(f"  --> Epoch checkpoint saved: {epoch_ckpt}", flush=True)

            print(f"\n--- Running Validation on 115-pool (Epoch {epoch}) ---", flush=True)
            val_pq, val_tp, val_fp, val_fn = evaluate_val115(
                model=model,
                val_ids=val_ids,
                images_dict=images_dict,
                anns_dict=anns_dict,
                images_dir=imgs_dir,
                device=device,
                postproc_kwargs=postproc_kwargs,
            )
            print(f"VAL Epoch {epoch:02d}/{args.epochs:02d}: Strict PQ = {val_pq:.4f} | TP={val_tp} FP={val_fp} FN={val_fn}", flush=True)
            if val_pq > best_val_pq:
                delta = val_pq - best_val_pq
                best_val_pq = val_pq
                best_epoch = epoch
                torch.save(model.state_dict(), args.save_path)
                print(f"  --> NEW BEST VAL PQ! Delta: +{delta:.4f}. Saved to {args.save_path}", flush=True)
            else:
                print(f"  --> Best remains {best_val_pq:.4f} at epoch {best_epoch}", flush=True)

    print("\nTraining completed.")
    print(f"Final checkpoint location: {args.save_path}")

if __name__ == "__main__":
    main()
