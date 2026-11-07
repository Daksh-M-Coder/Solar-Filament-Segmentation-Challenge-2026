import os
import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from src.unet_postprocess import postprocess_binary_mask, resolve_overlaps

def encode_mask_to_rle(mask: np.ndarray) -> str:
    """
    Encodes binary mask to Fortran-contiguous Kaggle RLE string.
    """
    if np.sum(mask) == 0:
        return ""
    fortran_mask = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(fortran_mask)
    return rle["counts"].decode("utf-8")

def get_tile_4views_raw_probs(
    model: torch.nn.Module,
    crop: np.ndarray,
    device: torch.device,
    mean: float = 0.449,
    std: float = 0.226,
):
    """
    Computes and returns the 4 un-flipped raw sigmoid probability maps [H, W]:
      1. Identity view
      2. H-flip view (un-flipped)
      3. V-flip view (un-flipped)
      4. HV-flip view (un-flipped)
    Also computes raw probability map variance across the 4 views.
    """
    norm = (crop.astype(np.float32) / 255.0 - mean) / std
    t_crop = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        # View 1: Identity
        out1 = model(t_crop)
        prob1 = torch.sigmoid(out1["main"])[0, 0].cpu().numpy()

        # View 2: H-flip
        t_crop_h = torch.flip(t_crop, dims=[3])
        out2 = model(t_crop_h)
        prob2 = torch.sigmoid(out2["main"])[0, 0]
        prob2 = torch.flip(prob2, dims=[1]).cpu().numpy()

        # View 3: V-flip
        t_crop_v = torch.flip(t_crop, dims=[2])
        out3 = model(t_crop_v)
        prob3 = torch.sigmoid(out3["main"])[0, 0]
        prob3 = torch.flip(prob3, dims=[0]).cpu().numpy()

        # View 4: HV-flip
        t_crop_hv = torch.flip(t_crop, dims=[2, 3])
        out4 = model(t_crop_hv)
        prob4 = torch.flip(torch.sigmoid(out4["main"])[0, 0], dims=[0, 1]).cpu().numpy()

    stack_probs = np.stack([prob1, prob2, prob3, prob4], axis=0) # [4, H, W]
    # Raw probability variance per pixel across the 4 views
    var_map = np.var(stack_probs, axis=0)
    mean_tile_variance = float(np.mean(var_map))
    # Unstable tile diagnostic: mean raw variance > 0.05
    is_unstable = mean_tile_variance > 0.05

    return (prob1, prob2, prob3, prob4), is_unstable, mean_tile_variance

def get_tile_avg_prob(
    model: torch.nn.Module,
    crop: np.ndarray,
    device: torch.device,
    mean: float = 0.449,
    std: float = 0.226,
):
    """
    Returns the within-tile 4-view averaged RAW probability map (float32, [H, W]),
    without thresholding or post-processing. Used by the prob_avg_seam stitching path.
    """
    (p1, p2, p3, p4), is_unstable, variance = get_tile_4views_raw_probs(
        model=model, crop=crop, device=device, mean=mean, std=std
    )
    avg_prob = (p1 + p2 + p3 + p4) / 4.0
    return avg_prob, is_unstable, variance


def predict_single_tile_views(
    model: torch.nn.Module,
    crop: np.ndarray,
    device: torch.device,
    mean: float = 0.449,
    std: float = 0.226,
    thresh: float = 0.45,
    tta_mode: str = "mask_vote",
    postproc_kwargs: dict = None,
):
    """
    Processes a single 512x512 tile with TTA:
    - If tta_mode == 'mask_vote':
        Runs post-processing on each of the 4 views, merges via strict majority vote (>= 3 of 4).
    - If tta_mode == 'prob_avg':
        Averages the 4 un-flipped raw probability maps, thresholds at thresh, and runs post-processing ONCE.
    Both modes use the corrected raw probability variance diagnostic (mean variance > 0.05).
    NOTE: 'prob_avg_seam' is handled directly in predict_tiled_unet_masks (does not go through this function).
    """
    (p1, p2, p3, p4), is_unstable, variance = get_tile_4views_raw_probs(
        model=model, crop=crop, device=device, mean=mean, std=std
    )
    kwargs = postproc_kwargs or {}

    if tta_mode == "prob_avg":
        # Average the 4 probability maps into a single probability map
        avg_prob = (p1 + p2 + p3 + p4) / 4.0
        # Run post-processing ONCE on the averaged probability map
        insts = postprocess_binary_mask(avg_prob, thresh=thresh, **kwargs)
        tile_mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        for inst in insts:
            tile_mask = np.maximum(tile_mask, inst)
        return tile_mask, is_unstable, variance

    else:
        # mask_vote mode: post-process each view individually
        insts1 = postprocess_binary_mask(p1, thresh=thresh, **kwargs)
        insts2 = postprocess_binary_mask(p2, thresh=thresh, **kwargs)
        insts3 = postprocess_binary_mask(p3, thresh=thresh, **kwargs)
        insts4 = postprocess_binary_mask(p4, thresh=thresh, **kwargs)

        m1 = np.zeros(crop.shape[:2], dtype=np.uint8)
        for inst in insts1: m1 = np.maximum(m1, inst)
        m2 = np.zeros(crop.shape[:2], dtype=np.uint8)
        for inst in insts2: m2 = np.maximum(m2, inst)
        m3 = np.zeros(crop.shape[:2], dtype=np.uint8)
        for inst in insts3: m3 = np.maximum(m3, inst)
        m4 = np.zeros(crop.shape[:2], dtype=np.uint8)
        for inst in insts4: m4 = np.maximum(m4, inst)

        # Strict majority vote: >= 3 out of 4 agree
        sum_agreement = (m1 > 0).astype(np.uint8) + (m2 > 0).astype(np.uint8) + (m3 > 0).astype(np.uint8) + (m4 > 0).astype(np.uint8)
        tile_mask = (sum_agreement >= 3).astype(np.uint8)

        return tile_mask, is_unstable, variance

def predict_tiled_unet_masks(
    model: torch.nn.Module,
    image_np: np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    overlap: int = 64,
    mean: float = 0.449,
    std: float = 0.226,
    thresh: float = 0.45,
    use_tta: bool = False,
    tta_mode: str = "mask_vote",
    postproc_kwargs: dict = None,
):
    """
    Tiled inference pipeline supporting:
    - use_tta=True:
        tta_mode='prob_avg': tile-level 4-view raw probability averaging, thresholded & post-processed,
                             stitched via overlap majority voting.
        tta_mode='mask_vote': tile-level 4-view mask majority voting (>=3 of 4),
                              stitched via overlap majority voting.
    - use_tta=False: Baseline identity view single forward pass.
    """
    H, W = image_np.shape[:2]
    step = tile_size - overlap

    y_steps = list(range(0, H - tile_size + 1, step))
    if y_steps[-1] != H - tile_size:
        y_steps.append(H - tile_size)

    x_steps = list(range(0, W - tile_size + 1, step))
    if x_steps[-1] != W - tile_size:
        x_steps.append(W - tile_size)

    kwargs = postproc_kwargs or {}
    model.eval()

    if use_tta:
        total_tiles = len(y_steps) * len(x_steps)
        unstable_tiles_count = 0
        tile_variances = []

        if tta_mode == "prob_avg_seam":
            # ---------------------------------------------------------------
            # NEW: prob_avg_seam — accumulate raw probability maps across tiles,
            # divide by coverage, then threshold + post-process ONCE on the
            # full stitched probability map. No per-tile binarization.
            # ---------------------------------------------------------------
            prob_canvas = np.zeros((H, W), dtype=np.float64)
            coverage_canvas = np.zeros((H, W), dtype=np.float32)

            for y1 in y_steps:
                for x1 in x_steps:
                    y2, x2 = y1 + tile_size, x1 + tile_size
                    crop = image_np[y1:y2, x1:x2]

                    avg_prob, is_unstable, var = get_tile_avg_prob(
                        model=model, crop=crop, device=device, mean=mean, std=std
                    )
                    tile_variances.append(var)
                    if is_unstable:
                        unstable_tiles_count += 1

                    prob_canvas[y1:y2, x1:x2] += avg_prob
                    coverage_canvas[y1:y2, x1:x2] += 1.0

            # Stitch: average probability across overlapping tiles
            stitched_prob = (prob_canvas / np.maximum(coverage_canvas, 1e-5)).astype(np.float32)
            # Threshold + post-process ONCE on the full stitched probability map
            insts = postprocess_binary_mask(stitched_prob, thresh=thresh, **kwargs)
            stitched_mask = np.zeros((H, W), dtype=np.uint8)
            for inst in insts:
                stitched_mask = np.maximum(stitched_mask, inst)

        else:
            # ---------------------------------------------------------------
            # ORIGINAL: prob_avg / mask_vote — accumulate binarized per-tile
            # masks, majority-vote at seams. Kept for A/B comparison.
            # ---------------------------------------------------------------
            vote_canvas = np.zeros((H, W), dtype=np.float32)
            coverage_canvas = np.zeros((H, W), dtype=np.float32)

            for y1 in y_steps:
                for x1 in x_steps:
                    y2, x2 = y1 + tile_size, x1 + tile_size
                    crop = image_np[y1:y2, x1:x2]

                    tile_mask, is_unstable, var = predict_single_tile_views(
                        model=model,
                        crop=crop,
                        device=device,
                        mean=mean,
                        std=std,
                        thresh=thresh,
                        tta_mode=tta_mode,
                        postproc_kwargs=kwargs,
                    )
                    tile_variances.append(var)
                    if is_unstable:
                        unstable_tiles_count += 1

                    vote_canvas[y1:y2, x1:x2] += tile_mask
                    coverage_canvas[y1:y2, x1:x2] += 1.0

            # Majority vote across overlapping tiles: >= 50% of overlapping tiles vote positive
            stitched_mask = ((vote_canvas / np.maximum(coverage_canvas, 1.0)) >= 0.50).astype(np.uint8)

        mean_var = float(np.mean(tile_variances)) if tile_variances else 0.0
        return stitched_mask, unstable_tiles_count, total_tiles, mean_var

    else:
        # Non-TTA Mode (Baseline identity path)
        prob_canvas = np.zeros((H, W), dtype=np.float32)
        count_canvas = np.zeros((H, W), dtype=np.float32)

        with torch.no_grad():
            for y1 in y_steps:
                for x1 in x_steps:
                    y2, x2 = y1 + tile_size, x1 + tile_size
                    crop = image_np[y1:y2, x1:x2]
                    norm = (crop.astype(np.float32) / 255.0 - mean) / std
                    t_crop = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    prob = torch.sigmoid(model(t_crop)["main"])[0, 0].cpu().numpy()

                    prob_canvas[y1:y2, x1:x2] += prob
                    count_canvas[y1:y2, x1:x2] += 1.0

        avg_prob = prob_canvas / np.maximum(count_canvas, 1e-5)
        return avg_prob, 0, len(y_steps) * len(x_steps), 0.0

def predict_single_image(
    model: torch.nn.Module,
    img_path: str,
    device: torch.device,
    thresh: float = 0.45,
    tile_size: int = 512,
    overlap: int = 64,
    use_tta: bool = False,
    tta_mode: str = "mask_vote",
    postproc_kwargs: dict = None,
):
    """
    Main image inference function.
    Returns list of non-overlapping binary instance masks.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [], 0, 0, 0.0

    kwargs = postproc_kwargs or {}

    if use_tta:
        stitched_mask, unstable_cnt, tot_tiles, mean_var = predict_tiled_unet_masks(
            model=model,
            image_np=img,
            device=device,
            tile_size=tile_size,
            overlap=overlap,
            thresh=thresh,
            use_tta=True,
            tta_mode=tta_mode,
            postproc_kwargs=kwargs,
        )
        from skimage.measure import label, regionprops
        labeled = label(stitched_mask, connectivity=2)
        regions = regionprops(labeled)
        raw_instances = []
        H, W = img.shape
        min_area = kwargs.get("min_area", 300)
        for r in regions:
            if r.area >= min_area:
                inst_m = np.zeros((H, W), dtype=np.uint8)
                inst_m[r.bbox[0]:r.bbox[2], r.bbox[1]:r.bbox[3]] = r.image.astype(np.uint8)
                raw_instances.append(inst_m)

        instances = resolve_overlaps(raw_instances, min_area=min_area)
        return instances, unstable_cnt, tot_tiles, mean_var

    else:
        avg_prob, _, _, _ = predict_tiled_unet_masks(
            model=model,
            image_np=img,
            device=device,
            tile_size=tile_size,
            overlap=overlap,
            thresh=thresh,
            use_tta=False,
            tta_mode="mask_vote",
            postproc_kwargs=kwargs,
        )
        instances = postprocess_binary_mask(avg_prob, thresh=thresh, **kwargs)
        return instances, 0, 25, 0.0
