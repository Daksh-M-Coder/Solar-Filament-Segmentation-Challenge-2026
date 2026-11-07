import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

def prune_skeleton(skel: np.ndarray, min_branch_len: int = 10) -> np.ndarray:
    """
    Prunes small skeleton spurs/branches shorter than min_branch_len.
    """
    skel_out = skel.copy().astype(np.uint8)
    for _ in range(min_branch_len):
        kernel = np.array([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]], dtype=np.uint8)
        filtered = cv2.filter2D(skel_out, -1, kernel)
        endpoints = (filtered == 11) & (skel_out == 1)
        if not np.any(endpoints):
            break
        skel_out[endpoints] = 0
    return skel_out

def count_branch_points(skel: np.ndarray) -> int:
    """
    Count pixels with 3 or more 8-connected neighbors in skeleton.
    """
    if np.sum(skel) == 0:
        return 0
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)
    filtered = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
    branch_points = (filtered >= 13) & (skel == 1)
    return int(np.sum(branch_points))

def resolve_overlaps(instances: list, min_area: int = 300) -> list:
    """
    Resolves instance mask overlaps greedily.
    Sorts instances by area descending, allocates pixels greedily to larger instances,
    and subtracts any pixel already claimed from subsequent instances.
    Filters out any instance whose remaining area is below min_area.
    """
    if len(instances) <= 1:
        return [inst for inst in instances if np.sum(inst > 0) >= min_area]

    # Calculate areas and sort descending
    inst_areas = [int(np.sum(inst > 0)) for inst in instances]
    sorted_indices = np.argsort(inst_areas)[::-1]

    H, W = instances[0].shape
    claimed_canvas = np.zeros((H, W), dtype=bool)
    resolved_instances = []

    for idx in sorted_indices:
        mask = (instances[idx] > 0)
        # Remove pixels already claimed by larger instances
        unclaimed_mask = mask & (~claimed_canvas)
        
        area = int(np.sum(unclaimed_mask))
        if area >= min_area:
            out_mask = unclaimed_mask.astype(np.uint8)
            resolved_instances.append(out_mask)
            # Update claimed pixels
            claimed_canvas |= unclaimed_mask

    return resolved_instances

def binary_to_instances(
    prob_map: np.ndarray,
    thresh: float = 0.50,
    min_area: int = 300,
    peak_min_dist: int = 20,
    morph_kernel_size: int = 5,
    dist_transform_type: str = "DIST_L2",
    skel_prune_len: int = 10,
    use_tip_recovery_b: bool = False,
) -> list:
    """
    Task 5: Gated Watershed with Pruning and Morphological Closing + Overlap Resolution.
    Returns list of non-overlapping binary instance masks at original resolution (H, W).
    """
    H, W = prob_map.shape
    bin_mask = (prob_map >= thresh).astype(np.uint8)

    # Step 1: Cleanup - Morphological open then close (5x5 ellipse)
    k_size = morph_kernel_size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

    if np.sum(bin_mask) == 0:
        return []

    # Step 2: Connected Components (8-connectivity)
    labeled_arr = label(bin_mask, connectivity=2)
    regions = regionprops(labeled_arr)

    instances = []

    # Step 3 & 4: Per-Component Processing
    for r in regions:
        # Area filter
        if r.area < min_area:
            continue

        min_row, min_col, max_row, max_col = r.bbox
        comp_mask = r.image.astype(np.uint8)

        # Skeletonize component
        skel = skeletonize(comp_mask > 0).astype(np.uint8)

        # Prune small branches
        if skel_prune_len > 0:
            skel_pruned = prune_skeleton(skel, min_branch_len=skel_prune_len)
        else:
            skel_pruned = skel

        # Count branch points
        n_branches = count_branch_points(skel_pruned)

        if n_branches == 0:
            # Branch points == 0: Keep as single instance (No watershed)
            full_inst = np.zeros((H, W), dtype=np.uint8)
            full_inst[min_row:max_row, min_col:max_col] = comp_mask
            instances.append(full_inst)
        else:
            # Branch points > 0: Proceed to watershed
            dt_flag = cv2.DIST_L2 if dist_transform_type == "DIST_L2" else cv2.DIST_L1
            dist = cv2.distanceTransform(comp_mask, dt_flag, 3)

            coords = peak_local_max(dist, min_distance=peak_min_dist, labels=comp_mask)
            if len(coords) <= 1:
                # Only 1 peak found, keep single instance
                full_inst = np.zeros((H, W), dtype=np.uint8)
                full_inst[min_row:max_row, min_col:max_col] = comp_mask
                instances.append(full_inst)
            else:
                # Watershed segmentation
                markers = np.zeros(comp_mask.shape, dtype=np.int32)
                for idx, pt in enumerate(coords, start=1):
                    markers[pt[0], pt[1]] = idx

                ws_labels = watershed(-dist, markers, mask=(comp_mask > 0))
                for seg_id in range(1, len(coords) + 1):
                    sub_m = (ws_labels == seg_id).astype(np.uint8)
                    if np.sum(sub_m) >= min_area:
                        full_inst = np.zeros((H, W), dtype=np.uint8)
                        full_inst[min_row:max_row, min_col:max_col] = sub_m
                        instances.append(full_inst)

    # Step 5: Tip Recovery (Option A: 3x3 morphological close on every instance)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_instances = []
    for inst in instances:
        refined = cv2.morphologyEx(inst, cv2.MORPH_CLOSE, close_kernel)
        closed_instances.append(refined)

    # Step 6: Greedy Overlap Resolution
    resolved_instances = resolve_overlaps(closed_instances, min_area=min_area)

    return resolved_instances

# Alias postprocess_binary_mask to binary_to_instances for backwards compatibility
postprocess_binary_mask = binary_to_instances

if __name__ == "__main__":
    dummy_prob = np.zeros((2048, 2048), dtype=np.float32)
    cv2.line(dummy_prob, (300, 300), (800, 400), 0.9, thickness=20)
    cv2.line(dummy_prob, (500, 350), (600, 700), 0.9, thickness=18)
    insts = binary_to_instances(dummy_prob, thresh=0.50)
    print(f"Post-processing + Overlap Resolution verified: Found {len(insts)} instance(s).")
    
    # Overlap check verification
    if len(insts) > 1:
        combined = np.zeros_like(insts[0], dtype=np.int32)
        for m in insts:
            combined += (m > 0).astype(np.int32)
        assert np.max(combined) <= 1, "Error: Overlaps detected!"
        print("Overlap verification passed: Max pixel overlap is 1.")
