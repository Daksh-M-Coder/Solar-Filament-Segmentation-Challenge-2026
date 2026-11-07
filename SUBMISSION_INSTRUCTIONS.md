# Solar Filament Segmentation 2026 - Submission Guide

## 1. The Format

Your submission MUST be a single `.csv` file.

- **Columns:** `filament_id`, `segmentation_rle`
- **No quotes:** Do NOT wrap the RLE string in single or double quotes.
- **No index:** Do not include the pandas index column (use `index=False`).

## 2. The Filament ID Rule

The `filament_id` must be formatted as: `[ImageStem]_[InstanceNumber]`

- Example: `20150125172714Mh_1`, `20150125172714Mh_2`
- The image stem is the filename without the `.jpeg` extension.
- The instance number starts at 1 and increments for every filament you find in that image.
- If an image has 3 filaments, you write 3 rows.
- **Warning:** Do NOT use the same `filament_id` twice. Every row must be unique.

## 3. The RLE Rule (Critical)

- The mask size is **fixed** at 2048 x 2048. Do not change this.
- You must use `pycocotools.mask.encode` to convert your binary numpy arrays into RLE.
- **Do not** store the RLE size in the CSV. Only store the counts string.
- The string must be ASCII-encoded, not bytes. (e.g., `rle['counts'].decode('utf-8')`).
- **No spaces:** Ensure there are no leading or trailing spaces in the RLE string.

## 4. The Coverage Rule

- The test set has 180 images.
- Your CSV must have at least one row for **every single one** of the 180 images.
- If your model predicts nothing for an image, you MUST still submit a fallback row (e.g., a 1x1 pixel mask at the corner) so the evaluation script doesn't crash. Use the `_1` suffix for the fallback.

## 5. The Evaluation Trap (Panoptic Quality)

- Your score is based on Panoptic Quality (PQ).
- If you merge two separate filaments into one mask, you get a **False Positive** and a **False Negative**. Your score drops massively.
- If you split one filament into two masks, you get a **False Positive** and a **False Negative**. Your score drops massively.
- **Action:** You must ensure your predicted masks are non-overlapping. Use `remove_instance_overlaps` before encoding to RLE.

## 6. Final Sanity Checks (Run this before submitting)

```python
import pandas as pd
from pathlib import Path

sub = pd.read_csv('submission.csv')
test_dir = Path('/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/test/test_images')
test_stems = set(p.stem for p in test_dir.glob('*.jpeg'))

# Check 1: No empty submissions
assert len(sub) > 0, "Submission is empty!"

# Check 2: Correct columns
assert sub.columns.tolist() == ['filament_id', 'segmentation_rle'], "Wrong columns!"

# Check 3: No quotes in RLE
assert not sub['segmentation_rle'].str.contains('"').any(), "Quotes found in RLE!"
assert not sub['segmentation_rle'].str.contains("'").any(), "Quotes found in RLE!"

# Check 4: Every test image is covered
sub_stems = set(sub['filament_id'].apply(lambda x: x.rsplit('_', 1)[0]))
missing = test_stems - sub_stems
assert len(missing) == 0, f"Missing {len(missing)} test images: {missing}"

print("ALL CHECKS PASSED. Ready to submit.")
```

## 7. What NOT to do

- Do NOT hardcode a giant Base64 string of pre-computed answers. The organizers are actively watching for this. It is cheating and will get you banned.
- Do NOT use the full 2048x2048 image for training. Crop to 512x512 or 256x256.
- Do NOT use a simple U-Net. Use Mask R-CNN or YOLOv8-Seg to handle instance separation.

Final Submission
(Added: Aug 7, 2026)

When ready, prepare the following items and submit them through this google form. The items are:

A technical report: A 4-page technical report is required. Please use this Overleaf template for your report. More details about the content of this report are provided in the template. The final report should be submitted as a single pdf file.
Code Repository: A publicly accessible link to a Git-based repository such as GitHub, GitLab, or Bitbucket, is required. The organizing committee should access your source code without requiring additional access permissions or approval. Within this repository, a file named requirements.txt must include all utilized packages and their corresponding versions. Also, a jupyter notebook should illustrate the entire pipeline. The committee should not require any additional file to reproduce the submitted predictions.
All participants who wish to be considered for the final evaluation must complete the Google form. Please ensure that all information provided is accurate and complete. Any errors or omissions that prevent the judging committee from accessing the required materials or contacting the participant may result in the submission being deemed ineligible for final evaluation.

https://www.overleaf.com/project/6a735222766a5d9bbc55c686
https://docs.google.com/forms/d/e/1FAIpQLSfyrM4dhc8QzxR90PtF6fb018KPf0S0Cbf5hPqMuUognmzQsA/viewform
