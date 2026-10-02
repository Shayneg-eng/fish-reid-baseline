#!/usr/bin/env python
# coding: utf-8
"""
Zero-shot fish re-identification baseline on the AutoFish dataset.

Adapted from the published AutoFish Re-ID benchmark
(https://github.com/msamdk/Fish_Re_Identification, zero-shot instance retrieval
scripts), extended to also support MegaDescriptor -- a pretrained animal
re-ID embedding model (https://huggingface.co/BVRA/MegaDescriptor-L-384,
via the WildlifeDatasets project: https://github.com/WildlifeDatasets) --
as an off-the-shelf feature extractor, in addition to the paper's own
ImageNet-pretrained baselines (ResNet-50, Swin-T, DINOv2).

WHY THIS SCRIPT EXISTS
-----------------------
The published AutoFish paper reports two very different numbers people
often conflate:
  - Zero-shot (no fine-tuning) ImageNet features:   Swin-T R1=3.19%,  mAP@R=0.27%
                                                     ResNet50 R1=23.40%, mAP@R=2.26%
  - Fine-tuned with triplet loss on AutoFish:       Swin-T R1=90.43%, mAP@R=41.65%
This script reproduces the ZERO-SHOT setting (no training), so the fine-tuned
90% number is NOT what this script should be expected to hit -- it is the
next milestone once this baseline is in hand. MegaDescriptor is included
because, unlike generic ImageNet features, it was actually pretrained for
animal re-identification (metric learning across ~30 species), so it should
land somewhere between the two ImageNet numbers and the fine-tuned target,
without requiring any training on our own data.

USAGE
-----
1. pip install -r requirements.txt
2. python zero_shot_reid.py --data-root data/autofish --extractor megadescriptor
   (extractor options: megadescriptor, swin_t, resnet50, dinov2_vits14)

If --data-root doesn't exist yet (or is missing annotations.json), the
script downloads it automatically via huggingface_hub -- but ONLY the
5 test-group folders (group_10, 14, 20, 21, 22) plus annotations.json,
since that's all this zero-shot evaluation actually reads. That's a small
slice of the full 15.9 GB repo (which also has all 25 groups, camera
calibration files, and unlabeled images -- none needed here). Pass
--full-download if you want the entire dataset instead (e.g. because
you're about to fine-tune and need the train/val groups too).

Results (R1, mAP@R) are printed and appended to results.csv next to this
script, so multiple extractors can be compared in one place.
"""

import argparse
import csv
import os
import json
import random
import time
from collections import defaultdict, namedtuple
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

SEED = 1234
TEST_GROUP_NAMES = ["group_10", "group_14", "group_20", "group_21", "group_22"]
CROP_PADDING = 2
HF_DATASET_REPO = "vapaau/autofish"


_HTTP_SESSION = None


def _get_http_session():
    """A shared requests.Session with a real browser User-Agent (some CDNs
    throttle/reset connections that look like default python-requests bot
    traffic) and transport-level retry/backoff for connection resets."""
    global _HTTP_SESSION
    if _HTTP_SESSION is not None:
        return _HTTP_SESSION

    import requests
    from requests.adapters import HTTPAdapter
    try:
        from urllib3.util.retry import Retry
    except ImportError:
        from requests.packages.urllib3.util.retry import Retry

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        }
    )
    retry = Retry(
        total=5,
        backoff_factor=2,  # 2s, 4s, 8s, 16s, 32s between transport-level retries
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    _HTTP_SESSION = session
    return session


def _download_file(url: str, dest: Path, connect_timeout=15, read_timeout=60, retries=5):
    """HTTP GET with a real User-Agent, transport-level + app-level retry/backoff
    for connection resets, writing to a .part file first and renaming on
    success so a killed/interrupted download never leaves a file that looks
    complete but isn't."""
    if dest.exists() and dest.stat().st_size > 0:
        return  # already have it; good enough for this baseline (not checksum-verified)

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    session = _get_http_session()

    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=(connect_timeout, read_timeout)) as r:
                r.raise_for_status()
                with open(part, "wb") as f:
                    for chunk in r.iter_content(chunk_size=256 * 1024):
                        if chunk:
                            f.write(chunk)
            part.rename(dest)
            time.sleep(0.3)  # small pause between files to avoid tripping rate limits
            return
        except Exception as e:
            print(f"    retry {attempt}/{retries} for {dest.name}: {e}")
            if part.exists():
                part.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(3 * attempt)


def ensure_dataset(data_root: Path, full_download: bool = False):
    """Downloads the AutoFish dataset from Hugging Face if it isn't fully
    present locally yet, using plain per-file HTTP GETs instead of
    huggingface_hub's internal snapshot_download -- that download path kept
    stalling silently (a known issue on some Windows setups, likely AV/
    firewall interference with its threaded/Xet transfer logic). Per-file
    requests.get with a short timeout makes every file's progress visible
    and isolates a stall to one file instead of hanging the whole transfer.

    Only pulls annotations.json + images belonging to the 5 test-group
    folders this zero-shot script actually reads -- a fraction of the full
    ~15.9 GB repo (which also has train/val groups, camera calibration
    files, and unlabeled images). Already-downloaded files are skipped, so
    interrupting and rerunning just resumes.
    """
    data_root.mkdir(parents=True, exist_ok=True)
    ann_path = data_root / "annotations.json"
    resolve_base = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main"

    if not ann_path.exists():
        print("Downloading annotations.json ...")
        _download_file(f"{resolve_base}/annotations.json", ann_path)
    else:
        print("annotations.json already present.")

    if full_download:
        raise NotImplementedError(
            "--full-download isn't supported by this per-file downloader; "
            "the default (test groups only) is what this script needs anyway."
        )

    with open(ann_path, "r") as f:
        coco_data = json.load(f)

    # Figure out exactly which image files belong to the test groups, straight
    # from the annotations we already have -- no extra API call needed.
    wanted = [
        img["file_name"]
        for img in coco_data.get("images", [])
        if img["file_name"].split("/")[0] in set(TEST_GROUP_NAMES)
    ]
    print(f"{len(wanted)} images needed across test groups {TEST_GROUP_NAMES}.")

    missing = [fn for fn in wanted if not (data_root / fn).exists()]
    if not missing:
        print("All test-group images already present, skipping download.")
        return

    print(f"{len(missing)} images still needed. Downloading...")
    failed = []
    for i, file_name in enumerate(tqdm(missing, desc="Downloading images"), 1):
        url = f"{resolve_base}/{file_name}"
        dest = data_root / file_name
        try:
            _download_file(url, dest)
        except Exception as e:
            print(f"    GIVING UP on {file_name} for now ({e}); continuing with the rest.")
            failed.append(file_name)

    if failed:
        print(
            f"\n{len(failed)} of {len(missing)} images could not be downloaded after "
            f"retries: {failed[:10]}{' ...' if len(failed) > 10 else ''}\n"
            "Just rerun the same command -- it'll skip everything already downloaded "
            "and only retry these."
        )
    else:
        print("Download complete.")

AnnotationInfo = namedtuple(
    "AnnotationInfo",
    ["image_path", "annotation_id", "gt_fish_id", "img_h", "img_w", "gt_mask_data"],
)


class ResizeAndPadToSquare:
    """Resize to fit within a square, preserving aspect ratio, padding the rest."""

    def __init__(self, output_size_square, fill_color=(0, 0, 0)):
        self.output_size = output_size_square
        self.fill_color = fill_color

    def __call__(self, img):
        original_w, original_h = img.size
        ratio = min(self.output_size / original_w, self.output_size / original_h)
        new_w, new_h = int(original_w * ratio), int(original_h * ratio)
        resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        padded_img = Image.new("RGB", (self.output_size, self.output_size), self.fill_color)
        pad_left = (self.output_size - new_w) // 2
        pad_top = (self.output_size - new_h) // 2
        padded_img.paste(resized_img, (pad_left, pad_top))
        return padded_img


def build_extractor(extractor_type, device):
    """Returns (model, transform, feat_dim) for the requested off-the-shelf extractor."""
    if extractor_type == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Identity()
        feat_dim, img_size = 2048, 224
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif extractor_type == "swin_t":
        import timm
        model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)
        model.head = nn.Identity()
        feat_dim, img_size = 768, 224
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif extractor_type == "dinov2_vits14":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
        feat_dim, img_size = 384, 224
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    elif extractor_type == "megadescriptor":
        # Pretrained specifically for animal re-ID (MegaDescriptor-L-384, Swin-L
        # backbone, trained across ~30 species with metric learning).
        import timm
        model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
        feat_dim, img_size = model.num_features, 384
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

    else:
        raise ValueError(f"Unsupported extractor type: {extractor_type}")

    model.eval().to(device)
    transform = transforms.Compose(
        [
            ResizeAndPadToSquare(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return model, transform, feat_dim


def get_gt_mask(annotation_seg_data, img_h, img_w):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if isinstance(annotation_seg_data, list) and annotation_seg_data:
        for poly in annotation_seg_data:
            if not poly:
                continue
            try:
                poly_np = np.array(poly, dtype=np.int32).reshape(-1, 2)
                if poly_np.shape[0] >= 3:
                    cv2.fillPoly(mask, [poly_np], 1)
            except (ValueError, TypeError):
                continue
    if mask.sum() == 0:
        return None
    return mask


def crop_image_from_gt_mask(img_np, gt_mask_np, padding=0):
    if gt_mask_np is None:
        return None
    y_coords, x_coords = np.where(gt_mask_np > 0)
    if len(y_coords) == 0:
        return None
    y_min, y_max, x_min, x_max = y_coords.min(), y_coords.max(), x_coords.min(), x_coords.max()
    img_h, img_w = img_np.shape[:2]
    y_min_p, y_max_p = max(0, y_min - padding), min(img_h - 1, y_max + padding)
    x_min_p, x_max_p = max(0, x_min - padding), min(img_w - 1, x_max + padding)
    return img_np[y_min_p : y_max_p + 1, x_min_p : x_max_p + 1]


def extract_feature_from_gt_crop(full_img_np, gt_mask_for_crop, model, transform_fn, device, crop_padding):
    fish_crop_np = crop_image_from_gt_mask(full_img_np, gt_mask_for_crop, padding=crop_padding)
    if fish_crop_np is None or fish_crop_np.size == 0:
        return None
    fish_crop_rgb = cv2.cvtColor(fish_crop_np, cv2.COLOR_BGR2RGB)
    fish_pil = Image.fromarray(fish_crop_rgb).convert("RGB")
    tensor_in = transform_fn(fish_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat_tensor = model(tensor_in)
    if feat_tensor.ndim == 4:
        return F.adaptive_avg_pool2d(feat_tensor, (1, 1)).flatten(start_dim=1)
    elif feat_tensor.ndim == 3:
        return torch.mean(feat_tensor, dim=1)
    return feat_tensor


def main():
    parser = argparse.ArgumentParser(description="Zero-shot fish re-ID baseline on AutoFish")
    parser.add_argument("--data-root", required=True, help="Path to AutoFish dataset folder (contains annotations.json)")
    parser.add_argument(
        "--extractor",
        default="megadescriptor",
        choices=["megadescriptor", "swin_t", "resnet50", "dinov2_vits14"],
    )
    parser.add_argument("--results-csv", default="results.csv")
    parser.add_argument(
        "--full-download",
        action="store_true",
        help="Download the entire ~15.9GB AutoFish repo instead of just the test groups",
    )
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Zero-Shot Re-ID Evaluation (mAP@R protocol) ---")
    print(f"Extractor: {args.extractor} | Device: {device}")

    model, transform, feat_dim = build_extractor(args.extractor, device)
    print(f"Feature dim: {feat_dim}")

    base_path = Path(args.data_root)
    ensure_dataset(base_path, full_download=args.full_download)
    coco_ann_path = base_path / "annotations.json"
    with open(coco_ann_path, "r") as f:
        coco_data = json.load(f)

    image_id_to_meta = {
        img["id"]: {
            "path": os.path.join(base_path, img["file_name"]),
            "height": img["height"],
            "width": img["width"],
        }
        for img in coco_data.get("images", [])
    }

    test_group_set = set(TEST_GROUP_NAMES)
    all_annotations_in_test_set = []
    for ann in tqdm(coco_data.get("annotations", []), desc="Filtering test annotations"):
        img_meta = image_id_to_meta.get(ann["image_id"])
        if not img_meta or ann.get("fish_id") is None:
            continue
        current_group = os.path.basename(os.path.dirname(img_meta["path"]))
        if current_group not in test_group_set:
            continue
        all_annotations_in_test_set.append(
            AnnotationInfo(
                image_path=img_meta["path"],
                annotation_id=str(ann["id"]),
                gt_fish_id=str(ann["fish_id"]),
                img_h=img_meta["height"],
                img_w=img_meta["width"],
                gt_mask_data=ann["segmentation"],
            )
        )
    print(f"Found {len(all_annotations_in_test_set)} total annotations in test groups.")

    all_embeddings_list, all_fish_ids_str, loaded_images_cache = [], [], {}
    for ann_info in tqdm(all_annotations_in_test_set, desc="Extracting features"):
        if ann_info.image_path not in loaded_images_cache:
            try:
                loaded_images_cache[ann_info.image_path] = cv2.imread(ann_info.image_path)
            except Exception:
                continue
        img_np = loaded_images_cache[ann_info.image_path]
        gt_mask_np = get_gt_mask(ann_info.gt_mask_data, ann_info.img_h, ann_info.img_w)
        if gt_mask_np is None:
            continue
        feature = extract_feature_from_gt_crop(img_np, gt_mask_np, model, transform, device, CROP_PADDING)
        if feature is not None:
            all_embeddings_list.append(feature.cpu())
            all_fish_ids_str.append(ann_info.gt_fish_id)

    if not all_embeddings_list:
        print("FATAL: No features were extracted. Exiting.")
        return

    all_embeddings = torch.cat(all_embeddings_list)
    all_fish_ids_str = np.array(all_fish_ids_str)
    print(f"Successfully extracted {len(all_embeddings)} total features.")
    del loaded_images_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    unique_ids = sorted(set(all_fish_ids_str))
    id_to_label_map = {fish_id: i for i, fish_id in enumerate(unique_ids)}
    all_labels = torch.tensor([id_to_label_map[fish_id] for fish_id in all_fish_ids_str])

    indices_by_id = defaultdict(list)
    for i, fish_id in enumerate(all_fish_ids_str):
        indices_by_id[fish_id].append(i)

    query_indices, gallery_indices = [], []
    for fish_id, indices in indices_by_id.items():
        if len(indices) < 2:
            continue
        random.shuffle(indices)
        query_indices.append(indices[0])
        gallery_indices.extend(indices[1:])

    query_embeddings = all_embeddings[query_indices]
    query_labels = all_labels[query_indices]
    gallery_embeddings = all_embeddings[gallery_indices]
    gallery_labels = all_labels[gallery_indices]
    print(f"Query set: {len(query_embeddings)} | Gallery set: {len(gallery_embeddings)}")

    if len(query_embeddings) == 0 or len(gallery_embeddings) == 0:
        print("FATAL: Query or Gallery set is empty after split. Cannot evaluate.")
        return

    calculator = AccuracyCalculator(include=("mean_average_precision_at_r", "precision_at_1"), k=None)
    accuracies = calculator.get_accuracy(query_embeddings, query_labels, gallery_embeddings, gallery_labels)

    r1 = accuracies.get("precision_at_1", 0.0) * 100
    map_r = accuracies.get("mean_average_precision_at_r", 0.0) * 100

    print("\n--- Zero-Shot Test Set Performance ---")
    print(f"Extractor: {args.extractor}")
    print(f"Rank-1 Accuracy (R1): {r1:.2f}%")
    print(f"Mean Average Precision @ R (mAP@R): {map_r:.2f}%")
    print("---------------------------------------")

    write_header = not os.path.exists(args.results_csv)
    with open(args.results_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["extractor", "rank1_pct", "map_at_r_pct", "n_query", "n_gallery", "timestamp"])
        writer.writerow(
            [args.extractor, f"{r1:.2f}", f"{map_r:.2f}", len(query_embeddings), len(gallery_embeddings), int(time.time())]
        )
    print(f"Appended result to {args.results_csv}")


if __name__ == "__main__":
    main()
