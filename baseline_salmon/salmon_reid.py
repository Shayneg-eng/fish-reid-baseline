#!/usr/bin/env python
# coding: utf-8
"""
Zero-shot fish re-ID baseline on the Salmon Re-ID (net-pen) dataset --
real free-swimming salmon filmed by GoPro in a commercial net-pen, as
opposed to AutoFish's conveyor-belt setup. Mirrors zero_shot_reid.py's
approach (MegaDescriptor embeddings + cosine nearest-neighbor, Rank-1/mAP@R)
but against this dataset's actual structure, which is different from
AutoFish's COCO format:

  data_root/
    reid_dataset/reid/analysis<N>/
      analysis<N>_results_refined.txt   <- per-frame detections, one row per
                                            body part (frame,traj_id,x,y,w,h,
                                            conf,-1,-1,-1,class_label);
                                            class_label=="salmon" rows are the
                                            whole-fish bounding box.
      images/a<N>_salmon_<local_id>_frame_<frame>.png  <- pre-cropped fish
                                            images, already usable directly
                                            (no segmentation-mask crop needed,
                                            unlike AutoFish).
    a15_a16_idmatch_with_traj_IDs.xlsx  <- 18 manually-verified cross-camera
                                            identity matches, columns
                                            (Query, Gallery), referring to the
                                            *global trajectory ID* (the txt
                                            file's 2nd column) -- NOT the
                                            <local_id> baked into filenames.

WHY THE MAPPING STEP
---------------------
The <local_id> in filenames (e.g. "005", "011") is a small per-analysis
enumeration, not the same number as the txt file's trajectory ID column
(e.g. 1006, 863) that a15_a16_idmatch_with_traj_IDs.xlsx actually refers to.
This script reconstructs local_id -> global_trajectory_id by matching each
local_id's exact set of frames (from image filenames) against each global
trajectory ID's exact set of frames (from the results txt) -- confirmed
correct because the frame sets match exactly for real matches.

EVAL PROTOCOL
-------------
Query: every image belonging to a local_id in analysis15 whose global
       trajectory ID appears in idmatch's "Query" column.
Gallery: every image in analysis16 (the whole test folder), labeled by its
       mapped global trajectory ID where known, or a unique per-local-id
       placeholder label otherwise (so it still acts as a distractor, but
       can never count as a correct match).
This mirrors zero_shot_reid.py's query/gallery split logic and reuses the
same extractor code and Rank-1/mAP@R accuracy calculation, so results are
directly comparable to the AutoFish run in results.csv.

USAGE
-----
python salmon_reid.py --data-root data\\salmon_reid --extractor megadescriptor
"""

import argparse
import csv
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))
import zero_shot_reid as zsr  # reuse build_extractor / ResizeAndPadToSquare

SEED = 1234
QUERY_ANALYSIS = 15
GALLERY_ANALYSIS = 16
IMG_NAME_RE = re.compile(r"a(\d+)_salmon_(\d+)_frame_(\d+)\.png")


def parse_results_txt(txt_path: Path):
    """frame,traj_id,x,y,w,h,conf,-1,-1,-1,class_label -- keep 'salmon' rows.
    Returns {global_traj_id(int): sorted list of frame numbers}."""
    frames_by_id = defaultdict(list)
    with open(txt_path, "r", errors="replace") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 11:
                continue
            if parts[-1].strip() != "salmon":
                continue
            try:
                frame = int(parts[0])
                traj_id = int(parts[1])
            except ValueError:
                continue
            frames_by_id[traj_id].append(frame)
    return {k: sorted(set(v)) for k, v in frames_by_id.items()}


def parse_image_filenames(images_dir: Path, analysis_num: int):
    """Returns (frames_by_local_id, path_by_local_id_frame)."""
    frames_by_local = defaultdict(list)
    path_by_local_frame = {}
    for p in images_dir.glob(f"a{analysis_num}_salmon_*_frame_*.png"):
        m = IMG_NAME_RE.match(p.name)
        if not m:
            continue
        _, local_id, frame = m.groups()
        local_id, frame = int(local_id), int(frame)
        frames_by_local[local_id].append(frame)
        path_by_local_frame[(local_id, frame)] = p
    return {k: sorted(set(v)) for k, v in frames_by_local.items()}, path_by_local_frame


def build_local_to_global(frames_by_local, frames_by_global):
    """Match each local_id to the global trajectory ID with the identical
    frame set (exact match expected for real data); falls back to best
    Jaccard overlap and warns if no exact match is found."""
    mapping = {}
    global_items = list(frames_by_global.items())
    for local_id, local_frames in frames_by_local.items():
        local_set = set(local_frames)
        exact = [gid for gid, gframes in global_items if set(gframes) == local_set]
        if len(exact) == 1:
            mapping[local_id] = exact[0]
            continue
        if len(exact) > 1:
            print(f"  WARNING: local_id {local_id} matches multiple global IDs exactly: {exact}; using first.")
            mapping[local_id] = exact[0]
            continue
        best_gid, best_score = None, 0.0
        for gid, gframes in global_items:
            gset = set(gframes)
            union = local_set | gset
            if not union:
                continue
            score = len(local_set & gset) / len(union)
            if score > best_score:
                best_gid, best_score = gid, score
        if best_gid is not None and best_score > 0.5:
            print(f"  local_id {local_id}: no exact frame-set match, using best overlap "
                  f"global_id={best_gid} (Jaccard={best_score:.2f})")
            mapping[local_id] = best_gid
        else:
            print(f"  local_id {local_id}: no confident global_id match found (best Jaccard={best_score:.2f}), skipping.")
    return mapping


def load_idmatch(xlsx_path: Path):
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, read_only=True)
    ws = wb[wb.sheetnames[0]]
    pairs = []
    rows = ws.iter_rows(min_row=2, values_only=True)  # skip header row
    for r in rows:
        if r[0] is None or r[1] is None:
            continue
        pairs.append((int(r[0]), int(r[1])))
    return pairs


def load_and_transform(path, transform_fn, device):
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).convert("RGB")
    tensor_in = transform_fn(pil_img).unsqueeze(0).to(device)
    return tensor_in


def extract_feature(tensor_in, model, device):
    with torch.no_grad():
        feat_tensor = model(tensor_in)
    if feat_tensor.ndim == 4:
        return F.adaptive_avg_pool2d(feat_tensor, (1, 1)).flatten(start_dim=1)
    elif feat_tensor.ndim == 3:
        return torch.mean(feat_tensor, dim=1)
    return feat_tensor


def main():
    parser = argparse.ArgumentParser(description="Zero-shot fish re-ID baseline on Salmon Re-ID (net-pen)")
    parser.add_argument("--data-root", required=True, help="Path containing reid_dataset/ and the idmatch xlsx")
    parser.add_argument("--extractor", default="megadescriptor",
                         choices=["megadescriptor", "swin_t", "resnet50", "dinov2_vits14"])
    parser.add_argument("--results-csv", default="results.csv")
    parser.add_argument("--cache-dir", default=".salmon_cache")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--max-gallery-fish", type=int, default=None,
                         help="cap the gallery to this many fish (whole fish, all their frames), for a fast "
                              "representative-subset run instead of the full analysis16 folder. The fish that "
                              "are actual verified matches for a query are always kept; the rest of the cap is "
                              "filled with a random sample of other fish as distractors.")
    args = parser.parse_args()

    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    base = Path(args.data_root)
    reid_root = base / "reid_dataset" / "reid"
    query_dir = reid_root / f"analysis{QUERY_ANALYSIS}"
    gallery_dir = reid_root / f"analysis{GALLERY_ANALYSIS}"
    idmatch_path = base / "a15_a16_idmatch_with_traj_IDs.xlsx"

    for p in [query_dir, gallery_dir, idmatch_path]:
        if not p.exists():
            print(f"FATAL: expected path not found: {p}\n"
                  f"Run download_salmon_reid.py first (see explore_structure.py output).")
            return

    print("Parsing frame/trajectory-ID mapping for analysis15 (query) and analysis16 (gallery)...")
    q_frames_by_global = parse_results_txt(query_dir / f"analysis{QUERY_ANALYSIS}_results_refined.txt")
    q_frames_by_local, q_path_by_local_frame = parse_image_filenames(query_dir / "images", QUERY_ANALYSIS)
    q_local_to_global = build_local_to_global(q_frames_by_local, q_frames_by_global)

    g_frames_by_global = parse_results_txt(gallery_dir / f"analysis{GALLERY_ANALYSIS}_results_refined.txt")
    g_frames_by_local, g_path_by_local_frame = parse_image_filenames(gallery_dir / "images", GALLERY_ANALYSIS)
    g_local_to_global = build_local_to_global(g_frames_by_local, g_frames_by_global)

    print(f"analysis15: {len(q_local_to_global)}/{len(q_frames_by_local)} local IDs mapped to a global trajectory ID.")
    print(f"analysis16: {len(g_local_to_global)}/{len(g_frames_by_local)} local IDs mapped to a global trajectory ID.")

    idmatch_pairs = load_idmatch(idmatch_path)
    print(f"Loaded {len(idmatch_pairs)} verified cross-camera identity matches from idmatch xlsx.")

    q_global_to_local = {v: k for k, v in q_local_to_global.items()}
    g_global_to_local = {v: k for k, v in g_local_to_global.items()}

    # query_global and gallery_global are two DIFFERENT numbering schemes
    # (each video's tracker starts its own IDs) -- idmatch is the only thing
    # that says "query_global in analysis15 is the same fish as gallery_global
    # in analysis16". So the shared identity label a query image gets must be
    # the *gallery-side* ID, not its own video's ID -- otherwise query and
    # gallery labels never share a value and nothing can ever match.
    query_local_to_label = {}  # q_local -> "g{gallery_global}"
    resolved_pairs = []
    for query_global, gallery_global in idmatch_pairs:
        q_local = q_global_to_local.get(query_global)
        g_local = g_global_to_local.get(gallery_global)
        if q_local is None or g_local is None:
            print(f"  Skipping idmatch pair (query_global={query_global}, gallery_global={gallery_global}): "
                  f"{'query side' if q_local is None else 'gallery side'} not found in this analysis' images.")
            continue
        query_local_to_label[q_local] = f"g{gallery_global}"
        resolved_pairs.append((query_global, gallery_global))
    print(f"{len(resolved_pairs)}/{len(idmatch_pairs)} verified matches resolved to actual images on both sides.")
    if not resolved_pairs:
        print("FATAL: no verified matches resolved to real images -- can't build an eval set.")
        return

    # Build the (path, identity_label) list for query crops.
    query_items = []  # (path, label)
    for local_id, label in query_local_to_label.items():
        for frame in q_frames_by_local[local_id]:
            path = q_path_by_local_frame.get((local_id, frame))
            if path is not None:
                query_items.append((path, label))

    # Gallery: every image in analysis16, labeled by mapped global id, or a
    # unique per-local-id placeholder (never matches anything -- pure distractor).
    gallery_local_ids = list(g_frames_by_local.keys())
    if args.max_gallery_fish is not None and args.max_gallery_fish < len(gallery_local_ids):
        must_keep_globals = {gallery_global for _, gallery_global in resolved_pairs}
        must_keep = [lid for lid in gallery_local_ids if g_local_to_global.get(lid) in must_keep_globals]
        rest = [lid for lid in gallery_local_ids if lid not in set(must_keep)]
        random.shuffle(rest)
        n_extra = max(0, args.max_gallery_fish - len(must_keep))
        gallery_local_ids = must_keep + rest[:n_extra]
        print(f"Capping gallery to {len(gallery_local_ids)} fish ({len(must_keep)} verified-match fish "
              f"kept + {min(n_extra, len(rest))} random distractor fish), out of {len(g_frames_by_local)} available.")

    gallery_items = []
    for local_id in gallery_local_ids:
        frames = g_frames_by_local[local_id]
        label = f"g{g_local_to_global[local_id]}" if local_id in g_local_to_global else f"unmapped16_{local_id}"
        for frame in frames:
            path = g_path_by_local_frame.get((local_id, frame))
            if path is not None:
                gallery_items.append((path, label))

    print(f"Query set: {len(query_items)} images ({len(query_local_to_label)} verified fish). "
          f"Gallery set: {len(gallery_items)} images ({len(g_frames_by_local)} fish, "
          f"{len(g_local_to_global)} with a known cross-camera identity).")

    cap_tag = f"_cap{args.max_gallery_fish}" if args.max_gallery_fish is not None else ""
    cache_path = Path(args.cache_dir) / f"salmon_features_{args.extractor}{cap_tag}.pt"
    if cache_path.exists() and not args.force_recompute:
        print(f"Loading cached features from {cache_path} (pass --force-recompute to redo extraction)...")
        cached = torch.load(cache_path, weights_only=False)
        query_embeddings, query_labels_str = cached["query_embeddings"], cached["query_labels_str"]
        gallery_embeddings, gallery_labels_str = cached["gallery_embeddings"], cached["gallery_labels_str"]
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Extractor: {args.extractor} | Device: {device}")
        total_images = len(query_items) + len(gallery_items)
        est_minutes = total_images * (1.5 if device == "cpu" else 0.05) / 60
        print(f"About to extract features for {total_images} images "
              f"(~{est_minutes:.0f} min estimated on {device}). Cached afterward -- reruns are instant.")

        model, transform, feat_dim = zsr.build_extractor(args.extractor, device)

        def extract_all(items, desc):
            embs, labels = [], []
            for path, label in tqdm(items, desc=desc):
                tensor_in = load_and_transform(path, transform, device)
                if tensor_in is None:
                    continue
                feat = extract_feature(tensor_in, model, device)
                embs.append(feat.cpu())
                labels.append(label)
            return torch.cat(embs) if embs else torch.empty(0, feat_dim), np.array(labels)

        query_embeddings, query_labels_str = extract_all(query_items, "Extracting query features")
        gallery_embeddings, gallery_labels_str = extract_all(gallery_items, "Extracting gallery features")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "query_embeddings": query_embeddings, "query_labels_str": query_labels_str,
            "gallery_embeddings": gallery_embeddings, "gallery_labels_str": gallery_labels_str,
        }, cache_path)
        print(f"Cached features to {cache_path}.")

    if len(query_embeddings) == 0 or len(gallery_embeddings) == 0:
        print("FATAL: query or gallery embeddings empty after extraction.")
        return

    unique_labels = sorted(set(query_labels_str) | set(gallery_labels_str))
    label_to_int = {l: i for i, l in enumerate(unique_labels)}
    query_labels = torch.tensor([label_to_int[l] for l in query_labels_str])
    gallery_labels = torch.tensor([label_to_int[l] for l in gallery_labels_str])

    calculator = AccuracyCalculator(include=("mean_average_precision_at_r", "precision_at_1"), k=None)
    accuracies = calculator.get_accuracy(query_embeddings, query_labels, gallery_embeddings, gallery_labels)

    r1 = accuracies.get("precision_at_1", 0.0) * 100
    map_r = accuracies.get("mean_average_precision_at_r", 0.0) * 100

    print("\n--- Zero-Shot Salmon Re-ID (net-pen, cross-camera) Performance ---")
    print(f"Extractor: {args.extractor}")
    print(f"Rank-1 Accuracy (R1): {r1:.2f}%")
    print(f"Mean Average Precision @ R (mAP@R): {map_r:.2f}%")
    print("(Paper's own cross-camera numbers for reference: full-image baseline 60.9% mAP, "
          "their patch-ensemble method 86.0% mAP.)")
    print("-------------------------------------------------------------------")

    write_header = not os.path.exists(args.results_csv)
    with open(args.results_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["dataset", "extractor", "rank1_pct", "map_at_r_pct", "n_query", "n_gallery", "timestamp"])
        writer.writerow(["salmon_reid_netpen", args.extractor, f"{r1:.2f}", f"{map_r:.2f}",
                          len(query_embeddings), len(gallery_embeddings), int(time.time())])
    print(f"Appended result to {args.results_csv}")


if __name__ == "__main__":
    main()
