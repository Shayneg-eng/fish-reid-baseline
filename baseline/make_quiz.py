#!/usr/bin/env python
# coding: utf-8
"""
Builds a "beat the model" re-ID quiz from the same test set and embeddings
used in zero_shot_reid.py: for a handful of query fish, saves the actual
cropped images plus a few candidate matches (including the model's own
top-1 pick and the true match), and a manifest.json describing which is
which -- so a human can guess the match, then see the ground truth and
what the model picked.

Reuses zero_shot_reid.py's data loading / feature extraction directly
(same file, same directory) rather than duplicating it, and uses the same
SEED so the query/gallery split matches the actual scored run in
results.csv.

USAGE
-----
python make_quiz.py --data-root data\\autofish --extractor megadescriptor --out-dir quiz_data
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import zero_shot_reid as zsr

# MegaDescriptor-L-384 is a big model -- on CPU it's roughly 1.5-2s PER IMAGE.
# The real scored run processes all ~3,700 test-set crops (~1.5-2 HOURS on
# CPU) because it needs the whole gallery for an accurate Rank-1/mAP number.
# A quiz only needs a small, diverse slice of individuals, so by default we
# cap how many crops get run through the model -- keeping this to minutes,
# not hours. Raise --max-instances for a closer (slower) approximation of
# the full run.
DEFAULT_MAX_INSTANCES = 350


def save_crop(image_path, gt_mask_data, img_h, img_w, dest_path, max_side=320):
    """Re-crops the fish out of its source image (same logic as scoring) and
    saves it as a small PNG for the quiz page."""
    img_np = cv2.imread(image_path)
    if img_np is None:
        return False
    gt_mask_np = zsr.get_gt_mask(gt_mask_data, img_h, img_w)
    crop_np = zsr.crop_image_from_gt_mask(img_np, gt_mask_np, padding=zsr.CROP_PADDING)
    if crop_np is None or crop_np.size == 0:
        return False
    crop_rgb = cv2.cvtColor(crop_np, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    w, h = pil_img.size
    scale = max_side / max(w, h)
    if scale < 1:
        pil_img = pil_img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img.save(dest_path, "PNG", optimize=True)
    return True


def main():
    parser = argparse.ArgumentParser(description="Build a re-ID quiz from the AutoFish test set")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--extractor", default="megadescriptor",
                         choices=["megadescriptor", "swin_t", "resnet50", "dinov2_vits14"])
    parser.add_argument("--out-dir", default="quiz_data")
    parser.add_argument("--num-correct", type=int, default=4, help="quiz questions where the model got it right")
    parser.add_argument("--num-incorrect", type=int, default=4, help="quiz questions where the model got it wrong")
    parser.add_argument("--num-choices", type=int, default=5, help="options shown per question, including the true match and the model's pick")
    parser.add_argument("--max-instances", type=int, default=DEFAULT_MAX_INSTANCES,
                         help="cap on how many crops get run through the model (bounds runtime; see comment above)")
    parser.add_argument("--cache-dir", default=".quiz_cache", help="where extracted features are cached so reruns are instant")
    parser.add_argument("--force-recompute", action="store_true", help="ignore any existing feature cache")
    args = parser.parse_args()

    random.seed(zsr.SEED)
    np.random.seed(zsr.SEED)
    torch.manual_seed(zsr.SEED)

    base_path = Path(args.data_root)
    cache_path = Path(args.cache_dir) / f"features_{args.extractor}_max{args.max_instances}.pt"

    if cache_path.exists() and not args.force_recompute:
        print(f"Loading cached features from {cache_path} (delete this file or pass --force-recompute to redo extraction)...")
        cached = torch.load(cache_path, weights_only=False)
        embeddings, fish_ids, ann_infos = cached["embeddings"], cached["fish_ids"], cached["ann_infos"]
        print(f"Loaded {len(embeddings)} cached features.")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Extractor: {args.extractor} | Device: {device}")
        model, transform, feat_dim = zsr.build_extractor(args.extractor, device)

        zsr.ensure_dataset(base_path)
        with open(base_path / "annotations.json", "r") as f:
            coco_data = json.load(f)

        image_id_to_meta = {
            img["id"]: {"path": str(base_path / img["file_name"]), "height": img["height"], "width": img["width"]}
            for img in coco_data.get("images", [])
        }

        test_group_set = set(zsr.TEST_GROUP_NAMES)
        anns_by_fish = defaultdict(list)
        for ann in coco_data.get("annotations", []):
            img_meta = image_id_to_meta.get(ann["image_id"])
            if not img_meta or ann.get("fish_id") is None:
                continue
            group = Path(img_meta["path"]).parent.name
            if group not in test_group_set:
                continue
            anns_by_fish[str(ann["fish_id"])].append(zsr.AnnotationInfo(
                image_path=img_meta["path"], annotation_id=str(ann["id"]), gt_fish_id=str(ann["fish_id"]),
                img_h=img_meta["height"], img_w=img_meta["width"], gt_mask_data=ann["segmentation"],
            ))

        # Only fish with 2+ instances can ever be a valid query (need a
        # separate gallery match). Pick a random subset of *whole fish*
        # (all their instances) until we hit the instance budget, rather
        # than sampling instances directly -- so we don't end up with a
        # pile of orphaned singletons that can never be a query or a match.
        eligible_fish = [fid for fid, anns in anns_by_fish.items() if len(anns) >= 2]
        random.shuffle(eligible_fish)

        all_annotations = []
        for fid in eligible_fish:
            if len(all_annotations) >= args.max_instances:
                break
            all_annotations.extend(anns_by_fish[fid])
        print(f"Sampled {len(all_annotations)} annotations across "
              f"{len({a.gt_fish_id for a in all_annotations})} individual fish "
              f"(capped at --max-instances={args.max_instances}; "
              f"{sum(len(v) for v in anns_by_fish.values())} total available in test groups).")

        embeddings, fish_ids, ann_infos, img_cache = [], [], [], {}
        for ann_info in tqdm(all_annotations, desc="Extracting features"):
            if ann_info.image_path not in img_cache:
                img_cache[ann_info.image_path] = cv2.imread(ann_info.image_path)
            img_np = img_cache[ann_info.image_path]
            gt_mask_np = zsr.get_gt_mask(ann_info.gt_mask_data, ann_info.img_h, ann_info.img_w)
            if gt_mask_np is None:
                continue
            feature = zsr.extract_feature_from_gt_crop(img_np, gt_mask_np, model, transform, device, zsr.CROP_PADDING)
            if feature is not None:
                embeddings.append(feature.cpu())
                fish_ids.append(ann_info.gt_fish_id)
                ann_infos.append(ann_info)
        del img_cache
        embeddings = torch.cat(embeddings)
        fish_ids = np.array(fish_ids)
        print(f"Extracted {len(embeddings)} features.")

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": embeddings, "fish_ids": fish_ids, "ann_infos": ann_infos}, cache_path)
        print(f"Cached features to {cache_path} -- reruns (different --num-correct etc.) will be instant.")

    # Same query/gallery split logic (and same SEED) as zero_shot_reid.py.
    indices_by_id = defaultdict(list)
    for i, fid in enumerate(fish_ids):
        indices_by_id[fid].append(i)

    query_indices, gallery_indices = [], []
    for fid, indices in indices_by_id.items():
        if len(indices) < 2:
            continue
        random.shuffle(indices)
        query_indices.append(indices[0])
        gallery_indices.extend(indices[1:])

    gallery_embeddings = embeddings[gallery_indices]  # (G, D)
    gallery_labels = fish_ids[gallery_indices]

    gallery_norm = torch.nn.functional.normalize(gallery_embeddings, dim=1)

    correct_qs, incorrect_qs = [], []
    for qi in query_indices:
        q_emb = torch.nn.functional.normalize(embeddings[qi : qi + 1], dim=1)
        sims = (q_emb @ gallery_norm.T).squeeze(0)  # (G,)
        best_gallery_pos = int(torch.argmax(sims).item())
        predicted_fish_id = gallery_labels[best_gallery_pos]
        true_fish_id = fish_ids[qi]
        record = {
            "query_ann_idx": qi,
            "predicted_gallery_pos": best_gallery_pos,
            "predicted_fish_id": predicted_fish_id,
            "true_fish_id": true_fish_id,
            "correct": bool(predicted_fish_id == true_fish_id),
        }
        (correct_qs if record["correct"] else incorrect_qs).append(record)

    print(f"Model got {len(correct_qs)} correct, {len(incorrect_qs)} incorrect out of {len(query_indices)} queries.")

    random.shuffle(correct_qs)
    random.shuffle(incorrect_qs)
    chosen = correct_qs[: args.num_correct] + incorrect_qs[: args.num_incorrect]
    random.shuffle(chosen)
    if len(chosen) < args.num_correct + args.num_incorrect:
        print(f"WARNING: only found {len(correct_qs)} correct / {len(incorrect_qs)} incorrect examples; using what's available.")

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    if img_dir.exists():
        for p in img_dir.glob("*.png"):
            p.unlink()
    img_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for qn, rec in enumerate(tqdm(chosen, desc="Building quiz questions"), 1):
        qi = rec["query_ann_idx"]
        q_ann = ann_infos[qi]

        # Candidate pool: true match instance + model's picked instance + random distractors.
        model_pos = rec["predicted_gallery_pos"]
        if rec["correct"]:
            # Model's pick IS a true-match instance -- use that exact one so we don't
            # show two different crops of the same correct fish as separate choices.
            true_pos = model_pos
        else:
            true_gallery_positions = [g for g in range(len(gallery_labels)) if gallery_labels[g] == rec["true_fish_id"]]
            true_pos = random.choice(true_gallery_positions)

        base_positions = {true_pos, model_pos}
        needed_distractors = max(0, args.num_choices - len(base_positions))

        other_positions = [g for g in range(len(gallery_labels)) if gallery_labels[g] != rec["true_fish_id"]]
        random.shuffle(other_positions)
        distractor_positions = []
        seen_ids = {rec["true_fish_id"], rec["predicted_fish_id"]}
        for g in other_positions:
            fid = gallery_labels[g]
            if fid in seen_ids:
                continue
            seen_ids.add(fid)
            distractor_positions.append(g)
            if len(distractor_positions) >= needed_distractors:
                break

        choice_positions = list(base_positions | set(distractor_positions))
        random.shuffle(choice_positions)

        # Save query crop.
        q_dest = img_dir / f"q{qn}_query.png"
        save_crop(q_ann.image_path, q_ann.gt_mask_data, q_ann.img_h, q_ann.img_w, q_dest)

        choices = []
        for ci, gpos in enumerate(choice_positions):
            ann_idx = gallery_indices[gpos]
            c_ann = ann_infos[ann_idx]
            c_dest = img_dir / f"q{qn}_choice{ci}.png"
            ok = save_crop(c_ann.image_path, c_ann.gt_mask_data, c_ann.img_h, c_ann.img_w, c_dest)
            if not ok:
                continue
            choices.append({
                "image": f"images/q{qn}_choice{ci}.png",
                "is_true_match": gpos == true_pos,
                "is_model_pick": gpos == model_pos,
            })

        manifest.append({
            "question": qn,
            "query_image": f"images/q{qn}_query.png",
            "model_correct": rec["correct"],
            "choices": choices,
        })

    with open(out_dir / "manifest.json", "w") as f:
        json.dump({"extractor": args.extractor, "questions": manifest}, f, indent=2)

    print(f"\nWrote {len(manifest)} quiz questions to {out_dir}/ (manifest.json + images/)")


if __name__ == "__main__":
    main()
