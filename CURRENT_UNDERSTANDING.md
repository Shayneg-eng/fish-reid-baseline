# Current Understanding — Fishal Identification

_Last updated: 2026-09-14_

## Where we are

First baseline result is in: **MegaDescriptor zero-shot on AutoFish scores
29.79% Rank-1 / 2.04% mAP@R** (94 query fish / 3,665 gallery instances, test
groups only). This beats the ImageNet zero-shot baselines on Rank-1 but not
on mAP@R — see "Baseline results" below. Nothing custom has been built yet;
still establishing baselines on existing models/datasets before touching our
own hardware/data pipeline.

## Active plan (baseline stage)

- **Dataset**: [AutoFish](https://huggingface.co/datasets/vapaau/autofish)
  (Bengtson et al., WACVW 2025) — 1,500 images / 454 individual fish, 6 North
  Sea species, COCO-format instance segmentation + individual fish IDs.
  Chosen for being small, clean, and already having a published re-ID
  benchmark to compare against.
- **Model**: [MegaDescriptor-L-384](https://huggingface.co/BVRA/MegaDescriptor-L-384)
  — a Swin-L model pretrained for animal re-identification (metric learning
  across ~30 species, no fish). Applied zero-shot (no training) via
  embedding + nearest-neighbor matching, using the
  [WildlifeDatasets](https://github.com/WildlifeDatasets) toolkit's approach.
- **Baseline code**: `baseline/zero_shot_reid.py` (adapted from the published
  AutoFish Re-ID benchmark repo, https://github.com/msamdk/Fish_Re_Identification).
  Supports MegaDescriptor plus the paper's own ImageNet extractors
  (Swin-T, ResNet-50, DINOv2-ViT-S/14) for direct comparison.

## Baseline results

| Extractor | Setting | Rank-1 | mAP@R | Source |
|---|---|---|---|---|
| Swin-T (ImageNet) | zero-shot | 3.19% | 0.27% | AutoFish paper |
| ResNet-50 (ImageNet) | zero-shot | 23.40% | 2.26% | AutoFish paper |
| **MegaDescriptor** | **zero-shot** | **29.79%** | **2.04%** | **us, 2026-09-14** |
| Swin-T | fine-tuned (triplet loss on AutoFish) | 90.43% | 41.65% | AutoFish paper |
| ResNet-50 | fine-tuned (triplet loss on AutoFish) | 70.21% | 13.56% | AutoFish paper |

MegaDescriptor's cross-species re-ID pretraining does transfer to fish
morphology better than generic ImageNet classification features — it beats
both ImageNet zero-shot baselines on Rank-1 (29.79% vs. 23.40% best). But
mAP@R is roughly tied with ResNet-50 (2.04% vs 2.26%), not a clean win —
MegaDescriptor is more often right about the single closest match, but not
obviously better at ranking *all* of a fish's other appearances near the
top. All zero-shot numbers remain far below the fine-tuned target
(90.43%/41.65%), as expected — this was a floor-setting exercise, not
competing with that number.

Run details: `baseline/results.csv` on Shayne's machine
(`C:\Coding\Research\Fishal Identification\baseline\`), 94 query fish /
3,665 gallery instances (5 of 25 AutoFish groups — the designated test
split), CPU inference, MegaDescriptor-L-384 (Swin-L backbone, 1536-dim
embeddings).

## Resolved: dataset download

Both the AutoFish dataset and MegaDescriptor weights live on Hugging Face,
which the cloud sandbox used to draft this baseline cannot reach (egress
locked to package registries + GitHub). Ran successfully on Shayne's local
machine instead, after some friction: `huggingface_hub`'s built-in
`snapshot_download` repeatedly stalled/hung on Windows (likely AV/firewall
interference with its threaded Xet transfer logic), so the download step
was rewritten as a plain per-file `requests.get()` loop with a real
browser User-Agent, connection retry/backoff, and per-file fault tolerance
(a stuck file no longer kills the whole run). See
`logs/2026-09-14_baseline-selection.md` for the full debugging trail.

## Other datasets considered, not yet used

- **Melops** (corkwing wrasse, [Zenodo](https://doi.org/10.5281/zenodo.17404087),
  [Sci Data paper](https://www.nature.com/articles/s41597-026-07045-1)):
  24,578 images / 9,861 individuals (1,882 resighted), PIT-tag verified,
  7-year wild longitudinal survey. Much closer to our actual research setting
  (natural environment, growth tracking over time) than AutoFish's
  conveyor-belt setup, but its own published one-shot ID baseline is weak
  (35–53%). Strong candidate for the next baseline once AutoFish is done,
  since it's the closer analog to our real deployment.
- **Undulate skate re-ID** (few-shot, [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1574954123000651)):
  smaller/narrower, kept as a few-shot-framing reference.

## Open questions

- Fine-tuning (triplet loss, matching the paper's recipe) is the natural
  next step to close the gap toward 90%/41.65% — worth doing on AutoFish
  directly, or better spent on Melops since it's the closer analog to our
  real deployment (wild, natural environment, growth tracking)?
- MegaDescriptor's mAP@R not clearly beating ResNet-50 despite winning on
  Rank-1 is a bit odd — worth digging into whether that's noise (small
  query set, 94 fish) or a real pattern (e.g. MegaDescriptor nails the
  easy/obvious matches but is no better, or worse, on harder within-species
  confusions further down the ranking).
