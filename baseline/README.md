# Fish Re-ID Baseline: MegaDescriptor zero-shot on AutoFish

This is a ready-to-run baseline: an existing pretrained model (MegaDescriptor)
evaluated zero-shot (no training) on an existing public dataset (AutoFish),
to get a first re-ID number before any custom hardware/data work.

## Why this combo

- **AutoFish** (Bengtson et al., WACVW 2025): 1,500 images / 454 individual
  fish across 6 North Sea species, COCO-format instance segmentation +
  individual fish IDs. Small enough to run on a laptop, and it already has a
  published re-ID benchmark to compare against.
  Paper: https://arxiv.org/abs/2501.03767
  Dataset: https://huggingface.co/datasets/vapaau/autofish
  Benchmark code (reference): https://github.com/msamdk/Fish_Re_Identification

- **MegaDescriptor-L-384**: a Swin-L model pretrained specifically for animal
  re-identification via metric learning across ~30 species (not including
  fish, but the embedding space is designed to generalize to new re-ID tasks
  zero-shot). https://huggingface.co/BVRA/MegaDescriptor-L-384

## Important: set expectations correctly

The AutoFish paper's own zero-shot (no fine-tuning) numbers with generic
ImageNet features are weak:

| Extractor (zero-shot) | Rank-1 | mAP@R |
|---|---|---|
| Swin-T (ImageNet) | 3.19% | 0.27% |
| ResNet-50 (ImageNet) | 23.40% | 2.26% |
| Swin-T (fine-tuned w/ triplet loss on AutoFish) | 90.43% | 41.65% |

The 90% number that circulates as "the AutoFish result" is the **fine-tuned**
one, not zero-shot — don't expect this baseline to land anywhere near it.
MegaDescriptor should beat the ImageNet zero-shot numbers by a wide margin
(it's actually re-ID-pretrained, not classification-pretrained), but it is
still a floor to compare against, not a target. Fine-tuning (triplet loss,
same as the paper) is the natural next step once this number is in hand.

## Setup

```bash
pip install -r requirements.txt
```

Requires a real internet connection to Hugging Face (for the MegaDescriptor
weights) and to wherever you downloaded the AutoFish dataset. GPU optional
but much faster (454 individuals / ~1500 crops is fine on CPU too, just
slower).

## Get the dataset

Download from https://huggingface.co/datasets/vapaau/autofish and unzip
somewhere, e.g. `data/autofish/`. It should contain `annotations.json` plus
per-group image folders (`group_01/`, `group_02/`, ... `group_25/`).

## Run

```bash
python zero_shot_reid.py --data-root data/autofish --extractor megadescriptor
```

Other extractors for comparison (reproduces the paper's own zero-shot table):

```bash
python zero_shot_reid.py --data-root data/autofish --extractor swin_t
python zero_shot_reid.py --data-root data/autofish --extractor resnet50
python zero_shot_reid.py --data-root data/autofish --extractor dinov2_vits14
```

Each run appends a row to `results.csv` (extractor, Rank-1, mAP@R, query/gallery
sizes, timestamp) so you can compare all four side by side.

## Why this couldn't just run automatically

This was built and tested for correctness in a cloud sandbox whose network
is locked to package registries + GitHub — it can't reach Hugging Face,
Zenodo, Kaggle, or even torchvision's own weight servers, so it can't
actually download the dataset or the model weights itself. Run it locally
where you have normal internet access.
