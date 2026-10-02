# Fish Re-Identification

Baselines for **individual fish re-identification** (telling one fish apart from another, not
just the species) using pretrained animal-re-ID embedding models evaluated zero-shot.

Two tracks:

- **`baseline/`** — MegaDescriptor-L-384 evaluated **zero-shot** on the public
  [AutoFish](https://huggingface.co/datasets/vapaau/autofish) dataset (1,500 images / 454
  individual fish, 6 North Sea species; Bengtson et al., WACVW 2025). This establishes a first
  re-ID number before any custom data or training. See `baseline/README.md` for the full
  rationale and expectation-setting vs. the published benchmark.
- **`baseline_salmon/`** — the same approach applied to a salmon re-ID dataset
  (`download_salmon_reid.py`, `salmon_reid.py`, plus structure-probing helpers).

## Scripts

| File | Role |
|---|---|
| `baseline/zero_shot_reid.py` | Embed query/gallery images, rank by similarity, score re-ID |
| `baseline/make_quiz.py` | Build a human-comparable "which-is-the-same-fish" quiz |
| `baseline_salmon/download_salmon_reid.py` | Fetch the salmon re-ID dataset |
| `baseline_salmon/salmon_reid.py` | Zero-shot re-ID on salmon |
| `baseline_salmon/explore_structure.py`, `probe_salmon.py` | Dataset inspection |

## Data & models

This repo is **code only** — datasets and generated outputs are not committed. See
[`DATA.md`](DATA.md) for how to fetch AutoFish, the salmon data, and the MegaDescriptor
weights (downloaded automatically from Hugging Face on first run).

## Run it

```bash
python -m pip install torch timm transformers pillow numpy scikit-learn
# then follow DATA.md to place the dataset, and:
python baseline/zero_shot_reid.py
```
