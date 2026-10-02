#!/usr/bin/env python
# coding: utf-8
"""
Downloads the "Salmon Re-ID Patch Ensemble" dataset (cAIge Salmon
Re-Identification Dataset), Zenodo record 20280854:
https://zenodo.org/records/20280854

GoPro video crops of Atlantic salmon swimming freely in a commercial
net-pen, individual fish tracked across frames via automated tracking
("trajectory ID" = weak re-ID label). This is the follow-up dataset to
AutoFish -- fish actually swimming, not on a conveyor belt.

We only pull what a re-ID baseline needs, not the full 17.6GB record:
  - reid_dataset.zip (3.4GB)  -- the actual images, folders 1-16
      (1-14 = train, 15 = val, 16 = test, recorded from a different camera
      -- this is the "cross-camera" split the paper reports numbers on)
  - a15_a16_idmatch_with_traj_IDs.xlsx (5.2KB) -- the 18 manually-verified
      cross-camera identity matches between folder 15 and folder 16

Skipped: analysis1-8.zip, segmentation.zip, test.zip, ap_per_query_all_models.zip
(supplementary/analysis artifacts from the paper, not needed to run our own
zero-shot eval). Re-run with --full if you want everything.

Uses the same download approach that worked reliably for AutoFish: a
requests.Session with a real browser User-Agent + retry/backoff, atomic
.part-then-rename writes, and per-file fault tolerance (one stuck file
doesn't kill the run -- it's logged and skipped, rerun the script to retry
just the missing ones).

USAGE
-----
python download_salmon_reid.py --data-root data\\salmon_reid
"""

import argparse
import sys
import time
import zipfile
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ZENODO_RECORD = "20280854"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

CORE_FILES = [
    "reid_dataset.zip",
    "a15_a16_idmatch_with_traj_IDs.xlsx",
]
FULL_EXTRA_FILES = [
    "analysis1.zip", "analysis2.zip", "analysis3.zip", "analysis4.zip",
    "analysis5.zip", "analysis6.zip", "analysis7.zip", "analysis8.zip",
    "segmentation.zip", "test.zip", "ap_per_query_all_models.zip",
]


def _get_http_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/128.0.0.0 Safari/537.36"),
    })
    retry = Retry(
        total=5, connect=5, read=5, backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _download_file(session, url, dest: Path, chunk_size=256 * 1024):
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with session.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            written = 0
            with open(part, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = 100 * written / total
                        print(f"\r  {dest.name}: {written/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)",
                              end="", flush=True)
                    else:
                        print(f"\r  {dest.name}: {written/1e6:.1f} MB", end="", flush=True)
        print()
        part.rename(dest)
        return True
    except Exception as e:
        print(f"\n  FAILED: {dest.name}: {e}")
        if part.exists():
            part.unlink(missing_ok=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download the Salmon Re-ID dataset from Zenodo")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--full", action="store_true",
                         help="also download the analysis/segmentation/test extras (17.6GB total instead of ~3.4GB)")
    parser.add_argument("--extract", action="store_true", default=True)
    parser.add_argument("--no-extract", dest="extract", action="store_false")
    args = parser.parse_args()

    base_path = Path(args.data_root)
    zips_dir = base_path / "_downloads"
    base_path.mkdir(parents=True, exist_ok=True)

    files_to_get = CORE_FILES + (FULL_EXTRA_FILES if args.full else [])
    session = _get_http_session()

    failed = []
    for fname in files_to_get:
        dest = zips_dir / fname
        if dest.exists():
            print(f"Already have {fname}, skipping.")
            continue
        url = f"{ZENODO_BASE}/{fname}?download=1"
        print(f"Downloading {fname} from {url}")
        ok = _download_file(session, url, dest)
        if not ok:
            failed.append(fname)
        time.sleep(0.3)

    if failed:
        print(f"\n{len(failed)} file(s) failed: {failed}")
        print("Rerun this script to retry just the missing ones (completed files are skipped).")

    if args.extract:
        reid_zip = zips_dir / "reid_dataset.zip"
        if reid_zip.exists():
            out_dir = base_path / "reid_dataset"
            if out_dir.exists() and any(out_dir.iterdir()):
                print(f"{out_dir} already extracted, skipping.")
            else:
                print(f"Extracting {reid_zip} -> {out_dir} ...")
                with zipfile.ZipFile(reid_zip) as zf:
                    zf.extractall(out_dir)
                print("Done extracting.")
        idmatch = zips_dir / "a15_a16_idmatch_with_traj_IDs.xlsx"
        if idmatch.exists():
            dest = base_path / idmatch.name
            if not dest.exists():
                import shutil
                shutil.copy(idmatch, dest)

    if failed:
        sys.exit(1)

    print("\nDone. Next: run `python explore_structure.py --data-root <same path>` "
          "to print out the actual folder/file layout so the eval script can be built against it.")


if __name__ == "__main__":
    main()
