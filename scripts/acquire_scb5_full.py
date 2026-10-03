"""Acquire full SCB-Dataset5 via parallel chunk download and extract safely."""
import concurrent.futures
import json
import os
from pathlib import Path
import sys
import time
import zipfile
import requests

DATASET_DIR = Path("datasets/raw/scb_dataset5_full").resolve()
METADATA_FILE = DATASET_DIR / "metadata.json"
ARCHIVE_FILE = DATASET_DIR / "original_download.zip"
PARTIAL_FILE = DATASET_DIR / "original_download.zip.partial"
KAGGLE_API_URL = "https://www.kaggle.com/api/v1/datasets/download/shreyasudaya/scb-05-dataset"


def get_signed_url():
    with requests.get(KAGGLE_API_URL, stream=True, allow_redirects=False) as res:
        if res.status_code == 302:
            return res.headers.get("Location")
        raise RuntimeError(f"Unexpected status getting signed URL: {res.status_code}")


def download_chunk(url, start, end, filepath):
    headers = {"Range": f"bytes={start}-{end}"}
    for attempt in range(5):
        try:
            resp = requests.get(url, headers=headers, timeout=(30, 90))
            resp.raise_for_status()
            with open(filepath, "r+b") as f:
                f.seek(start)
                f.write(resp.content)
            return len(resp.content)
        except Exception as e:
            if attempt == 4:
                raise
            time.sleep(1 + attempt * 2)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if not ARCHIVE_FILE.exists():
        print(f"Resolving signed URL from {KAGGLE_API_URL}...")
        signed_url = get_signed_url()
        head = requests.head(signed_url)
        total_size = int(head.headers.get("content-length", 0))
        print(f"Total archive size: {total_size} bytes ({total_size / (1024**3):.2f} GB)")

        if not PARTIAL_FILE.exists() or PARTIAL_FILE.stat().st_size != total_size:
            print("Pre-allocating partial file...")
            with open(PARTIAL_FILE, "wb") as f:
                f.seek(total_size - 1)
                f.write(b"\0")

        chunk_size = 16 * 1024 * 1024  # 16 MB chunks
        chunks = []
        for start in range(0, total_size, chunk_size):
            end = min(start + chunk_size - 1, total_size - 1)
            chunks.append((start, end))

        print(f"Downloading {len(chunks)} chunks using 16 worker threads...")
        t0 = time.time()
        downloaded = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_chunk = {
                executor.submit(download_chunk, signed_url, c[0], c[1], PARTIAL_FILE): c
                for c in chunks
            }
            last_report = time.time()
            for future in concurrent.futures.as_completed(future_to_chunk):
                size = future.result()
                downloaded += size
                now = time.time()
                if now - last_report > 5 or downloaded == total_size:
                    pct = (downloaded / total_size) * 100
                    mb_s = (downloaded / (1024 * 1024)) / (now - t0 + 0.001)
                    print(f"Progress: {pct:.1f}% ({downloaded / (1024**2):.1f}/{total_size / (1024**2):.1f} MB) at {mb_s:.1f} MB/s", flush=True)
                    last_report = now

        print(f"Download complete in {time.time() - t0:.1f}s. Verifying zip integrity...")
        if not zipfile.is_zipfile(PARTIAL_FILE):
            raise ValueError("Downloaded file is not a valid zip archive!")
        PARTIAL_FILE.rename(ARCHIVE_FILE)
        print("Archive successfully verified and saved to:", ARCHIVE_FILE)
    else:
        print("Archive already exists:", ARCHIVE_FILE)

    print("Extracting members...")
    with zipfile.ZipFile(ARCHIVE_FILE) as z:
        members = z.infolist()
        print(f"Total archive members: {len(members)}")
        extracted = 0
        last_report = time.time()
        for i, member in enumerate(members):
            target = (DATASET_DIR / member.filename).resolve()
            if not target.is_relative_to(DATASET_DIR):
                raise ValueError(f"Unsafe member: {member.filename}")
            if not target.exists():
                z.extract(member, DATASET_DIR)
                extracted += 1
            now = time.time()
            if now - last_report > 5:
                print(f"Extraction progress: {i + 1}/{len(members)} entries...", flush=True)
                last_report = now
        print(f"Extraction complete. Newly extracted: {extracted} files.")

    metadata = {
        "dataset_name": "scb_dataset5_full",
        "source": "Kaggle (shreyasudaya/scb-05-dataset)",
        "source_url": "https://www.kaggle.com/datasets/shreyasudaya/scb-05-dataset",
        "upstream_author": "Fan Yang (Winston Yang Fan) / Shreyas Udaya",
        "paper_citation": "SCB-dataset: A dataset for detecting student classroom behavior (arXiv:2304.02488)",
        "license": "MIT (as declared on Kaggle host) / Academic Research (upstream)",
        "license_status": "Academic Research / Non-commercial",
        "download_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "local_path": str(DATASET_DIR),
        "archive_size_bytes": ARCHIVE_FILE.stat().st_size if ARCHIVE_FILE.exists() else 0,
        "notes": "Full SCB-05 Dataset release covering all 20 classes described in arXiv:2304.02488."
    }
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("Metadata written to:", METADATA_FILE)


if __name__ == "__main__":
    main()
