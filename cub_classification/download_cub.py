#!/usr/bin/env python3
import pathlib
import tarfile
import urllib.request

DATASET_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
RAW_DIR = pathlib.Path("data/raw")
EXTERNAL_DIR = pathlib.Path("data/external")
ARCHIVE_PATH = RAW_DIR / "CUB_200_2011.tgz"

def download_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ARCHIVE_PATH.exists():
        print(f"{ARCHIVE_PATH} already exists. Skipping download.")
    else:
        print(f"Downloading dataset from {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, ARCHIVE_PATH)
        print(f"Downloaded to {ARCHIVE_PATH}")

def extract_dataset():
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    extract_path = EXTERNAL_DIR / "CUB_200_2011"
    if extract_path.exists():
        print(f"{extract_path} already exists. Skipping extraction.")
    else:
        print(f"Extracting dataset to {EXTERNAL_DIR}...")
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(path=EXTERNAL_DIR)
        print("Extraction complete.")

if __name__ == "__main__":
    download_dataset()
    extract_dataset()

