"""
Step 1: Download the EuroSAT RGB dataset.

What this script does, in plain English:
1. Downloads a zip file containing 27,000 satellite images (organized into
   10 folders, one per land-cover class) from Zenodo, a trusted open-data
   repository used by researchers.
2. Saves the zip into ./data/
3. Unzips it, so we end up with ./data/EuroSAT_RGB/<ClassName>/<image>.jpg

We only need to run this once. After this, the images live on disk and
every later script just reads them from ./data/EuroSAT_RGB/.
"""

import os
import zipfile
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ZIP_PATH = os.path.join(DATA_DIR, "EuroSAT_RGB.zip")
EXTRACT_DIR = os.path.join(DATA_DIR, "EuroSAT_RGB")
URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"


def download():
    if os.path.exists(EXTRACT_DIR):
        print(f"Dataset already extracted at {EXTRACT_DIR}, skipping download.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(ZIP_PATH):
        print(f"Downloading EuroSAT_RGB.zip (~95MB) from Zenodo...")

        last_reported = -1

        def show_progress(block_num, block_size, total_size):
            nonlocal last_reported
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 // total_size)
            if percent != last_reported and percent % 10 == 0:
                last_reported = percent
                print(f"  {percent}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)")

        urllib.request.urlretrieve(URL, ZIP_PATH, reporthook=show_progress)
        print("Download complete.")
    else:
        print("Zip file already downloaded, skipping download step.")

    print("Extracting zip file...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print(f"Extraction complete. Dataset is at: {EXTRACT_DIR}")

    # Clean up the zip file to save disk space now that it's extracted.
    os.remove(ZIP_PATH)
    print("Removed the zip file (no longer needed, data is extracted).")


if __name__ == "__main__":
    download()
