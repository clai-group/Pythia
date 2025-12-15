import os
import random
import shutil
import argparse
from typing import Tuple


def split_csv_folder(
    source_folder: str,
    ratio: float,
    dev_folder: str = None,
    val_folder: str = None,
) -> Tuple[str, str]:
    """
    Splits a folder of CSVs into dev and val sets using the given ratio.
    
    Args:
        source_folder (str): Path to the folder that contains CSV files.
        ratio (float): Fraction for dev set (e.g., 0.7 means 70% dev, 30% val).
        dev_folder (str, optional): Output folder for dev CSVs. Defaults to source_folder/dev.
        val_folder (str, optional): Output folder for val CSVs. Defaults to source_folder/val.

    Returns:
        (dev_folder_path, val_folder_path): The actual folders where files were placed.
    """
    if not 0 < ratio < 1:
        raise ValueError("ratio must be between 0 and 1")

    if dev_folder is None:
        dev_folder = os.path.join(source_folder, "dev")
        os.makedirs(dev_folder, exist_ok=True)
    if val_folder is None:
        val_folder = os.path.join(source_folder, "val")
        os.makedirs(val_folder, exist_ok=True)

    csv_files = [f for f in os.listdir(source_folder) if f.lower().endswith(".csv")]
    if not csv_files:
        raise ValueError("No CSV files found in source folder")

    random.shuffle(csv_files)

    split_index = int(len(csv_files) * ratio)
    dev_files = csv_files[:split_index]
    val_files = csv_files[split_index:]

    os.makedirs(dev_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    for f in dev_files:
        shutil.move(os.path.join(source_folder, f), dev_folder)

    for f in val_files:
        shutil.move(os.path.join(source_folder, f), val_folder)

    return dev_folder, val_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a folder of CSVs into dev and val sets.")
    parser.add_argument("source_folder", type=str, help="Path to folder with CSVs")
    parser.add_argument("ratio", type=float, help="Dev split ratio (e.g., 0.7)")
    args = parser.parse_args()

    dev_path, val_path = split_csv_folder(args.source_folder, args.ratio)

    print(f"Dev dataset created at: {dev_path}")
    print(f"Val dataset created at: {val_path}")
