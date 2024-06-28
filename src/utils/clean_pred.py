# Delete extra data
import glob
import multiprocessing
import os
import re
import shutil
import sys

import numpy as np
import rasterio
from PIL import Image


def info_from_path(path):
    """Extracts epoch number and area name from a given file path.

    Args:
        path (str): The file path.

    Returns:
        tuple: A tuple containing the epoch number and area name.
    """
    # Get file name
    file_name = os.path.basename(path)
    # Parse name
    match = re.search(r"eval_(\d+)_(\w+?)_(\d+c?m_)?argmax", file_name)
    if match is None:
        return None, None
    epoch_number = int(match.group(1))
    area_name = match.group(2)

    # matching rule for raster_targets
    # match = re.search(r"(\w+?)_(\d+c?m_)?target", file_name)
    # epoch_number = 0  # int(match.group(1))
    # area_name = match.group(1)
    return epoch_number, area_name


def process_dir(path):
    """Process the directory containing prediction files from on training.

    Args:
        path (str): The path to the directory.

    Returns:
        None
    """
    # Create a dict(epoch:dict(name:path)) from all sub .tif files
    print(f"Exploring path {path}")
    file_dict = {}
    for file_path in glob.glob(path + "/*.tif"):
        # print(f"Exploring {file_path}")
        epoch_number, area_name = info_from_path(file_path)
        if epoch_number is None or area_name is None:
            print(f"Invalid file name {file_path}")
            continue
        if epoch_number not in file_dict:
            file_dict[epoch_number] = {}
        file_dict[epoch_number][area_name] = file_path

    if len(file_dict) == 0:
        print("No prediction files found")
        return

    print("Deleting extra files...", end="")
    # Delete extra prediction files
    # Only best and last prediction as well as generated jpg are kept
    last_epoch = max(int(epoch) for epoch in file_dict.keys())
    # look into ../checkpoints/ for best model epoch
    best_epoch = max(
        [
            int(file_name.split("_")[1].split(".")[0])
            for file_name in os.listdir(os.path.join(path, "../checkpoints/"))
            if len(file_name.split("_")) > 1
        ]
    )
    for file_path in glob.glob(path + "/*.tif"):
        epoch_number, area_name = info_from_path(file_path)
        if epoch_number != last_epoch and epoch_number != best_epoch:
            os.remove(file_path)

    # Delete eval_prepared directory
    eval_prepared_dir = os.path.join(path, "../eval_prepared")
    if os.path.exists(eval_prepared_dir):
        shutil.rmtree(eval_prepared_dir)
    print("Done")


def explore_directories(path):
    """Explore all subdirectories recurcively and process the 'preds' subdirectory if it exists.

    Args:
        path (str): The path to the root directory.

    Returns:
        None
    """
    for root, dirs, files in os.walk(path):
        if "preds" in dirs:
            subdir_path = os.path.join(root, "preds")
            # check if there is already png file
            if not len(glob.glob(subdir_path + "/*.png")) > 0:
                process_dir(subdir_path)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 0:
        Run_path = args[0]
    else:
        Run_path = "./logs/"
    explore_directories(Run_path)
