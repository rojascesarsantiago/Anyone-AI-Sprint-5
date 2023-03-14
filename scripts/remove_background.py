"""
This script will be used to remove noisy background from cars images to
improve the quality of our data and get a better model.
The main idea is to use a vehicle detector to extract the car
from the picture, getting rid of all the background, which may cause
confusion to our CNN model.
We must create a new folder to store this new dataset, following exactly the
same directory structure with its subfolders but with new images.
"""

import argparse
import os
import cv2
from utils.utils import walkdir
from utils.detection import get_vehicle_coordinates
from tensorflow import keras


def parse_args():
    parser = argparse.ArgumentParser(description="Train your model.")
    parser.add_argument(
        "data_folder",
        type=str,
        help=(
            "Full path to the directory having all the cars images. Already "
            "splitted in train/test sets. E.g. "
            "`/home/app/src/data/car_ims_v1/`."
        ),
    )
    parser.add_argument(
        "output_data_folder",
        type=str,
        help=(
            "Full path to the directory in which we will store the resulting "
            "cropped pictures. E.g. `/home/app/src/data/car_ims_v2/`."
        ),
    )

    args = parser.parse_args()

    return args


def main(data_folder, output_data_folder):
    """
    Parameters
    ----------
    data_folder : str
        Full path to train/test images folder.

    output_data_folder : str
        Full path to the directory in which we will store the resulting
        cropped images.
    """
    os.mkdir(output_data_folder)
    os.mkdir(f"{output_data_folder}/test")
    os.mkdir(f"{output_data_folder}/train")
    # For this function, you must:
    #   1. Iterate over each image in `data_folder`, you can
    #      use Python `os.walk()` or `utils.waldir()``
    for i in walkdir(data_folder):
    #   2. Load the image
        image = cv2.imread(os.path.join(i[0], i[1]))
        
    #   3. Run the detector and get the vehicle coordinates, use
    #      utils.detection.get_vehicle_coordinates() for this task
        x0,y0,x1,y1 = get_vehicle_coordinates(image)
        
    #   4. Extract the car from the image and store it in
    #      `output_data_folder` with the same image name. You may also need
    #      to create additional subfolders following the original
    #      `data_folder` structure.
        cropped_image = image[y0:y1,x0:x1,:]
        folder_path = os.path.join(output_data_folder, "/".join(i[0].split("/")[2:]))
        image_path = os.path.join(folder_path, i[1])        
        
        if os.path.exists(folder_path):
            keras.utils.save_img(image_path, cropped_image)
        else:
            os.mkdir(folder_path)
            keras.utils.save_img(image_path, cropped_image)

if __name__ == "__main__":
    args = parse_args()
    main(args.data_folder, args.output_data_folder)
