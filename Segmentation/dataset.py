import os

import torch
from torch.utils.data import Dataset
from torch import from_numpy
import numpy as np

"""
Class represents MS Dataset. It requires image directory and mask directory.
It is not handling the 3D dataset, just the database of slices.

@:param image_dir: path to directory of images
@:param mask_dir: path to directory of masks

The slices must be stored in binary format as the numpy arrays with .npy extension.
The numpy arrays are designed to be one-channel only, which means the images must be "grayscale".
"""


class MSDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None):
        self.current_image_num = 0
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_labels = sorted([image for image in os.listdir(image_dir) if image.endswith(".npy")],
                                   key=lambda x: int(os.path.splitext(x)[0]))

        self.current_image = None, None

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_labels[index])
        mask_path = os.path.join(self.mask_dir, self.image_labels[index].replace(".npy", "_mask.npy"))

        ms_image = from_numpy(np.load(image_path)).to(torch.float32)
        ms_mask = from_numpy(np.load(mask_path)).to(torch.float32)

        ms_image = self.transform(ms_image)
        ms_mask = self.target_transform(ms_mask)

        assert ms_image.shape == ms_mask.shape, "The shapes of image and mask no: " + str(index) + "are not matching!"
        assert ms_image.device == torch.device(
            'cpu'), "The device of MS image tensor in Data Loader is not cpu, but {}!".format(ms_image.device)
        assert ms_mask.device == torch.device(
            'cpu'), "The device of MS mask tensor in Data Loader is not cpu, but {}!".format(ms_image.device)
        return ms_image, ms_mask


if __name__ == "__main__":
    IMAGES_PATH = r'/mnt/c/Users/pegaz/Desktop/Praca-Magisterska/dataset/organized/images/'
    ANNOTATIONS_PATH = r'/mnt/c/Users/pegaz/Desktop/Praca-Magisterska/dataset/organized/annotations/'
    ORIGINAL_DATASET_PATH = r'/mnt/c/Users/pegaz/Desktop/Praca-Magisterska/dataset/zenodo_read_only'
    METADATA_FILE = r'sources.txt'
    ds = MSDataset(IMAGES_PATH, ANNOTATIONS_PATH)

    for no in range(ds.__len__()):
        image, mask = ds.__getitem__(no)

    print("The test of shapes of images and masks in the dataset has been finished.")
