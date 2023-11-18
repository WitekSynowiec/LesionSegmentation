import numpy as np
from scipy import ndimage

from FilesHelpers.Nifti import Nifti


class SegmentedImage:

    def __init__(self, image: Nifti, mask: Nifti):

        if image.get_shape() != mask.get_shape():
            raise TypeError("Input image and nifti shapes must be the same. Got {} for image and {} for mask".format(image.get_shape(), mask.get_shape()))

        self.__image = image
        self.__mask = mask

        self.__center_of_mass = self.__image.get_center_of_mass()

    """
    Returns image and masks.
    """

    def get_data(self, crop: bool = True) -> (np.array, np.array):
        # Half-length of a maximum possible rectangle side, making center of mass in the center of image.
        # Take into consideration, that center of mass is the center of gravity for whole 3D image, not just a slice.
        if crop:
            image = self.__image.center_crop(self.__image.get_center_of_mass())
            mask = self.__mask.center_crop(self.__image.get_center_of_mass())

            return image, mask
        else:
            return self.__image.get_data(), self.__mask.get_data()

    def get_shape(self):
        return self.__image.get_shape()
