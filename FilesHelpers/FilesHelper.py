import gzip
import logging
import os
import shutil
import glob

import numpy as np

from FilesHelpers.Nifti import Nifti

IMAGES_NIFTI_PATH = r'/mnt/c/Users/pegaz/Desktop/Praca-Magisterska/dataset/organized/images_nifti/'
IMAGES_PATH = r'/mnt/c/Users/pegaz/Desktop/Praca-Magisterska/dataset/organized/images/'
ANNOTATIONS_NIFTI_PATH = r'/mnt/c/Users/pegaz/Desktop/Praca-Magisterska/dataset/organized/annotations_nifti/'
ANNOTATIONS_PATH = r'/mnt/c/Users/pegaz/Desktop/Praca-Magisterska/dataset/organized/annotations/'
ORIGINAL_DATASET_PATH = r'/mnt/c/Users/pegaz/Desktop/Praca-Magisterska/dataset/zenodo_read_only'
METADATA_FILE = r'sources.txt'


# Returns number of gunzips in the directory
def list_gunzip(path):
    no = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            print(file)
            if file.endswith(".gz"):
                print(os.path.join(root, file))
                no = no + 1
    return no


def unpack_gunzip(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.gz'):
                gz_path = os.path.join(root, file)
                try:
                    with gzip.open(gz_path, 'rb') as gz_file:
                        # Remove the .gz extension to create the output file name
                        output_file = os.path.splitext(gz_path)[0]
                        with open(output_file, 'wb') as out_file:
                            out_file.write(gz_file.read())
                    # Remove the .gz file after extraction (optional)
                    os.remove(gz_path)
                except gzip.BadGzipFile:
                    pass
                    logging.warning(f"Skipping file {gz_path}: Not a valid gzip file")


def number_of_files(path, extension) -> int:
    no = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                no = no + 1
    return no


def get_file_names(path, extension):
    names = list()
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                names.append(file)
    return names


def empty_dir(path, extension='*'):
    files = glob.glob(path + extension)
    for f in files:
        os.remove(f)


"""Organize files from nii gz with names to list of nii files."""


def organize_files(path, modality='flair'):
    datasets_directories_dict = {
        "shifts_ms_pt1": {
            "msseg": ["dev_in", "eval_in", "train"]
        },
        "shifts_ms_pt2": {
            "best": ["dev_in", "eval_in", "train"],
            "ljubljana": ["dev_out"]
        }
    }

    file_name_modality = modality if modality != "flair" else modality.upper()

    empty_dir(IMAGES_NIFTI_PATH)
    empty_dir(ANNOTATIONS_NIFTI_PATH)

    fp_image_path = open(os.path.join(IMAGES_NIFTI_PATH + METADATA_FILE), 'w')
    fp_annotation_path = open(os.path.join(ANNOTATIONS_NIFTI_PATH + METADATA_FILE), 'w')
    fp_image_path.write(modality + os.linesep)
    fp_annotation_path.write(modality + os.linesep)

    it = 0
    for dataset, sources in datasets_directories_dict.items():
        for source, assignments in sources.items():
            for assignment in assignments:
                directory_path = os.path.join(path, dataset, dataset, source, assignment, modality)
                number_of_niigz_files_in_directory = number_of_files(path=directory_path,
                                                                     extension=".nii.gz")
                first_file_number = np.min(np.array(
                    [int((name.split("_")[0])) for name in get_file_names(path=directory_path, extension=".nii.gz")]))

                for file_number in range(number_of_niigz_files_in_directory):
                    data_file_name = str(
                        file_number + int(first_file_number)) + "_" + file_name_modality + "_isovox.nii.gz"
                    annotation_file_name = str(file_number + int(first_file_number)) + "_gt_isovox.nii.gz"

                    image_file_path = os.path.join(directory_path, data_file_name)
                    annotation_file_path = os.path.join(directory_path, "..", "gt", annotation_file_name)

                    new_image_path = os.path.join(IMAGES_NIFTI_PATH, str(it) + '.nii.gz')
                    new_annotation_path = os.path.join(ANNOTATIONS_NIFTI_PATH, str(it) + '_mask' + '.nii.gz')

                    shutil.copy(image_file_path, new_image_path)
                    shutil.copy(annotation_file_path, new_annotation_path)

                    unpack_gunzip(IMAGES_NIFTI_PATH)
                    unpack_gunzip(ANNOTATIONS_NIFTI_PATH)

                    fp_image_path.write(image_file_path + os.linesep + new_image_path + os.linesep)
                    fp_annotation_path.write(annotation_file_path + os.linesep + new_annotation_path + os.linesep)

                    slice_num = str(Nifti(new_image_path.replace('.gz', ''),
                                          new_annotation_path.replace('.gz', '')).get_number_of_slices())
                    annotation_slice_num = str(Nifti(new_image_path.replace('.gz', ''),
                                                     new_annotation_path.replace('.gz',
                                                                                 '')).get_number_of_annotation_slices())
                    fp_image_path.write(slice_num + os.linesep)
                    fp_annotation_path.write(annotation_slice_num + os.linesep)
                    it = it + 1

    fp_image_path.close()
    fp_annotation_path.close()


"""Retrieve slices from nii images"""


def retrieve_slices():
    empty_dir(IMAGES_PATH)
    empty_dir(ANNOTATIONS_PATH)
    empty_dir(IMAGES_NIFTI_PATH, '*.nii.gz')
    empty_dir(ANNOTATIONS_NIFTI_PATH, '*.nii.gz')
    list_image_labels = [val for val in os.listdir(IMAGES_NIFTI_PATH) if val.endswith(".nii")]
    list_mask_labels =  [val for val in os.listdir(ANNOTATIONS_NIFTI_PATH) if val.endswith(".nii")]
    i = 0
    for image in list_image_labels:
        if image.endswith(".nii"):
            nii = Nifti(os.path.join(IMAGES_NIFTI_PATH, image), os.path.join(ANNOTATIONS_NIFTI_PATH,image.replace(".nii", "_mask.nii") ))
            for slice_number in range(nii.get_number_of_slices()):
                data, annotation = nii.get_slices(slice_number)

                # Adding channel to saved matrix
                data = np.expand_dims(data, 0)
                annotation = np.expand_dims(annotation, 0)

                # print("Shape of {}. Image: {}, Mask: {}".format(image, data.shape, annotation.shape))
                new_image_path = os.path.join(IMAGES_PATH, str(i) + '.npy')
                new_annotation_path = os.path.join(ANNOTATIONS_PATH, str(i) + '_mask' + '.npy')
                np.save(new_image_path, data)
                np.save(new_annotation_path, annotation)
                i = i + 1


# Returns the number of nifti slices from nii images in path directory

def get_slices_num(path):
    slices = []
    with open(os.path.join(path, METADATA_FILE)) as file:
        for count, line in enumerate(file, start=0):
            if count % 3 == 0:
                slices.append(line)

    return [int(slice) for slice in slices[1:]]


def initiate_dataset():
    organize_files(ORIGINAL_DATASET_PATH)
    retrieve_slices()


def get_classes_numel(directory):
    mask_labels = sorted([mask for mask in os.listdir(directory) if mask.endswith(".npy")])

    positives = 0
    numels = 0
    for mask_label in mask_labels:
        print(mask_label)
        mask = np.load(os.path.join(directory, mask_label))
        positives += np.concatenate(mask).sum()
        numels += np.size(mask)


    return positives, numels-positives

if __name__ == "__main__":
    # initiate_dataset()
    print(get_classes_numel(ANNOTATIONS_PATH))