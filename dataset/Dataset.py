import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

from vrfilter.Filter import Filter


def recreate_dataset(img_names, labels, img_folder_filter, img_folder_orig, img_folder_raw):
    # Remove previous iteration's folder.
    if os.path.exists(img_folder_filter):
        shutil.rmtree(img_folder_filter)
    if not os.path.exists(img_folder_filter):
        os.makedirs(img_folder_filter)
    if os.path.exists(img_folder_orig):
        shutil.rmtree(img_folder_orig)
    if not os.path.exists(img_folder_orig):
        os.makedirs(img_folder_orig)

    # Copy images into folder
    dataset_raw_images_path = img_folder_raw
    for image_name in img_names:
        shutil.copy(dataset_raw_images_path + image_name, img_folder_orig)

    # Create filtered images
    filter_vr = Filter()
    failed_image_names = filter_vr.apply_filter_folder(img_folder_orig, img_folder_filter)
    print("Face was not recognized for", len(failed_image_names), "images. Removing...")

    # Remove failed elements from img_names and labels.
    for image_name in failed_image_names:
        idx = img_names.index(image_name)
        del img_names[idx]
        labels = np.delete(labels, idx)

    # Return failed image paths
    return img_names, labels


def load_images(img_names, img_folder):
    # Load images
    images = []
    for img_name in img_names:
        img_path = img_folder + img_name
        img = plt.imread(img_path)
        images.append(img)
    images = np.array(images)

    return images
