import os
import shutil

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from vrfilter.Filter import Filter


def recreate_dataset(img_names, labels, img_folder_orig, img_folder_filter_vr, img_folder_filter_face_mask, img_folder_raw):
    # Remove previous iteration's folder.
    if os.path.exists(img_folder_filter_vr):
        shutil.rmtree(img_folder_filter_vr)
    if not os.path.exists(img_folder_filter_vr):
        os.makedirs(img_folder_filter_vr)
    if os.path.exists(img_folder_filter_face_mask):
        shutil.rmtree(img_folder_filter_face_mask)
    if not os.path.exists(img_folder_filter_face_mask):
        os.makedirs(img_folder_filter_face_mask)
    if os.path.exists(img_folder_orig):
        shutil.rmtree(img_folder_orig)
    if not os.path.exists(img_folder_orig):
        os.makedirs(img_folder_orig)

    # Copy images into folder
    dataset_raw_images_path = img_folder_raw
    for image_name in img_names:
        shutil.copy(dataset_raw_images_path + image_name, img_folder_orig)

    # Create filtered vr images
    filters = Filter()
    failed_image_names = filters.apply_filter_vr_folder(img_folder_orig, img_folder_filter_vr)
    print("Face was not recognized for", len(failed_image_names), "images. Removing...")

    # Create filtered face mask images
    filters = Filter()
    failed_image_names = filters.apply_filter_face_mask_folder(img_folder_orig, img_folder_filter_face_mask)
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
        img = img / 255.0
        images.append(img)
    images = np.array(images)

    return images


def calculate_statistics(labels):
    label_counts = Counter(labels)

    # Calculate the percentage of each label
    total_count = len(labels)
    label_percentages = {}
    for label, count in label_counts.items():
        label_percentages[label] = count / total_count * 100

    return label_percentages
