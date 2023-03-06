import os

import numpy as np
from dataset.Dataset import recreate_dataset
from model.CNN import CNN
from dataset.Dataset import load_images
import tensorflow as tf
from dataset.Dataset import calculate_statistics
from model.RandomClassifier import RandomClassifier

if __name__ == '__main__':
    # Print GPU information.
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    if len(tf.config.list_physical_devices('GPU')) > 0:
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            print("Name:", gpu.name, "  Type:", gpu.device_type)
    print()

    # Print CPU information.
    print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
    if len(tf.config.list_physical_devices('CPU')) > 0:
        cpus = tf.config.list_physical_devices('CPU')
        for cpu in cpus:
            print("Name:", cpu.name, "  Type:", cpu.device_type)
    print()

    # Load dataset
    img_folder_raw = "assets/raw_dataset/RAF/Image/aligned/"
    img_folder_filter_vr = "assets/dataset/images/filter_vr/"
    img_folder_filter_mask = "assets/dataset/images/filter_mask/"
    img_folder_orig = "assets/dataset/images/original/"
    label_file = "assets/raw_dataset/RAF/EmoLabel/list_partition_label.txt"
    img_names = []
    labels = []
    print("Loading labels...")
    with open(label_file, "r") as f:
        for line in f:
            img_name, label = line.strip().split()
            img_names.append(img_name)
            labels.append(int(label))
    labels = np.array(labels)
    labels = labels - 1
    img_names = [sub.replace('.jpg', '_aligned.jpg') for sub in img_names]
    print("Labels loaded:", len(labels))
    print()

    # Recreate dataset
    print("Recreating dataset...")
    img_names, labels = recreate_dataset(img_names, labels, img_folder_orig, img_folder_filter_vr, img_folder_filter_mask, img_folder_raw)
    print("Dataset recreated:", len(labels))
    print()

    # Calculate statistics
    label_percentages = calculate_statistics(labels)
    # label_names = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]
    # Ordered as expected by the loop.
    label_names = ["Sadness", "Happiness", "Surprise", "Anger", "Fear", "Disgust", "Neutral"]
    # Print the label percentages
    i = 0
    for label, percentage in label_percentages.items():
        print(f"{label_names[i]}: {percentage:.2f}%")
        i = i + 1

    # Load orig images.
    print("Loading orig images...")
    images = load_images(img_names, img_folder_orig)
    print("Orig images loaded:", len(images))
    print()

    # Define, compile and train a Random classifier over orig images.
    print("Applying random classifier over orig images...")
    randomClassifier = RandomClassifier(7, (100, 100, 3), 5, 20, "rand")
    randomClassifier.train_and_evaluate(images, labels, ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"])
    print("Random classifier applied over orig images...")
    print()

    # Define, compile and train a CNN model over orig images
    cnn_model = CNN(7, (100, 100, 3), 5, 20, "orig")
    print("Training model over orig images...")
    cnn_model.train_and_evaluate(images, labels, ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"])
    print("Model trained over orig images...")
    print()

    # Load filter vr images.
    print("Loading filter images...")
    images = load_images(img_names, img_folder_filter_vr)
    print("Filter images loaded...")
    print()

    # Define, compile and train a CNN model over filter vr images
    cnn_model = CNN(7, (100, 100, 3), 5, 20, "filter_vr")
    print("Training model over filter images...")
    cnn_model.train_and_evaluate(images, labels, ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"])
    print("Model trained over filter images...")
    print()

    # Load filter face mask images.
    print("Loading filter images...")
    images = load_images(img_names, img_folder_filter_mask)
    print("Filter images loaded...")
    print()

    # Define, compile and train a CNN model over filter mask images
    cnn_model = CNN(7, (100, 100, 3), 5, 20, "filter_mask")
    print("Training model over filter images...")
    cnn_model.train_and_evaluate(images, labels, ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"])
    print("Model trained over filter images...")
    print()
