import numpy as np
from dataset.Dataset import recreate_dataset
from model.CNN import CNN
from dataset.Dataset import load_images

if __name__ == '__main__':
    # Load dataset
    img_folder_raw = "assets/raw_dataset/FER+/Image/aligned/"
    img_folder_filter = "assets/dataset/images/filter/"
    img_folder_orig = "assets/dataset/images/original/"
    label_file = "assets/raw_dataset/FER+/EmoLabel/list_partition_label.txt"
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

    # Recreate dataset
    print("Recreating dataset...")
    img_names, labels = recreate_dataset(img_names, labels, img_folder_orig, img_folder_filter, img_folder_raw)
    print("Dataset recreated:", len(labels))

    # Load orig images.
    print("Loading orig images...")
    images = load_images(img_names, img_folder_orig)
    print("Orig images loaded:", len(images))

    # Define, compile and train a CNN model over orig images
    cnn_model = CNN(7, (100, 100, 3), 5, 100, "orig")
    print("Training model over orig images...")
    cnn_model.train_and_evaluate(images, labels, ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger",
                                                  "Neutral"])
    print("Model trained over orig images...")

    # Load filter images.
    print("Loading filter images...")
    images = load_images(img_names, img_folder_filter)
    print("Filter images loaded...")

    # Define, compile and train a CNN model over filter images
    cnn_model = CNN(7, (100, 100, 3), 5, 100, "filter")
    print("Training model over filter images...")
    cnn_model.train_and_evaluate(images, labels, ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger",
                                                  "Neutral"])
    print("Model trained over filter images...")
