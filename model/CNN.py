import os
import csv
import shutil

import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

from metrics.Metrics import entropy


class CNN:

    def __init__(self, num_classes, image_size, folds, epochs, directory):

        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        self.num_classes = num_classes
        self.image_size = image_size
        self.folds = folds
        self.epochs = epochs
        self.dir = directory
        pass

    def define_and_compile_model(self):
        # Define model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.image_size),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model

    def train_and_evaluate(self, images, labels, class_names):
        # Define cross-validation splits
        skf = StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=42)

        # Store fold metrics.
        fold_metrics = []

        # Remove previous results
        if os.path.exists(f"results/{self.dir}/"):
            shutil.rmtree(f"results/{self.dir}/")

        # Train and evaluate models
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(images, labels)):
            print(f"Training model for fold {fold_idx + 1}")

            # Create dir to store results
            fold_dir = f"results/{self.dir}/fold_{fold_idx + 1}"
            if not os.path.exists(fold_dir):
                os.makedirs(fold_dir)

            # Define and compile model
            model = self.define_and_compile_model()

            # Split data into train, dev, and test sets
            train_images, train_labels = images[train_idx], labels[train_idx]
            test_images, test_labels = images[test_idx], labels[test_idx]

            # Print sizes for train-dev-test
            print("Train Set Size:", len(train_labels))
            print("Test Set Size:", len(test_labels))

            # Train model
            history = model.fit(train_images, train_labels, epochs=self.epochs, validation_data=(test_images,
                                                                                                 test_labels))

            # Predict labels of test set
            test_pred_probs = model.predict(test_images)
            test_pred_labels = np.argmax(test_pred_probs, axis=1)

            # Calculate metrics
            acc = accuracy_score(test_labels, test_pred_labels)
            ent = entropy(test_labels, test_pred_labels)
            f1_m = f1_score(test_labels, test_pred_labels, average='macro', zero_division=0)
            prec_m = precision_score(test_labels, test_pred_labels, average='macro', zero_division=0)
            rec_m = recall_score(test_labels, test_pred_labels, average='macro', zero_division=0)
            f1_w = f1_score(test_labels, test_pred_labels, average='weighted', zero_division=0)
            prec_w = precision_score(test_labels, test_pred_labels, average='weighted', zero_division=0)
            rec_w = recall_score(test_labels, test_pred_labels, average='weighted', zero_division=0)
            cm = confusion_matrix(test_labels, test_pred_labels)

            # Create heatmap of confusion matrix
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Predicted labels', fontsize=16)
            ax.set_ylabel('True labels', fontsize=16)
            ax.set_title('Confusion Matrix', fontsize=18)
            ax.xaxis.set_ticklabels(class_names, fontsize=14)
            ax.yaxis.set_ticklabels(class_names, fontsize=14, rotation=0)
            plt.savefig(f"{fold_dir}/confusion_matrix.png", dpi=300)
            plt.close()

            # plot the training history for the fold
            plt.plot(history.history['loss'], label='train loss')
            print(history)
            plt.plot(history.history['val_loss'], label='test loss')
            plt.title(f"Fold {fold_idx + 1} - Model Loss")
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['train', 'val'], loc='upper right')
            plt.savefig(f"{fold_dir}/loss_plot.png")
            plt.close()

            # plot the accuracy history for the fold
            plt.plot(history.history['accuracy'], label='train acc')
            plt.plot(history.history['val_accuracy'], label='test acc')
            plt.title(f"Fold {fold_idx + 1} - Model Accuracy")
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(['train', 'val'], loc='lower right')
            plt.savefig(f"{fold_dir}/accuracy_plot.png")
            plt.close()

            # Save weights
            model.save_weights(f'{fold_dir}/model_weights_fold{fold_idx + 1}.h5')

            # Print metrics
            print(f"Metrics for fold {fold_idx + 1}:")
            print(f"\tAccuracy: {acc}")
            print(f"\tEntropy: {ent}")
            print(f"\tF1-score (macro): {f1_m}")
            print(f"\tPrecision (macro): {prec_m}")
            print(f"\tRecall (macro): {rec_m}")
            print(f"\tF1-score (weighted)': {f1_w}")
            print(f"\tPrecision (weighted)': {prec_w}")
            print(f"\tRecall (weighted)': {rec_w}")
            print(f"\tConfusion Matrix: {cm}")

            # append the metrics to the list
            fold_metrics.append([round(acc, 2), round(ent, 2),
                                 round(f1_m, 2), round(prec_m, 2), round(rec_m, 2),
                                 round(f1_w, 2), round(prec_w, 2), round(rec_w, 2)])

        # write the metrics to a CSV file
        with open(f"results/{self.dir}/fold_metrics.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Fold', 'Accuracy', 'Entropy', 'F1-score (macro)', 'Precision (macro)', 'Recall (macro)',
                             'F1-score (weighted)', 'Precision (weighted)', 'Recall (weighted)'])
            for i, fold in enumerate(fold_metrics):
                writer.writerow([i + 1] + fold)
