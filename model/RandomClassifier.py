import os
import csv
import shutil
import random

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

import seaborn as sns

from metrics.Metrics import entropy

import matplotlib.pyplot as plt
import matplotlib as mpl


class RandomClassifier:

    def __init__(self, num_classes, image_size, folds, epochs, directory):
        mpl.use('TkAgg')

        self.num_classes = num_classes
        self.image_size = image_size
        self.folds = folds
        self.epochs = epochs
        self.dir = directory
        pass

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

            # Split data into train, dev, and test sets
            train_images, train_labels = images[train_idx], labels[train_idx]
            test_images, test_labels = images[test_idx], labels[test_idx]

            # Print sizes for train-dev-test
            print("Train Set Size:", len(train_labels))
            print("Test Set Size:", len(test_labels))

            # Predict labels of test set
            test_pred_labels = self.predict(test_images)

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
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(['Fold', 'Accuracy', 'Entropy', 'F1-score (macro)', 'Precision (macro)', 'Recall (macro)',
                             'F1-score (weighted)', 'Precision (weighted)', 'Recall (weighted)'])
            for i, fold in enumerate(fold_metrics):
                writer.writerow([i + 1] + fold)

    def predict(self, images):
        labels = []
        for image in images:
            label = random.choice([0, 1, 2, 3, 4, 5, 6])  # replace with your own class names
            labels.append(label)
        return labels
