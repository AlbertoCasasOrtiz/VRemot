import numpy as np

from sklearn.metrics import confusion_matrix


def entropy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    n = np.sum(cm)
    cm_norm = cm / n
    cm_norm[cm_norm == 0] = 1  # Avoid taking the log of 0
    return -np.sum(cm_norm * np.log2(cm_norm))
