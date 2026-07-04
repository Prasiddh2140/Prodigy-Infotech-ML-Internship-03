import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
TRAIN_DIR = "train"
TEST_DIR = "test1"

IMAGE_SIZE = 64
print(os.path.exists(TRAIN_DIR))
print(os.path.exists(TEST_DIR))
print("Training Images :", len(os.listdir(TRAIN_DIR)))
print("Testing Images :", len(os.listdir(TEST_DIR)))
sample_images = sorted(os.listdir(TRAIN_DIR))[:6]

plt.figure(figsize=(12,6))

for i, image_name in enumerate(sample_images):
    image = cv2.imread(os.path.join(TRAIN_DIR, image_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.subplot(2,3,i+1)
    plt.imshow(image)
    plt.title(image_name)
    plt.axis("off")

plt.tight_layout()
plt.show()
def load_images(folder, limit=None):
    X = []
    y = []

    image_files = sorted(os.listdir(folder))

    if limit:
        image_files = image_files[:limit]

    for image_name in tqdm(image_files):

        image_path = os.path.join(folder, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        image = image.flatten()

        X.append(image)

        if image_name.startswith("cat"):
            y.append(0)

        elif image_name.startswith("dog"):
            y.append(1)

    return np.array(X), np.array(y)
