import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def load_utkface_data(data_dir):
    images = []
    ages = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg'):
            age = int(filename.split('_')[0])
            img_path = os.path.join(data_dir, filename)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (64, 64))
            images.append(image)
            ages.append(age)
    return np.array(images), np.array(ages)


# Etiket haritası
label_map = {
    'angry': 1,
    'disgust': 2,
    'fear': 3,
    'happy': 4,
    'sad': 5,
    'surprise': 6,
    'neutral': 7
}


def load_fer2013_data(data_dir):
    images = []
    emotions = []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        for filename in os.listdir(label_dir):
            img_path = os.path.join(label_dir, filename)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (48, 48))
            images.append(image)
            # Metin etiketini sayısal değere dönüştür
            emotions.append(label_map[label])
    return np.array(images), np.array(emotions)


utkface_images, utkface_ages = load_utkface_data('datasets/utkface/crop_part1')
fer2013_images, fer2013_emotions = load_fer2013_data('datasets/fer2013/train')
