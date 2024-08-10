from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2

# Etiket haritası
label_map = {
    'angry': 0,
    'disgust': 1,
    'fear': 2,
    'happy': 3,
    'sad': 4,
    'surprise': 5,
    'neutral': 6
}

# Veri setlerini yükleme fonksiyonları


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


# UTKFace veri setini yükleme
utkface_images, utkface_ages = load_utkface_data(
    '../datasets/utkface/crop_part1')

utkface_train_images, utkface_test_images, utkface_train_ages, utkface_test_ages = train_test_split(
    utkface_images, utkface_ages, test_size=0.2, random_state=42)

# Yaş tahmini modeli oluşturma
age_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='linear')  # Yaş tahmini için tek çıktı
])

# Modeli derleme ve eğitme
age_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
age_model.fit(utkface_train_images, utkface_train_ages, validation_data=(
    utkface_test_images, utkface_test_ages), epochs=10, batch_size=32)

# Modeli kaydetme
age_model.save('models/age_model.keras')


# FER-2013 veri setini yükleme
fer2013_images, fer2013_emotions = load_fer2013_data(
    '../datasets/fer2013/train')


fer2013_train_images, fer2013_test_images, fer2013_train_emotions, fer2013_test_emotions = train_test_split(
    fer2013_images, fer2013_emotions, test_size=0.2, random_state=42)

# Yüz ifadesi tanıma modeli oluşturma
expression_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 farklı ifade sınıfı için softmax
])

# Modeli derleme ve eğitme
expression_model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
expression_model.fit(fer2013_train_images, fer2013_train_emotions, validation_data=(
    fer2013_test_images, fer2013_test_emotions), epochs=10, batch_size=32)

# Modeli kaydetme
expression_model.save('models/expression_model.keras')
