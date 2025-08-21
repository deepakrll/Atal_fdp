!pip install tensorflow opencv-python-headless
from google.colab import drive
drive.mount('/content/drive')

import os
import shutil

IMG_SIZE = 128
BATCH_SIZE = 1  # Changed from 32 to 1 to work with small dataset
DATASET_PATH = '/content/drive/MyDrive/FoodSpoilageDataset'
TEMP_TRAIN_DIR = '/content/train_only'

# Recreate fresh/spoiled folders without test/
if os.path.exists(TEMP_TRAIN_DIR):
    shutil.rmtree(TEMP_TRAIN_DIR)

os.makedirs(os.path.join(TEMP_TRAIN_DIR, 'fresh'), exist_ok=True)
os.makedirs(os.path.join(TEMP_TRAIN_DIR, 'spoiled'), exist_ok=True)

# Copy only .jpg/.png images (4 each)
for cls in ['fresh', 'spoiled']:
    src = os.path.join(DATASET_PATH, cls)
    dst = os.path.join(TEMP_TRAIN_DIR, cls)
    for fname in os.listdir(src):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            shutil.copy(os.path.join(src, fname), dst)

# Check copied images
print("Fresh:", len(os.listdir(TEMP_TRAIN_DIR + "/fresh")))
print("Spoiled:", len(os.listdir(TEMP_TRAIN_DIR + "/spoiled")))

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    TEMP_TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

EPOCHS = 5  # keep small due to tiny dataset

history = model.fit(
    train_data,
    epochs=EPOCHS
)

import cv2
import numpy as np
import matplotlib.pyplot as plt

TEST_PATH = os.path.join(DATASET_PATH, 'test')
class_names = list(train_data.class_indices.keys())

def predict_test_images():
    for fname in os.listdir(TEST_PATH):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(TEST_PATH, fname)
            img = cv2.imread(path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

            pred = model.predict(img)
            label = class_names[np.argmax(pred)]

            plt.imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
            plt.title(f"{fname} → Predicted: {label}")
            plt.axis('off')
            plt.show()

predict_test_images()

