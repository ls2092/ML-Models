import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set base folder
base_folder = 'archive'

# Load CSVs
train_df = pd.read_csv(os.path.join(base_folder, 'Train.csv'))
test_df = pd.read_csv(os.path.join(base_folder, 'Test.csv'))

# Target image size
img_size = 32

# Load training images and labels
X_train = []
y_train = []

print("Loading training images...")
for index, row in train_df.iterrows():
    class_id = str(row['ClassId'])
    image_path = os.path.join(base_folder, 'Train', class_id, os.path.basename(row['Path']))
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (img_size, img_size))
        X_train.append(img)
        y_train.append(row['ClassId'])
    else:
        print(f"Missing: {image_path}")

X_train = np.array(X_train)
y_train = np.array(y_train)

# Load testing images and labels
X_test = []
y_test = []

print("Loading test images...")
for index, row in test_df.iterrows():
    image_path = os.path.join(base_folder, 'Test', os.path.basename(row['Path']))
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (img_size, img_size))
        X_test.append(img)
        y_test.append(row['ClassId'])
    else:
        print(f"Missing: {image_path}")

X_test = np.array(X_test)
y_test = np.array(y_test)

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode labels
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Data Augmentation
datagen = ImageDataGenerator(rotation_range=10,
                             zoom_range=0.1,
                             width_shift_range=0.1,
                             height_shift_range=0.1)
datagen.fit(X_train)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                    epochs=10,
                    validation_data=(X_test, y_test))

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc:.4f}")

# Classification Report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))

# Confusion Matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=False, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
