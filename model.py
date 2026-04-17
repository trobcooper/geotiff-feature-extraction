import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras import layers, Model

IMG_SIZE = 256

# 📂 UPDATE PATH (VERY IMPORTANT)
IMAGE_PATH = "data/train/tiles/*.tif"
MASK_PATH = "data/train/masks/*.tif"

image_paths = sorted(glob(IMAGE_PATH))
mask_paths = sorted(glob(MASK_PATH))

print("Images:", len(image_paths))
print("Masks:", len(mask_paths))

# 🚀 LOAD FUNCTIONS
def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

def load_mask(path):
    mask = cv2.imread(path, 0)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = mask / 255.0
    mask = np.expand_dims(mask, axis=-1)
    return mask

# 🔥 LOAD DATA
X = np.array([load_image(p) for p in image_paths])
Y = np.array([load_mask(p) for p in mask_paths])

print("X shape:", X.shape)
print("Y shape:", Y.shape)

# 🚀 SPLIT
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
Y_train, Y_val = Y[:split], Y[split:]

# 🔥 U-NET MODEL
def build_unet():
    inputs = layers.Input((256, 256, 3))

    # Encoder
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)

    # Decoder
    u1 = layers.UpSampling2D()(c3)
    u1 = layers.Concatenate()([u1, c2])
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c1])
    c5 = layers.Conv2D(16, 3, activation='relu', padding='same')(u2)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c5)

    return Model(inputs, outputs)

model = build_unet()

# 🔥 COMPILE (IMPORTANT)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# 🚀 TRAIN
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=10,
    batch_size=8
)

# 💾 SAVE MODEL
model.save("unet_model.h5")

print("🔥 Training Completed & Model Saved")
