import os
import numpy as np
import rasterio
import cv2

image_folder = "data/train/tiles"
mask_folder = "data/train/masks"

os.makedirs(mask_folder, exist_ok=True)

count = 0

for file in os.listdir(image_folder):
    if file.endswith(".tif"):
        img_path = os.path.join(image_folder, file)
        mask_path = os.path.join(mask_folder, file.replace(".tif", "_mask.tif"))

        with rasterio.open(img_path) as src:
            img = src.read([1,2,3]).transpose(1,2,0)

        # 🔥 STEP 1: normalize strongly
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 🔥 STEP 2: convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 🔥 STEP 3: edge detection (VERY IMPORTANT)
        edges = cv2.Canny(gray, 50, 150)

        # 🔥 STEP 4: dilate edges to form regions
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=1)

        # 🔥 STEP 5: fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        mask = (mask > 0).astype(np.uint8)

        with rasterio.open(
            mask_path,
            "w",
            driver="GTiff",
            height=mask.shape[0],
            width=mask.shape[1],
            count=1,
            dtype="uint8",
        ) as dst:
            dst.write(mask, 1)

        count += 1
        print(f"Mask {count}")

print("🔥 FINAL MASKS:", count)
