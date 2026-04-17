import os
import rasterio
from rasterio.windows import Window
import numpy as np

# 📂 PATHS
input_folder = "data/train/images"
output_folder = "data/train/tiles"

os.makedirs(output_folder, exist_ok=True)

tile_size = 256
step = 256
max_tiles_per_image = 60   # 🔥 stricter (quality > quantity)

count = 0

for file in os.listdir(input_folder):
    if file.endswith(".tif"):
        path = os.path.join(input_folder, file)
        print(f"\n📂 Processing: {file}")

        with rasterio.open(path) as src:
            height, width = src.height, src.width
            tiles_per_image = 0

            for i in range(0, height - tile_size, step):
                for j in range(0, width - tile_size, step):

                    if tiles_per_image >= max_tiles_per_image:
                        break

                    window = Window(j, i, tile_size, tile_size)
                    transform = src.window_transform(window)

                    # ⚡ FAST READ
                    tile = src.read([1, 2, 3], window=window)

                    tile_np = tile.astype(np.float32)

                    # =========================
                    # 🔥 ULTRA STRICT FILTERS
                    # =========================

                    # 1. BLACK PIXEL RATIO (very strict)
                    black_ratio = np.sum(tile_np < 10) / tile_np.size
                    if black_ratio > 0.15:
                        continue

                    # 2. EDGE BLACK CHECK (removes triangles)
                    edges = np.concatenate([
                        tile_np[:, 0, :],      # left
                        tile_np[:, -1, :],     # right
                        tile_np[:, :, 0],      # top
                        tile_np[:, :, -1],     # bottom
                    ])
                    edge_black_ratio = np.sum(edges < 10) / edges.size
                    if edge_black_ratio > 0.05:
                        continue

                    # 3. WHITE FILTER
                    mean_val = np.mean(tile_np)
                    if mean_val > 230:
                        continue

                    # 4. LOW VARIANCE (boring tiles)
                    if np.std(tile_np) < 15:
                        continue

                    # =========================

                    out_path = os.path.join(output_folder, f"tile_{count}.tif")

                    with rasterio.open(
                        out_path,
                        "w",
                        driver="GTiff",
                        height=tile.shape[1],
                        width=tile.shape[2],
                        count=tile.shape[0],
                        dtype=tile.dtype,
                        transform=transform,
                        crs=src.crs,
                    ) as dst:
                        dst.write(tile)

                    count += 1
                    tiles_per_image += 1

                    print(f"   ✅ Tile {tiles_per_image}")

                if tiles_per_image >= max_tiles_per_image:
                    break

print(f"\n🔥 FINAL CLEAN TILES: {count}")
