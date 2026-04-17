import rasterio
import matplotlib.pyplot as plt

with rasterio.open("data/train/masks/tile_0_mask.tif") as src:
    mask = src.read(1)

print(mask.min(), mask.max())  # should show 0 and 1

plt.imshow(mask, cmap='gray')
plt.show()
