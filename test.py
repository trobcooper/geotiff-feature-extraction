import rasterio
import matplotlib.pyplot as plt

with rasterio.open("data/train/tiles/tile_0.tif") as src:
    img = src.read([1,2,3]).transpose(1,2,0)

plt.imshow(img)
plt.show()
