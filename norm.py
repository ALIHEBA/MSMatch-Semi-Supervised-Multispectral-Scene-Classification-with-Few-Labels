import numpy as np
from PIL import Image
from glob import glob
import os
import rasterio
from tqdm import tqdm
from datasets.eurosat_dataset import EurosatDataset

# dset = EurosatDataset(train=True)
# print(dset.__getitem__(0))

idx = 2

dataset = ['EuroSATallBands', 'EuroSAT_RGB', 'landcover.ai.v0'][idx]
channels = [14, 3, 3][idx]
ext = ['*.tif', '*.jpg', '*.jpg'][idx]

folders = list(os.walk(f'./data/{dataset}/'))[0][1]

images_paths = []
for folder in folders:
    if folder != 'output':
        pass
    paths = glob(os.path.join(f'./data/{dataset}/', folder, ext))
    images_paths = images_paths + paths

images_paths = np.array(images_paths)

total = np.zeros((512, 512, channels))


def normalize_to_0_to_1(img):
    img = img + np.minimum(0, np.min(img))  # move min to 0
    img = img / np.max(img)  # scale to 0 to 1
    return img

for path in tqdm(images_paths):
    if idx == 0:
        img = np.array(rasterio.open(path).read())
        img = img.transpose([2, 1, 0])
        NIR = img[:, :, 7]
        RED = img[:, :, 3]
        EPS = 1E-9
        NDVI = ((NIR - RED) / (NIR + RED + EPS)).reshape((64, 64, 1))
        img = np.concatenate((img, NDVI), axis = 2)
        img = normalize_to_0_to_1(img) * 255
    else:
        img = np.array(Image.open(path))

    total += img

mean = np.mean(total, axis = (0, 1)) / len(images_paths)

x = np.zeros(channels)

for path in tqdm(images_paths):
    if idx == 0:
        img = np.array(rasterio.open(path).read())
        img = img.transpose([2, 1, 0])
        NIR = img[:, :, 7]
        RED = img[:, :, 3]
        EPS = 1E-9
        NDVI = ((NIR - RED) / (NIR + RED + EPS)).reshape((64, 64, 1))
        img = np.concatenate((img, NDVI), axis = 2)
        img = normalize_to_0_to_1(img) * 255
    else:
        img = np.array(Image.open(path))


    x += ((img - mean) ** 2).sum(axis = (0, 1))

std = np.sqrt(x / (64 * 64 * len(images_paths)))

x = np.sum(total, axis = (0, 1)) - mean
y = x ** 2
z = y / (64 * 64 * len(images_paths))
f = np.sqrt(z)

for x in mean:
    print(f"{x:.8f}")

print('-' * 10)
for x in std:
    print(f"{x:.8f}")
# MS
# mean = [91.81513271,
# 74.47150146,
# 67.30504803,
# 58.40996290,
# 72.16675935,
# 114.36199834,
# 134.36785946,
# 129.68636479,
# 41.58579270,
# 0.78319516,
# 101.72677708,
# 62.38354261,
# 145.79203478,
# 2.26866869]
#
# std = [52.33770545,
# 41.04291450,
# 35.17643346,
# 35.09425443,
# 32.68138135,
# 39.78180825,
# 50.82429891,
# 53.91976216,
# 21.47243378,
# 0.42788569,
# 56.63376260,
# 42.26581119,
# 60.02185460,
# 7.38383386]


# RGB
# 87.81408822
# 96.97454435
# 103.98326613
# ----------
# 51.93970918
# 34.84227383
# 29.28509720