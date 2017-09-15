import pandas as pd
from PIL import Image
import os
import numpy as np
img_dir = './lfw-deepfunneled'
new_width  = 200
new_height = 200
image_arr = []
i = 0
for root, dirs, files in os.walk(img_dir):
    for name in files:
        path = os.path.join(root,name)
        img = Image.open(path)
        width, height = img.size
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        cropped = img.crop((left, top, right, bottom))
        cropped.thumbnail((64,64))
        cropped.save(os.path.join('./lfw_cropped',name))
        rgb_cropped = cropped.convert('RGB')
        image_numpy = np.array(rgb_cropped)
        image_arr.append(image_numpy)

image_arr = np.array(image_arr)
mean = np.mean(image_arr,axis=(0,1,2))
std = np.std(image_arr, axis=(0,1,2))
print(mean)
print(std)

