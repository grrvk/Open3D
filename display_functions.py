import random

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def display_entry(image_path, mask_path):
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    fig, arr = plt.subplots(1, 2, figsize=(12, 6))

    arr[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    arr[0].set_title('Original Image')

    arr[1].imshow(mask)
    arr[1].set_title('Mask')

    plt.show()

def main(images_path='dataset/images', masks_path='dataset/masks'):
    image_files = [os.path.join(images_path, f) for f in os.listdir(images_path)]
    masks_files = [os.path.join(masks_path, f) for f in os.listdir(masks_path)]

    index = random.randint(0, len(image_files) - 1)
    print(index)

    random_image = image_files[index]
    random_mask = masks_files[index]
    display_entry(random_image, random_mask)

if __name__ == '__main__':
    main()