import pathlib
import cv2
import os
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def load_and_parse_annotation(annotation_file, img_shape):
    height, width = img_shape[:2]
    polygons = []

    with open(annotation_file, 'r') as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            coords = parts[1:]
            if max(coords) > 1:
                coords[::2] = [x / width for x in coords[::2]]
                coords[1::2] = [y / height for y in coords[1::2]]
            scaled_coords = np.array([[int(x * width), int(y * height)] for x, y in zip(coords[::2], coords[1::2])],
                                     dtype=np.int32)
            polygons.append((class_id, scaled_coords))

    return polygons


def create_mask_from_annotations(polygons, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for class_id, polygon in polygons:
        cv2.fillPoly(mask, [polygon], color=class_id)
    return mask


def display_entry_with_annotations(image_path, mask_path, annotation_path):
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    polygons = load_and_parse_annotation(annotation_path, img.shape)
    ann_mask = create_mask_from_annotations(polygons, img.shape)

    fig, arr = plt.subplots(1, 3, figsize=(18, 6))

    arr[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    arr[0].set_title('Original Image')

    cmap = plt.get_cmap('nipy_spectral')
    norm = mcolors.Normalize(vmin=0, vmax=ann_mask.max())

    arr[1].imshow(mask,  cmap=cmap, norm=norm, interpolation='nearest')
    arr[1].set_title('True Mask')

    arr[2].imshow(ann_mask, cmap=cmap, norm=norm, interpolation='nearest')
    arr[2].set_title('Mask from Annotations (Color)')

    for ax in arr:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_data(dataset, images_f='images', masks_f='masks', ann_f='annotations'):
    images_path = os.path.join(dataset, images_f)
    masks_path = os.path.join(dataset, masks_f)
    ann_path = os.path.join(dataset, ann_f)

    image_files = sorted([os.path.join(images_path, f) for f in os.listdir(images_path)])
    masks_files = sorted([os.path.join(masks_path, f) for f in os.listdir(masks_path)])
    ann_files = sorted([os.path.join(ann_path, f) for f in os.listdir(ann_path)])

    index = random.randint(0, len(image_files) - 1)
    print(f'Plotting data for image {index}')

    random_image = image_files[index]
    random_mask = masks_files[index]
    random_ann = ann_files[index]

    display_entry_with_annotations(random_image, random_mask, random_ann)

if __name__ == '__main__':
    base_path = pathlib.Path(__file__).resolve().parent.parent
    dataset_path = os.path.join(base_path, 'results/augmented_board_dataset_2025-05-07_22-21-36')
    plot_data(dataset=dataset_path)