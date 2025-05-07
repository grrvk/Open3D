import csv
import os
import pathlib
import numpy as np
from os.path import isfile
import shutil
import tqdm

from base import BaseDatasetFormatter


COLORS = ['red', 'green', 'yellow', 'blue']

def labels_to_csv(original_file_path):
    with open(original_file_path, 'r') as infile:
        line = infile.readline().strip().split()
        coords = list(map(float, line[1:]))

    points = [(coords[i + 1], coords[i]) for i in range(0, len(coords), 2)]
    return zip(COLORS, points)

class CorDDatasetFormatter(BaseDatasetFormatter):
    def __init__(self, split: list, prev_dataset_path: str, config_path: str):
        super().__init__(split, prev_dataset_path)
        self.config_path = config_path
        self.prepare_folders()

    def assign_base_folder(self):
        base_save_path = os.path.join(pathlib.Path(self.prev_dataset_path).parent.parent, 'formatted_results')
        self.formatted_folder_path = os.path.join(base_save_path, f'cord_{self.prev_dataset_path.split("/")[-1]}')
        pathlib.Path(self.formatted_folder_path).parent.mkdir(parents=True, exist_ok=True)

    def prepare_folders(self):
        self.prev_dataset_images = os.path.join(self.prev_dataset_path, 'images')
        self.prev_dataset_labels = os.path.join(self.prev_dataset_path, 'annotations')

    def generate_folders(self):
        folder_types = self.generate_folder_structure()
        image_files = [f for f in os.listdir(self.prev_dataset_images) if
                       isfile(os.path.join(self.prev_dataset_images, f))]
        label_files = [f for f in os.listdir(self.prev_dataset_labels) if
                       isfile(os.path.join(self.prev_dataset_labels, f))]

        image_files.sort()
        label_files.sort()

        assert len(image_files) == len(label_files), 'Number of images and label files must match'

        split_indices = (np.cumsum(self.split) * len(image_files)).astype(int)[:-1]
        print('Split indices:', split_indices)
        images_split = np.split(image_files, split_indices)
        labels_split = np.split(label_files, split_indices)

        for i, f_type in enumerate(folder_types):
            type_path = os.path.join(self.formatted_folder_path, f_type)
            os.makedirs(type_path, exist_ok=True)

            print(f'Copying {f_type} images from ', self.prev_dataset_images, ' to ', type_path)
            images_files = images_split[i]
            for f in tqdm.tqdm(images_files, ncols=len(images_split[i])):
                shutil.copy(os.path.join(self.prev_dataset_images, f), os.path.join(type_path, f))

            print(f'Copying {f_type} labels from ', self.prev_dataset_labels, ' to ', type_path)
            labels_files = labels_split[i]
            for f in tqdm.tqdm(labels_files, ncols=len(labels_split[i])):
                annotations = labels_to_csv(os.path.join(self.prev_dataset_labels, f))
                with open(f'{os.path.splitext(os.path.join(type_path, f))[0]}.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for color, (x, y) in annotations:
                        writer.writerow([color, f'{x:.6f}', f'{y:.6f}'])

        print('Done creating dataset folder')

    def generate(self):
        self.generate_folders()
        print('Done full pipeline')

if __name__ == '__main__':
    base_path = pathlib.Path(__file__).resolve().parent.parent.parent
    prev_dataset_path = os.path.join(base_path, 'results/augmented_detail_dataset_2025-05-07_12-39-26')
    config_path = os.path.join(base_path, 'src/configs/details_config.json')
    yolo_formatter = CorDDatasetFormatter(split=[0.6, 0.2, 0.2], prev_dataset_path=prev_dataset_path,
                                           config_path=config_path)
    yolo_formatter.generate()