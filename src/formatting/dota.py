import json
import pathlib
from os.path import isfile
import shutil
import tqdm

import numpy as np
from base import BaseDatasetFormatter
import os


def labels_to_dota(original_file_path, id2labels):
    with open(original_file_path, 'r') as f:
        lines = f.readlines()

    dota_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 9:
            continue

        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        class_name = id2labels[str(class_id)]
        difficulty = 0 if class_id >= 5 else 1

        dota_line = f"{int(coords[0])} {int(coords[1])} {int(coords[2])} {int(coords[3])} "
        dota_line += f"{int(coords[4])} {int(coords[5])} {int(coords[6])} {int(coords[7])} "
        dota_line += f"{class_name} {difficulty}\n"
        dota_lines.append(dota_line)

    return dota_lines


class DOTADatasetFormatter(BaseDatasetFormatter):
    def __init__(self, split: list, prev_dataset_path: str, config_path: str):
        super().__init__(split, prev_dataset_path)
        self.config_path = config_path
        self.prepare_folders()

    def assign_base_folder(self):
        base_save_path = os.path.join(pathlib.Path(self.prev_dataset_path).parent.parent, 'formatted_results')
        self.formatted_folder_path = os.path.join(base_save_path, f'dota_{self.prev_dataset_path.split("/")[-1]}')
        pathlib.Path(self.formatted_folder_path).parent.mkdir(parents=True, exist_ok=True)


    def prepare_folders(self, images='images', annot='labels'):
        self.prev_dataset_images = os.path.join(self.prev_dataset_path, 'images')
        self.prev_dataset_labels = os.path.join(self.prev_dataset_path, 'annotations')

        self.image_folder = os.path.join(self.formatted_folder_path, images)
        self.annotations_folder = os.path.join(self.formatted_folder_path, annot)
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.annotations_folder, exist_ok=True)

    def generate_folders(self):
        folder_types = self.generate_folder_structure()
        image_files = [f for f in os.listdir(self.prev_dataset_images) if isfile(os.path.join(self.prev_dataset_images, f))]
        label_files = [f for f in os.listdir(self.prev_dataset_labels) if isfile(os.path.join(self.prev_dataset_labels, f))]

        image_files.sort()
        label_files.sort()

        with open(self.config_path) as f:
            json_data = json.load(f)
            id2labels = json_data['id2labels']

        assert len(image_files) == len(label_files), 'Number of images and label files must match'

        split_indices = (np.cumsum(self.split) * len(label_files)).astype(int)[:-1]
        print('Split indices:', split_indices)
        images_split = np.split(image_files, split_indices)
        labels_split = np.split(label_files, split_indices)

        for i, f_type in enumerate(folder_types):
            type_image_path = os.path.join(self.image_folder, f_type)
            type_label_path = os.path.join(self.annotations_folder, f_type)
            os.makedirs(type_image_path, exist_ok=True)
            os.makedirs(type_label_path, exist_ok=True)

            print(f'Copying {f_type} images from ', self.prev_dataset_images, ' to ', type_image_path)
            images_files = images_split[i]
            for f in tqdm.tqdm(images_files, ncols=len(images_split[i])):
                shutil.copy(os.path.join(self.prev_dataset_images, f), os.path.join(type_image_path, f))

            print(f'Copying {f_type} labels from ', self.prev_dataset_labels, ' to ', type_label_path)
            labels_files = labels_split[i]
            for f in tqdm.tqdm(labels_files, ncols=len(labels_split[i])):
                annotations = labels_to_dota(os.path.join(self.prev_dataset_labels, f), id2labels)
                with open(os.path.join(type_label_path, f), "w") as txt_file:
                    for line in annotations:
                        txt_file.write(" ".join(line) + "\n")

        print('Done creating dataset folder')

    def generate(self):
        self.generate_folders()
        print('Done full pipeline')


if __name__ == '__main__':
    base_path = pathlib.Path(__file__).resolve().parent.parent.parent
    prev_dataset_path = os.path.join(base_path, 'results/augmented_detail_dataset_2025-05-07_12-39-26')
    config_path = os.path.join(base_path, 'src/configs/details_config.json')
    yolo_formatter = DOTADatasetFormatter(split=[0.6, 0.2, 0.2], prev_dataset_path=prev_dataset_path,
                                           config_path=config_path)
    yolo_formatter.generate()