import json
import os
import pathlib
from os.path import isfile
import shutil
import tqdm
import yaml

import numpy as np
from src.formatting.base import BaseDatasetFormatter

class YOLODatasetFormatter(BaseDatasetFormatter):
    def __init__(self, split: list, prev_dataset_path: str, config_path: str):
        super().__init__(split, prev_dataset_path)
        self.prepare_folders()
        self.config_path = config_path

    def assign_base_folder(self):
        base_save_path = os.path.join(pathlib.Path(self.prev_dataset_path).parent.parent, 'formatted_results')
        self.formatted_folder_path = os.path.join(base_save_path, f'yolo_{self.prev_dataset_path.split("/")[-1]}')
        pathlib.Path(self.formatted_folder_path).parent.mkdir(parents=True, exist_ok=True)

    def prepare_folders(self, images='images', annot='labels'):
        self.prev_dataset_images = os.path.join(self.prev_dataset_path, 'images')
        self.prev_dataset_labels = os.path.join(self.prev_dataset_path, 'annotations')

        self.image_folder = os.path.join(self.formatted_folder_path, images)
        self.annotations_folder = os.path.join(self.formatted_folder_path, annot)
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.annotations_folder, exist_ok=True)

    def get_weights(self):
        class_weights = {}
        label_files = [os.path.join(self.prev_dataset_labels, f) for f in os.listdir(self.prev_dataset_labels) if
                       isfile(os.path.join(self.prev_dataset_labels, f))]
        for label_file in label_files:
            with open(label_file, 'r') as f:
                obj_class = [line.split()[0] for line in f if line.strip()]
                for c in obj_class:
                    if c in class_weights.keys():
                        class_weights[c] = class_weights[c] + 1
                    else:
                        class_weights[c] = 1
            f.close()

        total = sum(class_weights.values())
        normalized_data = {key: value / total for key, value in class_weights.items()}
        return dict(sorted(normalized_data.items()))

    def generate_folders(self):
        folder_types = self.generate_folder_structure()
        image_files = [f for f in os.listdir(self.prev_dataset_images) if isfile(os.path.join(self.prev_dataset_images, f))]
        label_files = [f for f in os.listdir(self.prev_dataset_labels) if isfile(os.path.join(self.prev_dataset_labels, f))]

        image_files.sort()
        label_files.sort()

        assert len(image_files) == len(label_files), 'Number of images and label files must match'

        split_indices = (np.cumsum(self.split) * len(image_files)).astype(int)[:-1]
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
                shutil.copy(os.path.join(self.prev_dataset_labels, f), os.path.join(type_label_path, f))

        print('Done creating dataset folder')

    def generate_yaml(self):
        data = {'path': self.formatted_folder_path.split('/')[-1],
                'train': 'images/train',
                'val': 'images/val'}
        if len(self.split) == 3:
            data['test'] = 'images/test'

        normalized_data = self.get_weights()
        data['nc'] = len(normalized_data)
        data['weights'] = list(normalized_data.values())
        with open(self.config_path) as f:
            json_data = json.load(f)
            data['names'] = list(json_data['id2labels'].values())
            data['names'] = data['names'][:len(data['weights'])]

        with open(f'{self.formatted_folder_path}/{self.formatted_folder_path.split("/")[-1]}.yaml', 'w') as yaml_file:
            yaml.dump(
                {k: v for k, v in data.items() if k not in ['names', 'weights', 'nc']},
                yaml_file, default_flow_style=False
            )

            yaml_file.write("\n")
            yaml_file.write(f"nc: {data['nc']}\n")
            yaml_file.write(f"names: {data['names']}\n")
            yaml_file.write(f"weights: {data['weights']}\n")

        print('Done generating yaml file')

    def generate(self):
        self.generate_folders()
        self.generate_yaml()
        print('Done full pipeline')

if __name__ == '__main__':
    base_path = pathlib.Path(__file__).resolve().parent.parent.parent
    prev_dataset_path = os.path.join(base_path, 'results/augmented_board_dataset_2025-05-07_22-21-36')
    config_path = os.path.join(base_path, 'src/configs/details_config.json')
    yolo_formatter = YOLODatasetFormatter(split=[0.6, 0.2, 0.2], prev_dataset_path=prev_dataset_path,
                                           config_path=config_path)
    yolo_formatter.generate()