import os
from os.path import isfile
import shutil
import tqdm
import yaml

import numpy as np

class DatasetFormatting:
    def __init__(self, split: list, dataset_path: str):
        self.split = split
        self.dataset_path = dataset_path
        self.prev_dataset_images = os.path.join(dataset_path, 'images')
        self.prev_dataset_labels = os.path.join(dataset_path, 'annotations')

        self.general_folder = 'yolo_dataset'
        self.images = os.path.join(self.general_folder, 'images')
        self.labels = os.path.join(self.general_folder, 'labels')

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
        return normalized_data


    def generate_general_structure(self):
        assert 2 <= len(self.split) <= 3, 'Split must be into 2 or 3 folders (train, val, test(optional))'
        assert sum(self.split) == 1, 'Split rates must summ to 1'

        os.makedirs(self.general_folder, exist_ok=True)
        os.makedirs(self.images, exist_ok=True)
        os.makedirs(self.labels, exist_ok=True)

        if len(self.split) == 2:
            folder_types = ['train', 'val']
        else:
            folder_types = ['train', 'val', 'test']

        return folder_types

    def generate_folders(self, folder_types: list):
        image_files = [f for f in os.listdir(self.prev_dataset_images) if isfile(os.path.join(self.prev_dataset_images, f))]
        label_files = [f for f in os.listdir(self.prev_dataset_labels) if isfile(os.path.join(self.prev_dataset_labels, f))]

        assert len(image_files) == len(label_files), 'Number of images and label files must match'

        split_indices = (np.cumsum(self.split) * len(image_files)).astype(int)[:-1]
        images_split = np.split(image_files, split_indices)
        labels_split = np.split(label_files, split_indices)

        for i, f_type in enumerate(folder_types):
            type_image_path = os.path.join(self.images, f_type)
            type_label_path = os.path.join(self.labels, f_type)
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
        data = {'path': self.general_folder,
                'train': 'images/train',
                'val': 'images/val'}
        if len(self.split) == 3:
            data['test'] = 'images/test'

        normalized_data = self.get_weights()
        data['nc'] = len(normalized_data)
        data['names'] = list(normalized_data.keys())
        data['weights'] = list(normalized_data.values())

        with open(f'{self.general_folder}.yaml', 'w') as yaml_file:
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
        folder_types = self.generate_general_structure()
        self.generate_folders(folder_types)
        self.generate_yaml()
        print('Done full pipeline')



def main():
    dataset_formatting = DatasetFormatting(split=[0.8, 0.2], dataset_path='./dataset')
    dataset_formatting.generate()

if __name__ == '__main__':
    main()
