import os
import pathlib

import cv2
import albumentations as A
import numpy as np
from tqdm import tqdm


class Augmentations:
    def __init__(self, general_dir, transforms):
        self.general_dir = general_dir
        self.subdir_names = [p.name for p in pathlib.Path(self.general_dir).iterdir() if p.is_dir()]
        self.subdir_names.sort()
        self.augmented_dir = os.path.join(pathlib.Path(self.general_dir).resolve().parent,
                                          f'augmented_{pathlib.Path(self.general_dir).name}')
        self.prepare_folders()
        self.transforms = transforms

    def prepare_folders(self):
        pathlib.Path(self.general_dir).parent.mkdir(parents=True, exist_ok=True)

        self.aug_annotations_folder = os.path.join(self.augmented_dir, self.subdir_names[0])
        self.aug_image_folder = os.path.join(self.augmented_dir, self.subdir_names[1])
        self.aug_mask_folder = os.path.join(self.augmented_dir, self.subdir_names[2])
        os.makedirs(self.aug_image_folder, exist_ok=True)
        os.makedirs(self.aug_mask_folder, exist_ok=True)
        os.makedirs(self.aug_annotations_folder, exist_ok=True)

    @staticmethod
    def read_polygon_label(label_path):
        polygons = []
        with open(label_path, "r") as f:
            for line in f:
                values = line.strip().split()
                cls = int(float(values[0]))
                coords = np.array(list(map(float, values[1:])), dtype=np.float32)

                if np.all((coords >= 0) & (coords <= 1)):
                    raise Exception('Coordinates must be denormalized for albumenations to work')
                coords = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
                polygons.append((cls, coords))
        return polygons

    @staticmethod
    def write_polygon_label(label_path, polygons, img_width, img_height, normalize):
        with open(label_path, "w") as f:
            for cls, points in polygons:
                normalized_points = np.array(points.copy())
                normalized_points = normalized_points.reshape(4, 2)
                if normalize:
                    normalized_points[:, 0] /= img_width
                    normalized_points[:, 1] /= img_height
                flat_coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in normalized_points])
                f.write(f"{cls} {flat_coords}\n")

    def get_full_paths(self, mode):
        directory = self.general_dir if mode == "general" else self.augmented_dir
        ann_dir = os.path.join(directory, self.subdir_names[0])
        img_dir = os.path.join(directory, self.subdir_names[1])
        mask_dir = os.path.join(directory, self.subdir_names[2])
        return ann_dir, img_dir, mask_dir

    def augment(self, normalize):
        ann_dir, img_dir, mask_dir = self.get_full_paths("general")
        ann_aug, img_aug, mask_aug = self.get_full_paths("augmented")
        for filename in tqdm(os.listdir(img_dir)):
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            name, ext = os.path.splitext(filename)

            img_path = os.path.join(img_dir, filename)
            mask_path = os.path.join(mask_dir, name + ext)
            label_path = os.path.join(ann_dir, name + '.txt')

            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            h, w = image.shape[:2]

            polygons = self.read_polygon_label(label_path)
            if not polygons:
                continue

            keypoints = []
            classes = []
            for (cls, pts) in polygons:
                keypoints.extend(pts)
                classes.append(cls)

            aug = self.transforms(image=image, mask=mask, keypoints=keypoints)
            aug_img = aug['image']
            aug_mask = aug['mask']
            aug_kps = aug['keypoints']

            # Group back to polygons of 4 points each
            new_polygons = []
            for i in range(0, len(aug_kps), 4):
                ps = aug_kps[i:i + 4]
                cls = classes[i//4]
                if all(0 <= x < w and 0 <= y < h for x, y in ps):
                    new_polygons.append((cls, ps))

            if not new_polygons:
                continue  # Skip if all bboxes are out of image

            out_img_path = os.path.join(img_aug, filename)
            out_mask_path = os.path.join(mask_aug, name + ext)
            out_label_path = os.path.join(ann_aug, name + '.txt')

            cv2.imwrite(out_img_path, aug_img)
            cv2.imwrite(out_mask_path, aug_mask)
            self.write_polygon_label(out_label_path, new_polygons, w, h, normalize=normalize)

if __name__ == '__main__':
    transform = A.Compose([
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.9),
        A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, shear=(-5, 5), p=0.9),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    base_path = pathlib.Path(__file__).resolve().parent.parent.parent
    augmentor = Augmentations(general_dir=os.path.join(base_path, 'results/board_dataset_2025-05-07_22-21-36'),
                              transforms=transform)
    augmentor.augment(normalize=False)