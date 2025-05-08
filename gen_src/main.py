import os
import cv2
import pathlib
from pathlib import Path
import albumentations as A

from gen_src.formatting.yolo import YOLODatasetFormatter
from gen_src.generation.augmentation import Augmentations
from gen_src.generation.structure import BoardDatasetGenerator


def main():
    base_path = pathlib.Path(__file__).resolve().parent.parent

    generator = BoardDatasetGenerator(asset_folder='texture')
    generator.generate(amount=4, cropped=False)

    generator_path = f"{Path(generator.general_folder).parent.name}/{Path(generator.general_folder).name}"

    transform = A.Compose([
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.9),
        A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, shear=(-5, 5), p=0.9),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    augmentor = Augmentations(general_dir=os.path.join(base_path, generator_path),
                              transforms=transform)
    augmentor.augment(normalize=True)

    augmentor_path = f"{Path(augmentor.augmented_dir).parent.name}/{Path(augmentor.augmented_dir).name}"

    prev_dataset_path = os.path.join(base_path, augmentor_path)
    config_path = os.path.join(base_path, 'src/configs/details_config.json')
    yolo_formatter = YOLODatasetFormatter(split=[0.6, 0.2, 0.2], prev_dataset_path=prev_dataset_path,
                                           config_path=config_path)
    yolo_formatter.generate()

    print('Generation complete.')

if __name__ == "__main__":
    main()