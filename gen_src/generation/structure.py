import os
import pathlib
import random
from datetime import datetime

import tqdm

import cv2
import numpy as np
import open3d as o3d

from src.layout.scene import Scene


class DatasetGenerator:
    def __init__(self, general_folder, asset_folder):
        general_folder = f'{general_folder}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        base_dir = os.path.join(pathlib.Path(__file__).resolve().parent.parent.parent, 'results')
        self.general_folder = os.path.join(base_dir, general_folder)
        self.asset_folder = asset_folder

        self.annotations_folder = None
        self.mask_folder = None
        self.image_folder = None
        self.prepare_folders()

    def prepare_folders(self, images='images', masks='masks', annot='annotations'):
        pathlib.Path(self.general_folder).parent.mkdir(parents=True, exist_ok=True)

        self.image_folder = os.path.join(self.general_folder, images)
        self.mask_folder = os.path.join(self.general_folder, masks)
        self.annotations_folder = os.path.join(self.general_folder, annot)
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.mask_folder, exist_ok=True)
        os.makedirs(self.annotations_folder, exist_ok=True)

    def prepare_assets(self):
        asset_images = []
        if self.asset_folder is None:
            return asset_images
        for e in ['*.jpg', '*.png', '*.jpeg']:
            asset_images.extend(list(pathlib.Path(self.asset_folder).glob(e)))
        return asset_images

    def get_folder_structure(self):
        print(f'Dataset structure:\n- {self.general_folder}'
              f' - {self.image_folder}'
              f' - {self.mask_folder}'
              f' - {self.annotations_folder}'
              f'Asset/texture folder: {self.asset_folder}')

    @staticmethod
    def _set_view(window, render_options='configs/render_option.json'):
        base_dir = pathlib.Path(__file__).resolve().parent.parent
        render_options = os.path.join(base_dir, render_options)
        view_ctl = window.get_view_control()
        view_ctl.rotate(0.0, 5.8178*90)

        zoom_rate = round(random.uniform(0.5, 0.8), 1)
        view_ctl.set_zoom(zoom_rate)

        render_ctl = window.get_render_option()
        render_ctl.load_from_json(render_options)

        window.update_renderer()
        return window

    @staticmethod
    def _create_mask(image_path, lower_color=[0, 0, 200], upper_color=[180, 30, 255]):
        image = cv2.imread(image_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lower_color), np.array(upper_color))

        mask_inv = cv2.bitwise_not(mask)
        kernel = np.ones((10, 10), np.uint8)
        mask_filled = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)
        return mask_filled, mask_inv, mask

    def _form_mask(self, temp_image, assets_list):
        board_img = cv2.imread(temp_image)
        mask_filled, mask_inv, mask = self._create_mask(temp_image)
        board_foreground = cv2.bitwise_and(board_img, board_img, mask=mask_inv)

        if len(assets_list) > 0:
            random_wood_texture = cv2.imread(str(random.choice(assets_list)))
            wood_texture = cv2.resize(random_wood_texture, (board_img.shape[1], board_img.shape[0]))
            wood_background = cv2.bitwise_and(wood_texture, wood_texture, mask=mask)
            result = cv2.add(board_foreground, wood_background)
            return mask_filled, result

        return mask_filled, board_foreground

    @staticmethod
    def crop(image, mask):
        y_indices, x_indices = np.where(mask > 0)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]
        return cropped_image, cropped_mask

    @staticmethod
    def normalize_annots(annotation, width, height):
        normalized_annots = [annotation[0]]
        for i, val in enumerate(annotation[1:], start=1):
            if i % 2 == 1:
                normalized_annots.append(val / width)
            else:
                normalized_annots.append(val / height)
        return normalized_annots

    def create_mask(self, image_path):
        pass

    def form_mask(self, temp_image, assets_list):
        pass

    def generate(self, *args):
        pass

class DetailDatasetGenerator(DatasetGenerator):
    def __init__(self, general_folder='detail_dataset', asset_folder='assets'):
        super().__init__(general_folder, asset_folder)
        self.window = o3d.visualization.Visualizer()
        self.window.create_window("Detail Dataset generation view")

        self.size = (14, 17)
        self.background_detail_labels = ['o']

    def _instance_mask_processing(self, board_matrix, binary_mask, big_details, normalized):
        mask_height, mask_width = binary_mask.shape

        y_indices, x_indices = np.where(binary_mask > 0)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        board_height = y_max - y_min + 1
        board_width = x_max - x_min + 1

        rows, cols = self.size[0], self.size[1]
        cell_height = board_height // rows
        cell_width = board_width // cols

        instance_mask = np.zeros_like(binary_mask)

        instance = 1
        annotations = []

        for i in range(rows):
            j = 0
            while j < cols:
                if np.any(binary_mask[y_min + i * cell_height: y_min + (i + 1) * cell_height,
                          x_min + j * cell_width: x_min + (j + 1) * cell_width] > 0):

                    x_min_b, y_min_b = x_min + j * cell_width, y_min + i * cell_height
                    x_max_b, y_max_b = x_min + (j + 1) * cell_width, y_min + (i + 1) * cell_height
                    if str(board_matrix[i, j]) in big_details:
                        x_max_b, y_max_b = x_min + (j + 2) * cell_width, y_min + (i + 1) * cell_height
                        if str(board_matrix[i, j]) == 'CL':
                            x_min_b, y_min_b = x_min + (j - 1) * cell_width, y_min + (i - 2) * cell_height
                        else:
                            x_min_b, y_min_b = x_min + (j - 1) * cell_width, y_min + (i - 1) * cell_height


                    square_instance_index = 0 if str(board_matrix[i, j]) in self.background_detail_labels else instance
                    cv2.rectangle(instance_mask, (x_min_b, y_min_b), (x_max_b, y_max_b), square_instance_index, -1)

                    if str(board_matrix[i, j]) not in self.background_detail_labels:
                        annotation_row_values = [square_instance_index, x_min_b, y_min_b, x_min_b, y_max_b, x_max_b, y_max_b,
                                                 x_max_b, y_min_b]
                        if normalized: annotation_row_values = self.normalize_annots(annotation=annotation_row_values,
                                                                                     width=mask_width,
                                                                                     height=mask_height)
                        annotations.append(list(map(str, annotation_row_values)))
                        instance += 1
                    if str(board_matrix[i, j]) in big_details: j += 1

                j += 1

        return instance_mask, annotations

    def _class_mask_processing(self, board_matrix, binary_mask, labels2id, big_details, normalized):
        mask_height, mask_width = binary_mask.shape

        y_indices, x_indices = np.where(binary_mask > 0)
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        board_height = y_max - y_min + 1
        board_width = x_max - x_min + 1

        rows, cols = self.size[0], self.size[1]
        cell_height = board_height // rows
        cell_width = board_width // cols

        class_mask = np.zeros_like(binary_mask)
        annotations = []
        for i in range(rows):
            j = 0
            while j < cols:
                if np.any(binary_mask[y_min + i * cell_height: y_min + (i + 1) * cell_height,
                          x_min + j * cell_width: x_min + (j + 1) * cell_width] > 0):
                    x_min_b, y_min_b = x_min + j * cell_width, y_min + i * cell_height
                    x_max_b, y_max_b = x_min + (j + 1) * cell_width, y_min + (i + 1) * cell_height
                    if str(board_matrix[i, j]) in big_details:
                            x_max_b, y_max_b = x_min + (j + 2) * cell_width, y_min + (i + 1) * cell_height
                            if str(board_matrix[i, j]) == 'CL':
                                x_min_b, y_min_b = x_min + (j - 1) * cell_width, y_min + (i - 2) * cell_height
                            else:
                                x_min_b, y_min_b = x_min + (j - 1) * cell_width, y_min + (i - 1) * cell_height

                    detail_class = 0 if str(board_matrix[i, j]) in self.background_detail_labels else \
                                                                    labels2id.get(str(board_matrix[i, j]))
                    cv2.rectangle(class_mask, (x_min_b, y_min_b), (x_max_b, y_max_b), detail_class, -1)

                    if str(board_matrix[i, j]) not in self.background_detail_labels:
                        annotation_row_values = [detail_class - 1, x_min_b, y_min_b, x_min_b, y_max_b, x_max_b, y_max_b,
                                                 x_max_b, y_min_b]
                        if normalized: annotation_row_values = self.normalize_annots(annotation=annotation_row_values,
                                                                                     width=mask_width,
                                                                                     height=mask_height)
                        annotations.append(list(map(str, annotation_row_values)))
                    if str(board_matrix[i, j]) in big_details: j += 1
                j += 1

        return class_mask, annotations

    def _mask_generation(self, amount, mode, assets_list, cropped, normalized, empty_cell_range=(0.55, 0.9),
                               temporary_im = 'temporary.png', **kwargs):
        scene = Scene()
        path_amount_range = kwargs.get('path_amount_range', None)
        generate_big_shapes = kwargs.get('generate_big_shapes', True)
        labels2id = scene.labels2id
        big_details = scene.bigShapes
        for j in tqdm.tqdm(range(amount)):
            empty_cell_rate_value = round(random.uniform(empty_cell_range[0], empty_cell_range[1]), 1)
            path_num = random.randint(*path_amount_range) if path_amount_range is not None else None
            scene.generateBoard(num=path_num, empty_rate=empty_cell_rate_value,
                                generate_big_shapes=generate_big_shapes)

            scene.fillObjects()
            geoms = scene.getGeoms()

            for i, g in enumerate(geoms):
                self.window.add_geometry(geometry=g)

            window = self._set_view(self.window)
            window.poll_events()
            window.capture_screen_image(temporary_im, do_render=True)

            mask_filled, image = self._form_mask(temp_image=temporary_im, assets_list=assets_list)
            if cropped: image, mask_filled = self.crop(image=image, mask=mask_filled)

            if mode == 'instance':
                instance_mask, annotations = self._instance_mask_processing(board_matrix=scene.getBoard().board_matrix,
                                                                            binary_mask=mask_filled,
                                                                            big_details=big_details,
                                                                            normalized=normalized)
            elif mode == 'class':
                instance_mask, annotations = self._class_mask_processing(board_matrix=scene.getBoard().board_matrix,
                                                                         binary_mask=mask_filled, normalized=normalized,
                                                                         labels2id=labels2id, big_details=big_details)
            else:
                raise Exception(f'Invalid mode: {mode}. Please choose from "instance" or "class"')

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cv2.imwrite(f'{self.mask_folder}/{j}_{timestamp}.png', instance_mask)
            cv2.imwrite(f'{self.image_folder}/{j}_{timestamp}.png', image)

            with open(f"{self.annotations_folder}/{j}_{timestamp}.txt", "w") as txt_file:
                for line in annotations:
                    txt_file.write(" ".join(line) + "\n")

            for i, g in enumerate(geoms):
                window.remove_geometry(geometry=g)

        self.window.destroy_window()
        os.remove(temporary_im)
        return True


    def generate(self, amount, mode, cropped=False, normalized=False, **kwargs):
        print(f'Generating {amount} images, {mode} mode, cropped flag: {cropped}, normalized flag: {normalized}\n'
              f'Additional arguments: {kwargs}')
        assets_list = self.prepare_assets()
        print(f'Assets loaded from {self.asset_folder}. Count: {len(assets_list)}\nStarting generation...')
        self._mask_generation(amount=amount, mode=mode, assets_list=assets_list, cropped=cropped,
                              normalized=normalized, **kwargs)
        print(f'Finished generating {amount} images')

class BoardDatasetGenerator(DatasetGenerator):
    def __init__(self, general_folder='board_dataset', asset_folder='assets'):
        super().__init__(general_folder, asset_folder)
        self.window = o3d.visualization.Visualizer()
        self.window.create_window("Board Dataset generation view")

        self.board_class_id = 1

    @staticmethod
    def _get_corners(image, normalized):
        image_height, image_width = image.shape
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        corners = []
        for cnt in contours:
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if len(approx) == 4:
                corners = approx.reshape(4, 2).astype(float)
                if normalized:
                    corners[:, 0] /= image_width
                    corners[:, 1] /= image_height
                break
        return corners

    def _mask_generation(self, amount, assets_list, cropped, normalized, empty_cell_range=(0.55, 0.9),
                               temporary_im = 'temporary.png', **kwargs):
        scene = Scene()
        path_amount_range = kwargs.get('path_amount_range', None)
        generate_big_shapes = kwargs.get('generate_big_shapes', True)

        for j in tqdm.tqdm(range(amount)):
            empty_cell_rate_value = round(random.uniform(empty_cell_range[0], empty_cell_range[1]), 1)
            path_num = random.randint(*path_amount_range) if path_amount_range is not None else None
            scene.generateBoard(num=path_num, empty_rate=empty_cell_rate_value,
                                generate_big_shapes=generate_big_shapes)
            scene.fillObjects()
            geoms = scene.getGeoms()

            for i, g in enumerate(geoms):
                self.window.add_geometry(geometry=g)

            window = self._set_view(self.window)
            window.poll_events()
            window.capture_screen_image(temporary_im, do_render=True)

            mask_filled, image = self._form_mask(temp_image=temporary_im, assets_list=assets_list)
            if cropped: self.crop(image=image, mask=mask_filled)

            annotations = np.append([self.board_class_id], self._get_corners(mask_filled, normalized))
            annotations = annotations.astype(str)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cv2.imwrite(f'{self.mask_folder}/{j}_{timestamp}.png', mask_filled)
            cv2.imwrite(f'{self.image_folder}/{j}_{timestamp}.png', image)

            with open(f"{self.annotations_folder}/{j}_{timestamp}.txt", "w") as txt_file:
                txt_file.write(" ".join(annotations) + "\n")

            for i, g in enumerate(geoms):
                window.remove_geometry(geometry=g)

        self.window.destroy_window()
        os.remove("temporary.png")
        return True

    def generate(self, amount, cropped=False, normalized=False, **kwargs):
        print(f'Generating {amount} images, cropped flag: {cropped}, normalized flag: {normalized}\n'
              f'Additional arguments: {kwargs}')
        assets_list = self.prepare_assets()
        print(f'Assets loaded from {self.asset_folder}. Count: {len(assets_list)}\nStarting generation...')
        self._mask_generation(amount=amount, assets_list=assets_list, cropped=cropped,
                              normalized=normalized, **kwargs)
        print(f'Finished generating {amount} images')


if __name__ == "__main__":
    generator = BoardDatasetGenerator(asset_folder='/Users/vika/Desktop/texture')
    generator.generate(amount=4, path_amount_range=(1, 3), generate_big_shapes=False)
