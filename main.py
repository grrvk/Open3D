import pathlib
import random
import tqdm

import numpy as np
import cv2
import os
import open3d as o3d
import open3d.visualization as vis

from scene.generation import Scene


def casual_run(pathnum):
    def run(pathnum, empty_rate=0.7, generate_big_shapes=True):
        app = vis.gui.Application.instance
        window = o3d.visualization.O3DVisualizer("ui")

        window.setup_camera(field_of_view=10, center=[0, 0, 0], eye=[0, 30, 0], up=[0, 60, 0])

        scene = Scene()

        scene.generateBoard(num=pathnum, empty_rate=empty_rate, generate_big_shapes=generate_big_shapes)
        objects = scene.fillObjects()
        geoms = scene.getGeoms()

        for i, g in enumerate(geoms):
            window.add_geometry(name=f'mesh_{i}', geometry=g)

        window.reset_camera_to_default()
        app.add_window(window)
    vis.gui.Application.instance.initialize()
    run(pathnum=pathnum)
    vis.gui.Application.instance.run()

class GenerationVisualization:
    def __init__(self, general_folder='dataset', asset_folder='assets'):
        self.window = o3d.visualization.Visualizer()
        self.window.create_window("GenerationView")

        self.general_folder = general_folder
        os.makedirs(self.general_folder, exist_ok=True)
        self.asset_folder = asset_folder

        self.image_folder = os.path.join(self.general_folder, 'images')
        self.mask_folder = os.path.join(self.general_folder, 'masks')
        self.annotations_folder = os.path.join(self.general_folder, 'annotations')
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.mask_folder, exist_ok=True)
        os.makedirs(self.annotations_folder, exist_ok=True)

    def _set_view(self, window):
        view_ctl = window.get_view_control()
        view_ctl.rotate(0.0, 5.8178*90)

        zoom_rate = round(random.uniform(0.5, 0.6), 1)
        view_ctl.set_zoom(zoom_rate)

        render_ctl = window.get_render_option()
        render_ctl.load_from_json('configs/render_option.json')

        window.update_renderer()
        return window

    def create_mask(self, image_path, lower_color=None, upper_color=None):
        image = cv2.imread(image_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 200]) if lower_color is None else lower_color
        upper_white = np.array([180, 30, 255]) if upper_color is None else upper_color
        mask = cv2.inRange(hsv, lower_white, upper_white)

        mask_inv = cv2.bitwise_not(mask)
        kernel = np.ones((10, 10), np.uint8)
        mask_filled = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)
        return mask_filled, mask_inv, mask

    def form_mask(self, index, temp_image):
        def random_wood_texture_path(asset):
            files = []
            for e in ['*.jpg', '*.png', '*.jpeg']: files.extend(list(pathlib.Path(asset).glob(e)))
            random_asset = random.choice(files)
            return str(random_asset)

        random_wood_texture = cv2.imread(random_wood_texture_path(asset=self.asset_folder))
        board_img = cv2.imread(temp_image)
        wood_texture = cv2.resize(random_wood_texture, (board_img.shape[1], board_img.shape[0]))
        mask_filled, mask_inv, mask = self.create_mask(temp_image)

        board_foreground = cv2.bitwise_and(board_img, board_img, mask=mask_inv)
        wood_background = cv2.bitwise_and(wood_texture, wood_texture, mask=mask)

        result = cv2.add(board_foreground, wood_background)
        return mask_filled, result

    def postprocess_instance_mask(self, matrix, binary_mask):
        y_indices, x_indices = np.where(binary_mask > 0)

        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        # Extract board size
        board_height = y_max - y_min + 1
        board_width = x_max - x_min + 1

        # Define number of rows and columns
        rows, cols = 14, 17
        cell_height = board_height // rows
        cell_width = board_width // cols

        grayscale_mask = np.zeros_like(binary_mask)

        # Draw colored squares on the board
        instance = 1
        for i in range(rows):
            for j in range(cols):
                if np.any(binary_mask[y_min + i * cell_height: y_min + (i + 1) * cell_height,
                          x_min + j * cell_width: x_min + (j + 1) * cell_width] > 0):
                    if str(matrix[i, j]) == 'o':
                        cv2.rectangle(grayscale_mask,
                                      (x_min + j * cell_width, y_min + i * cell_height),
                                      (x_min + (j + 1) * cell_width, y_min + (i + 1) * cell_height),
                                      0, -1)
                    else:
                        cv2.rectangle(grayscale_mask,
                                      (x_min + j * cell_width, y_min + i * cell_height),
                                      (x_min + (j + 1) * cell_width, y_min + (i + 1) * cell_height),
                                      instance, -1)  # -1 fills the rectangle
                        instance += 1

        # Show the result
        return grayscale_mask

    def postprocess_class_mask(self, matrix, binary_mask, labels2id, big_details, imsize=(1920, 1080)):
        y_indices, x_indices = np.where(binary_mask > 0)

        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        # Extract board size
        board_height = y_max - y_min + 1
        board_width = x_max - x_min + 1

        # Define number of rows and columns
        rows, cols = 14, 17
        cell_height = board_height // rows
        cell_width = board_width // cols

        grayscale_mask = np.zeros_like(binary_mask)

        test_board = []

        annotations = []
        # Draw colored squares on the board
        for i in range(rows):
            j = 0
            test_row = []
            while j < cols:
                if np.any(binary_mask[y_min + i * cell_height: y_min + (i + 1) * cell_height,
                          x_min + j * cell_width: x_min + (j + 1) * cell_width] > 0):
                    if str(matrix[i, j]) == 'o':
                        cv2.rectangle(grayscale_mask,
                                      (x_min + j * cell_width, y_min + i * cell_height),
                                      (x_min + (j + 1) * cell_width, y_min + (i + 1) * cell_height),
                                      0, -1)
                        test_row.append(0)
                    else:
                        if str(matrix[i, j]) in big_details:
                            x_min_b, y_min_b = None, y_min + (i - 1) * cell_height
                            x_max_b, y_max_b = x_min + (j + 2) * cell_width, y_min + (i + 1) * cell_height
                            if str(matrix[i, j]) == 'CL':
                                x_min_b = x_min + (j - 1) * cell_width
                                y_min_b = y_min + (i - 2) * cell_height
                            else:
                                x_min_b = x_min + (j - 1) * cell_width
                        else:
                            x_min_b, y_min_b = x_min + j * cell_width, y_min + i * cell_height
                            x_max_b, y_max_b = x_min + (j + 1) * cell_width, y_min + (i + 1) * cell_height
                        cv2.rectangle(grayscale_mask, (x_min_b, y_min_b), (x_max_b, y_max_b),
                                      labels2id.get(str(matrix[i, j])), -1)
                        annotations.append([str(labels2id.get(str(matrix[i, j]))-1),
                                            str(x_min_b / imsize[0]), str(y_min_b / imsize[1]),
                                            str(x_min_b / imsize[0]), str(y_max_b / imsize[1]),
                                            str(x_max_b / imsize[0]), str(y_min_b / imsize[1]),
                                            str(x_max_b / imsize[0]), str(y_max_b / imsize[1])])
                        test_row.append(int(labels2id.get(str(matrix[i, j]))))
                        if str(matrix[i, j]) in big_details:
                            test_row.append(int(labels2id.get(str(matrix[i, j]))))
                            j = j + 1
                else:
                    test_row.append(0)
                j += 1
            test_board.append(test_row)

        # print(np.array(test_board))

        return grayscale_mask, annotations

    def generate_board_mask(self, amount, generate_big_shapes=True, path_amount_range=(1, 3)):
        scene = Scene()
        labels2id = scene.labels2id
        big_details = scene.bigShapes

        for j in tqdm.tqdm(range(amount)):
            empty_cell_rate_value = round(random.uniform(0.65, 0.9), 1)
            pathnum = random.randint(*path_amount_range)
            scene.generateBoard(num=pathnum, empty_rate=empty_cell_rate_value, generate_big_shapes=generate_big_shapes)
            objects = scene.fillObjects()
            geoms = scene.getGeoms()

            for i, g in enumerate(geoms):
                self.window.add_geometry(geometry=g)

            window = self._set_view(self.window)
            window.poll_events()
            window.capture_screen_image(f'temporary.png', do_render=True)
            mask_filled, result = self.form_mask(index=j, temp_image=f'temporary.png')
            #color_mask = self.postprocess_instance_mask(scene.getBoard().board_matrix, mask_filled)
            color_mask, annotations = self.postprocess_class_mask(scene.getBoard().board_matrix,
                                                                  mask_filled, labels2id, big_details)

            cv2.imwrite(f'{self.mask_folder}/{j}.png', color_mask)
            cv2.imwrite(f'{self.image_folder}/{j}.png', result)

            with open(f"{self.annotations_folder}/{j}.txt", "w") as txt_file:
                for line in annotations:
                    txt_file.write(" ".join(line) + "\n")

            for i, g in enumerate(geoms):
                window.remove_geometry(geometry=g)

        self.window.destroy_window()
        os.remove("temporary.png")
        return True


if __name__ == "__main__":
    generator = GenerationVisualization(asset_folder='wood_texture')
    generator.generate_board_mask(amount=512, generate_big_shapes=True)
