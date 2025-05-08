import os
import pathlib
from datetime import datetime

import matplotlib
import torch
import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO
import numpy as np

class DetailFilter:
    def __init__(self, board_size=(14, 17), save_results_folder='pipeline_results'):
        self.grid = None
        self.grid_occupied = None
        self.filtered = []

        self.board_size = board_size
        self.grid_formation()

        self.small_classes = [0, 1, 2]
        self.large_classes = [5]

        base_path = pathlib.Path(__file__).resolve().parent.parent.parent
        self.save_results_folder = os.path.join(base_path, save_results_folder)

    def get_cell_shape(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        img_h, img_w = image.shape[:2]
        grid_w, grid_h = self.board_size[1], self.board_size[0]
        cell_w, cell_h = img_w // grid_w, img_h // grid_h
        return img_h, img_w, cell_w, cell_h, grid_w, grid_h

    def grid_formation(self):
        grid_w, grid_h = self.board_size[1], self.board_size[0]
        self.grid = [[None for _ in range(grid_w)] for _ in range(grid_h)]
        self.grid_occupied = [[False for _ in range(grid_w)] for _ in range(grid_h)]

    def large_detail_filtering(self, xyxy, conf, cls_ids, img):
        img_h, img_w, cell_w, cell_h, grid_w, grid_h = self.get_cell_shape(img)
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            w, h = x2 - x1, y2 - y1
            area = w * h
            cls_i = int(cls_ids[i])
            conf_i = float(conf[i])

            if cls_i not in self.large_classes:
                continue
            if area > 3 * img_w * img_h:
                print(f"Area {area} is too large")
                continue
            if area < 0.018 * img_w * img_h:
                print(f"Area {area} is too small")
                continue

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            col = int(cx // cell_w)
            row = int(cy // cell_h)

            if not (0 <= col < grid_w and 0 <= row < grid_h):
                continue

            if w > h:
                rows = range(max(0, row), min(grid_h, row + 2))
                cols = range(max(0, col - 1), min(grid_w, col + 2))
            else:
                rows = range(max(0, row - 1), min(grid_h, row + 2))
                cols = range(max(0, col), min(grid_w, col + 2))

            for r in rows:
                for c in cols:
                    self.grid_occupied[r][c] = True

            self.filtered.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'cls': cls_i, 'conf': conf_i
            })

    def small_detail_filtering(self, xyxy, conf, cls_ids, img):
        img_h, img_w, cell_w, cell_h, grid_w, grid_h = self.get_cell_shape(img)
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            w, h = x2 - x1, y2 - y1
            area = w * h
            cls_i = int(cls_ids[i])
            conf_i = float(conf[i])

            if cls_i not in self.small_classes:
                continue

            if area < 0.003 * img_w * img_h or area > 0.5 * img_w * img_h:
                continue

            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            col = int(cx // cell_w)
            row = int(cy // cell_h)

            if not (0 <= col < grid_w and 0 <= row < grid_h):
                continue

            if self.grid_occupied[row][col]:
                print(f"Cell {row}, {col} is occupied")
                continue

            existing = self.grid[row][col]
            if existing is None or conf_i > existing['conf']:
                self.grid[row][col] = {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'cls': cls_i, 'conf': conf_i
                }
        for row in self.grid:
            for cell in row:
                if cell is not None:
                    self.filtered.append(cell)

    def visualize(self, image, colormap: str = "viridis", plot: bool = True):
        unique_classes = np.unique(np.array([obj['cls'] for obj in self.filtered]))
        num_classes = len(unique_classes)

        cmap = matplotlib.colormaps.get_cmap(colormap)
        colors = [tuple(int(255 * x) for x in cmap(i)[:3]) for i in np.linspace(0, 1, num_classes)]
        class_colors = {cls_id: colors[i] for i, cls_id in enumerate(unique_classes)}

        for obj in self.filtered:
            x1, y1, x2, y2 = map(int, [obj['x1'], obj['y1'], obj['x2'], obj['y2']])
            cls_i = obj['cls']
            color = class_colors.get(cls_i, (255, 255, 255))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_i}: {float(obj['conf']):.2f}"
            cv2.putText(image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if plot:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 6))
            plt.imshow(rgb_image)
            plt.axis("off")
            plt.title("Details detected")
            plt.show()

        return image

    def run(self, image_obg, predictions, save: bool = True):
        image = image_obg.copy()
        if isinstance(image, str):
            image = cv2.imread(image)
        xyxy, conf, cls_ids = predictions
        self.large_detail_filtering(xyxy, conf, cls_ids, image)
        self.small_detail_filtering(xyxy, conf, cls_ids, image)
        image = self.visualize(image)
        if save:
            if isinstance(image_obg, str):
                image_name = f'filtered_{os.path.basename(image_obg)}'
            else:
                image_name = f'filtered_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
            output_path = os.path.join(self.save_results_folder, image_name)
            cv2.imwrite(output_path, image)
        return image, self.filtered



