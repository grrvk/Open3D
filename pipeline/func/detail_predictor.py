import os
import pathlib
from datetime import datetime

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from ultralytics import YOLO

class DetailPredictor:
    def __init__(self, model_path, save_results_folder='pipeline_results'):
        self.model = YOLO(model_path)

        base_path = pathlib.Path(__file__).resolve().parent.parent.parent
        self.save_results_folder = os.path.join(base_path, save_results_folder)

    def predict(self, image_path):
        results = self.model.predict(source=image_path, conf=0.5, save=False)
        result = results[0]

        boxes = result.boxes

        xyxy = boxes.xyxy
        conf = boxes.conf
        cls_ids = boxes.cls
        return xyxy, conf, cls_ids

    @staticmethod
    def visualize(image, boxes: np.ndarray, conf: np.ndarray,
                    cls_ids: np.ndarray, colormap: str = "viridis", plot: bool = True) -> np.ndarray:
        if isinstance(image, str):
            image = cv2.imread(image)
        unique_classes = np.unique(cls_ids)
        num_classes = len(unique_classes)

        cmap = matplotlib.colormaps.get_cmap(colormap)
        colors = [tuple(int(255 * x) for x in cmap(i)[:3]) for i in np.linspace(0, 1, num_classes)]
        class_colors = {cls_id: colors[i] for i, cls_id in enumerate(unique_classes)}

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            cls_i = int(cls_ids[i])
            color = class_colors.get(cls_i, (255, 255, 255))

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label = f"{cls_i}: {float(conf[i]):.2f}"
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

    def run(self, image_obg, save: bool = True):
        image = image_obg.copy()
        if isinstance(image, str):
            image = cv2.imread(image)
        xyxy, conf, cls_ids = self.predict(image)
        visualized_details_predicted = self.visualize(image, xyxy, conf, cls_ids)
        if save:
            if isinstance(image_obg, str):
                image_name = f'detected_{os.path.basename(image_obg)}'
            else:
                image_name = f'detected_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
            output_path = os.path.join(self.save_results_folder, image_name)
            cv2.imwrite(output_path, visualized_details_predicted)
        return visualized_details_predicted, (xyxy, conf, cls_ids)


if __name__ == '__main__':
    detail_predictor = DetailPredictor(model_path='models/best.pt')
    detail_predictor.run(image_obg='1.png')


