import pathlib
from typing import Optional, Tuple
import segmentation_models_pytorch as smp
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

import cv2

class BoardSegmentor:
    def __init__(self, model_path, save_results_folder='pipeline_results'):
        self.model_path = model_path
        self.model = self._load_model()

        base_path = pathlib.Path(__file__).resolve().parent.parent.parent
        self.save_results_folder = os.path.join(base_path, save_results_folder)


    def _load_model(self):
        model = smp.Unet(encoder_name='resnet34', in_channels=3, classes=1, activation=None)
        model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu'), weights_only=True))
        return model

    def predict(self, img_path, size=(256, 256)):
        self.model.eval()
        img = Image.open(img_path).convert("RGB")
        orig_size = img.size

        img_resized = img.resize(size)
        img_np = np.array(img_resized)

        transform = A.Compose([
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])
        transformed = transform(image=img_np)
        img_tensor = transformed['image'].unsqueeze(0)

        with torch.no_grad():
            pred = self.model(img_tensor)[0][0].sigmoid().cpu().numpy()

        pred_mask_resized = cv2.resize(pred, orig_size)
        return pred_mask_resized

    @staticmethod
    def get_corners(mask, padding: int = 5):
        binary_mask = (mask > 0.5).astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No contours found")

        kernel = np.ones((15, 15), np.uint8)
        smoothed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour = max(contours, key=cv2.contourArea)

        epsilon = 0.03 * cv2.arcLength(biggest_contour, True)
        approx = cv2.approxPolyDP(biggest_contour, epsilon, True)

        if len(approx) == 4:
            box = approx.reshape(4, 2)
        else:
            rect = cv2.minAreaRect(biggest_contour)
            box = cv2.boxPoints(rect)
            box = np.int8(box)

        center = np.mean(box, axis=0)
        padded_box = box.copy().astype(np.float32)

        for i in range(4):
            direction = box[i] - center
            norm_direction = direction / np.linalg.norm(direction)
            padded_box[i] += norm_direction * padding

        return padded_box.astype(np.int32)

    @staticmethod
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def get_output_size(rect):
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))
        output_size = (maxWidth, maxHeight)
        return output_size

    def warp_board(self, img_path: str, corners: np.ndarray, output_size: Optional[Tuple[int, int]] = None,
                   save: bool = True) -> np.ndarray:
        img = np.array(Image.open(img_path).convert("RGB"))
        rect = self.order_points(corners)
        if output_size is None:
            maxWidth, maxHeight = self.get_output_size(rect)
        else:
            maxWidth, maxHeight = output_size

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

        if save:
            result_img = Image.fromarray(warped)
            os.makedirs(self.save_results_folder, exist_ok=True)
            result_img.save(os.path.join(self.save_results_folder, f'cropped_{os.path.basename(img_path)}'))

        return warped

    def visualize(self, image_path: str, mask: Optional[np.ndarray] = None, corners: Optional[np.ndarray] = None,
                  board: Optional[np.ndarray] = None,):
        img = Image.open(image_path).convert("RGB")
        if mask is not None:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title("Original Image")
            plt.subplot(1, 2, 2)
            plt.imshow(mask > 0.5, cmap='gray')
            plt.title("Predicted Mask (Resized)")
            plt.show()
        if corners is not None:
            image = cv2.imread(image_path)
            for i in range(4):
                pt1 = tuple(corners[i])
                pt2 = tuple(corners[(i + 1) % 4])
                cv2.line(image, pt1, pt2, (0, 255, 0), 3)

            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title("Image with Corner Lines")
            plt.axis("off")
            plt.show()
        if board is not None:
            plt.figure(figsize=(8, 8))
            plt.imshow(board)
            plt.title('Warped Board')
            plt.axis('off')
            plt.show()

    def run(self, image_path, output_size: Optional[Tuple[int, int]] = None, save: bool = True, visualize: bool = True):
        mask = self.predict(image_path)
        corners = self.get_corners(mask)
        board = self.warp_board(image_path, corners, output_size, save)
        if visualize:
            self.visualize(image_path, mask, corners, board)
        return board

if __name__ == "__main__":
    board_segmentor = BoardSegmentor(model_path='models/board_seg_model.pth')
    board = board_segmentor.run(image_path='1.png')

