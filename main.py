from PIL import Image
import cv2
import numpy as np
import os
import time
import open3d as o3d
import open3d.visualization as vis
from generation import Scene


def on_menu_screenshot():
    print('screenshot')



class App:
    MENU_SCREENSHOT = 1
    MENU_QUIT = 2

    def __init__(self):
        self.app = vis.gui.Application.instance
        self.window = o3d.visualization.O3DVisualizer("ui")
        self.window.add_action('screenshot', lambda: on_menu_screenshot())

        #self.window.setup_camera([0, 0, 0])
        self.window.set_background(bg_color=np.array([0.8, 0.9, 0.9, 1.0], dtype=np.float32), bg_image=None)

        scene = Scene()
        scene.generateBoard()
        objects = scene.fillObjects()
        geoms = scene.getGeoms()

        for i, g in enumerate(geoms):
            self.window.add_geometry(name=f'mesh_{i}',geometry=g)

        self.window.reset_camera_to_default()
        self.app.add_window(self.window)

        # def rotate_view(vis):
        #     ctr = vis.get_view_control()
        #     ctr.rotate(10.0, 0.0)
        #     return False
        #
        # self.window.draw_geometries_with_animation_callback(geoms, rotate_view)

        # self.app = vis.gui.Application.instance
        # self.window = o3d.visualization.Visualizer()
        # self.window.create_window("Ui")
        #
        # scene = Scene()
        # scene.generateBoard()
        # objects = scene.fillObjects()
        # geoms = scene.getGeoms()
        #
        # for i, g in enumerate(geoms):
        #     self.window.add_geometry(geometry=g)
        #
        # self.window.run()

def main():
    vis.gui.Application.instance.initialize()
    App()
    vis.gui.Application.instance.run()

    # scene = Scene()
    # scene.generateBoard()
    # objects = scene.fillObjects()
    # geoms = scene.getGeoms()
    #
    # vis.draw(geoms,
    #          bg_color=(0.8, 0.9, 0.9, 1.0),
    #          show_ui=True,
    #          width=1920,
    #          height=1080)

if __name__ == "__main__":
    main()
