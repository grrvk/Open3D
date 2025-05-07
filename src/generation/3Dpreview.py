import open3d as o3d
import open3d.visualization as vis

from src.layout.scene import Scene

def o3d_preview(path_num=None, empty_rate=0.5, generate_big_shapes=True):
    def run():
        app = vis.gui.Application.instance
        window = o3d.visualization.O3DVisualizer("ui")

        window.setup_camera(field_of_view=10, center=[0, 0, 0], eye=[0, 30, 0], up=[0, 60, 0])

        scene = Scene()

        scene.generateBoard(num=path_num, empty_rate=empty_rate, generate_big_shapes=generate_big_shapes)
        scene.fillObjects()
        geoms = scene.getGeoms()

        for i, g in enumerate(geoms):
            window.add_geometry(name=f'mesh_{i}', geometry=g)

        window.reset_camera_to_default()
        app.add_window(window)
    vis.gui.Application.instance.initialize()
    run()
    vis.gui.Application.instance.run()

if __name__ == "__main__":
    o3d_preview()