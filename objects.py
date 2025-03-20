import copy
import random

import cv2
import numpy as np
import open3d as o3d
import open3d.visualization as vis

from map import Map, MapObject


class GeneralObject:
    def __init__(self, ply_filepath, index='o'):
        self._mesh = None
        self.index = index
        self.ply_filepath = ply_filepath

    def create_mesh(self):
        mesh = o3d.io.read_triangle_mesh(self.ply_filepath, enable_post_processing=True)

        if not mesh.has_vertex_normals(): mesh.compute_vertex_normals()
        if not mesh.has_triangle_normals(): mesh.compute_triangle_normals()

        self._mesh = mesh

    def get_mesh(self):
        return self._mesh

    def get_index(self):
        return self.index

    def get_center_coordinates(self):
        return self._mesh.get_center()

    def set_color(self, color):
        self._mesh.paint_uniform_color(color / 255.0)

    def rotate(self, rotation):
        rotate_mesh = o3d.geometry.get_rotation_matrix_from_xyz(rotation)
        self._mesh.rotate(rotate_mesh, center=(0, 0, 0))

    def translate(self, translation):
        self._mesh.translate(translation)

    def scale(self, scale_value):
        print(scale_value)
        self._mesh.scale(scale_value, self._mesh.get_center())

    def custom_scale(self, scale):
        center = np.asarray(self._mesh.get_center(), dtype=np.float64)
        vertices = np.asarray(self._mesh.vertices)
        vertices = (vertices - center) * np.array(scale) + center
        self._mesh.vertices = o3d.utility.Vector3dVector(vertices)

    def get_size_params(self):
        bbox = self._mesh.get_axis_aligned_bounding_box()
        min_bound = bbox.min_bound  # [x_min, y_min, z_min]
        max_bound = bbox.max_bound  # [x_max, y_max, z_max]

        width = max_bound[0] - min_bound[0]
        height = max_bound[2] - min_bound[2]
        depth = max_bound[1] - min_bound[1]

        return width, height, depth

    def perform_scaling(self, uniform_parameters):
        pass

    def perform_translate(self, matrix_center, uniform_parameters):
        pass


class Board(GeneralObject):
    def __init__(self, ply_filepath, size):
        super().__init__(ply_filepath=ply_filepath)
        self.board_matrix: np.ndarray = None
        self.size = size

        self.uniform = None
        self.center = None

    def random_fill(self, config_data, big_shapes, empty_rate):

        labels = [MapObject('o')]
        united_big_shapes = copy.deepcopy(big_shapes)
        united_big_shapes.extend(['B'])
        labels.extend([MapObject(data['idx']) for data in config_data if data['idx'] not in united_big_shapes])

        rotations = [0, -np.pi / 2, np.pi, np.pi / 2]

        probabilities = [empty_rate]
        probabilities.extend([(1-empty_rate)/(len(labels)-1)] * (len(labels)-1))

        self.board_matrix = np.random.choice(labels, size=self.size, p=probabilities)

        it = np.nditer(self.board_matrix, flags=['multi_index', "refs_ok"])
        while not it.finished:
            if it[0].item().detail_type != 'o':
                chance = random.random()
                if chance < .5:
                    random_rotation = random.choice(rotations)
                    it[0].item().rotation = (0, random_rotation, 0)
            is_not_finished = it.iternext()

        big_shapes_copy = copy.deepcopy(big_shapes)
        count = random.randint(0, len(big_shapes)-1)
        print(f'Big figures count: {count}')

        for _ in range(count):
            random_big_shape = random.choice(big_shapes_copy)
            big_shapes_copy.remove(random_big_shape)
            if random_big_shape == 'CL':
                i = np.random.randint(2, self.size[0]-1) # row of 14
                j = np.random.randint(1, self.size[1]-1) # column of 17

                self.board_matrix[i][j] = MapObject('CL')
                self.free_surroundings(i, j, x_shift=1, up_shift=0, down_shift=2)
            else:
                i = np.random.randint(1, self.size[0] - 1)  # row of 14
                j = np.random.randint(1, self.size[1] - 1)  # column of 17

                self.board_matrix[i][j] = MapObject(random_big_shape)
                self.free_surroundings(i, j, x_shift=1, up_shift=0, down_shift=1)

        for row in self.board_matrix:
            print(" ".join([e.detail_type for e in row]))

    def free_surroundings(self, i, j, x_shift, up_shift=0, down_shift=0):
        radius = [j - x_shift, j, j + x_shift]
        for pos_x in radius:
            if up_shift != 0:
                for k in range(1, up_shift+1):
                    self.board_matrix[i+k][pos_x] = MapObject('o')
            if down_shift != 0:
                for k in range(1, down_shift+1):
                    self.board_matrix[i-k][pos_x] = MapObject('o')

        self.board_matrix[i][j - x_shift] = MapObject('o')
        self.board_matrix[i][j + x_shift] = MapObject('o')

    def map_fill(self, num):
        map = Map()
        self.board_matrix = map.create_plane(num)

    def uniformBlockSize(self):
        width, height, depth = self.get_size_params()
        matrix_height, matrix_width = np.shape(self.board_matrix)

        uniform_width = width / matrix_width
        uniform_height = height / matrix_height

        self.uniform = uniform_width, uniform_height, depth

        return self.uniform

    def get_center(self):
        matrix_height, matrix_width = np.shape(self.board_matrix)

        self.center = (matrix_height / 2, matrix_width / 2)
        return self.center


class SceneObject(GeneralObject):
    def __init__(self, index, ply_filepath, matrix_pos):
        super().__init__(ply_filepath=ply_filepath, index=index)
        self.matrix_position = matrix_pos

    def perform_scaling(self, uniform_parameters):
        width, height, depth = self.get_size_params()
        scale_x, scale_z, scale_y = uniform_parameters[0] / width, uniform_parameters[1] / height, 1 / depth
        self.custom_scale([scale_x, scale_z, scale_y])

    def perform_translate(self, matrix_center, uniform_parameters):
        vertical_movement, horizontal_movement = (self.matrix_position[0] - matrix_center[0],
                                                  self.matrix_position[1] - matrix_center[1])

        translation_x = horizontal_movement * uniform_parameters[0] + uniform_parameters[0] / 2
        translation_z = vertical_movement * uniform_parameters[1] + uniform_parameters[1] / 2

        translation = (translation_x, 0, translation_z)

        self.translate(translation)
