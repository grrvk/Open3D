import copy
import random

import numpy as np
import open3d as o3d

from .map import Map, MapObject


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

    def place_big_shape(self, detail, x_shift, down_shift, up_shift=0, trial_count=100):
        for _ in range(trial_count):
            i = np.random.randint(down_shift, self.size[0] - 1)  # row of 14
            j = np.random.randint(x_shift, self.size[1] - 1)  # column of 17

            if self.check_surroundings(i, j, x_shift=x_shift, up_shift=up_shift, down_shift=down_shift):
                self.board_matrix[i][j] = MapObject(detail)
                self.fill_surroundings(i, j, x_shift=x_shift, up_shift=up_shift, down_shift=down_shift)
                break

    def random_fill(self, config_data, config_labels, big_shapes, empty_rate, generate_big_shapes):
        labels = [MapObject('o')]
        united_big_shapes = copy.deepcopy(big_shapes)
        united_big_shapes.extend(['B'])
        labels.extend([MapObject(data['idx']) for data in config_data if data['idx'] not in united_big_shapes])

        rotations = [0, -np.pi / 2, np.pi, np.pi / 2]
        big_shapes_rotations = [0, np.pi]
        # probabilities = [empty_rate]
        # probabilities.extend([(1-empty_rate)/(len(labels)-1)] * (len(labels)-1))
        #

        probabilities = [empty_rate]
        probabilities.extend([(1 - empty_rate) / (len(labels) - 1)] * (len(labels) - 1))

        # Adjust probabilities based on label_id frequencies in config_labels
        label_id_counts = {label_id: list(config_labels.values()).count(label_id) for label_id in
                           set(config_labels.values())}

        for i, label in enumerate(labels):
            if label.detail_type in config_labels.keys():
                label_id = config_labels[label.detail_type]
                count = label_id_counts[label_id]
                probabilities[i] /= count

        probabilities = np.array(probabilities) / np.sum(probabilities)

        self.board_matrix = np.full(self.size, None)
        if generate_big_shapes:
            big_shapes_copy = copy.deepcopy(big_shapes)
            count = random.randint(5, 20)  # static: len()-1
            #print(f'Big figures count: {count}')

            # ADD OFFSET DETAIL
            for _ in range(count):
                random_big_shape = random.choice(big_shapes_copy)
                if random_big_shape == 'CL':
                    self.place_big_shape(detail = random_big_shape, x_shift=1, down_shift=2)
                else:
                    self.place_big_shape(detail = random_big_shape, x_shift=1, down_shift=1)

        #self.board_matrix[self.board_matrix == None] = MapObject('o')

        # for row in self.board_matrix:
        #     print(" ".join([e.detail_type for e in row]))

        small_shapes_board_matrix = np.random.choice(labels, size=self.size, p=probabilities)
        for i in range(self.board_matrix.shape[0]):
            for j in range(self.board_matrix.shape[1]):
                if self.board_matrix[i][j] is None and small_shapes_board_matrix[i][j].detail_type != 'o':
                    self.board_matrix[i][j] = small_shapes_board_matrix[i][j]

        self.board_matrix[self.board_matrix == None] = MapObject('o')

        it = np.nditer(self.board_matrix, flags=['multi_index', "refs_ok"])
        while not it.finished:
            obj = it[0].item()
            if obj.detail_type == 'p':
                self.board_matrix[it.multi_index] = MapObject('o')
            elif obj.detail_type != 'o' and obj.detail_type not in big_shapes:
                chance = random.random()
                if chance < 0.5:
                    random_rotation = random.choice(rotations)
                    obj.rotation = (0, random_rotation, 0)
            # elif obj.detail_type in big_shapes:
            #     chance = random.random()
            #     if chance < 0.5:
            #         random_rotation = random.choice(big_shapes_rotations)
            #         obj.rotation = (0, random_rotation, 0)
            it.iternext()

    def check_surroundings(self, i, j, x_shift, up_shift=0, down_shift=0):
        radius = [j - x_shift, j, j + x_shift]
        for pos_x in radius:
            if up_shift != 0:
                for k in range(0, up_shift+1):
                    if self.board_matrix[i+k][pos_x] is not None:
                        return False
            if down_shift != 0:
                for k in range(0, down_shift+1):
                    if self.board_matrix[i-k][pos_x] is not None:
                        return False
        return True


    def fill_surroundings(self, i, j, x_shift, up_shift=0, down_shift=0):
        radius = [j - x_shift, j, j + x_shift]
        for pos_x in radius:
            if up_shift != 0:
                for k in range(1, up_shift+1):
                    self.board_matrix[i+k][pos_x] = MapObject('p')
            if down_shift != 0:
                for k in range(1, down_shift+1):
                    self.board_matrix[i-k][pos_x] = MapObject('p')

        self.board_matrix[i][j - x_shift] = MapObject('p')
        self.board_matrix[i][j + x_shift] = MapObject('p')

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
