import json
import os.path
import pathlib
import random

import numpy as np
from enum import Enum

from gen_src.layout.objects import SceneObject, Board


class Color(Enum):
    PALE = [255, 231, 166]


class Scene:
    def __init__(self):
        self._board = None
        self.geoms = None
        self.detail_config = 'configs/details_config.json'
        base_dir = pathlib.Path(__file__).resolve().parent.parent
        config = os.path.join(base_dir, self.detail_config)
        with open(config) as f:
            json_data = json.load(f)
            self.general_data_path = json_data['general_path']
            self.config_data = json_data['data']
            self.labels2id = dict(json_data['labels2id'])
            self.bigShapes = list(json_data['labelsOfBigShapes'])

    def get_config_entry(self, idx):
        data = [data for data in self.config_data if data['idx'] == idx][0]


    def generateBoard(self, empty_rate = 0.7, num=None, size=(14, 17), generate_big_shapes=True):
        config = [data for data in self.config_data if data['name'] == 'Board'][0]
        board = Board(os.path.join(self.general_data_path, config['mesh']), size)
        board.create_mesh()
        board.set_color(np.array(config['color']))
        board.rotate(config['rotate'])

        if num is not None:
            board.map_fill(num)
        else:
            board.random_fill(self.config_data, self.labels2id, self.bigShapes, empty_rate, generate_big_shapes)

        board.get_center()
        board.uniformBlockSize()

        self._board = board
        #print(f'BOARD: \n{self._board.board_matrix}')

    def createObject(self, idx, matrix_position, rotation):

        config = [data for data in self.config_data if data['idx'] == idx][0]
        triangle_object = SceneObject(config['idx'], os.path.join(self.general_data_path, config['mesh']), matrix_position)
        triangle_object.create_mesh()
        triangle_object.rotate(rotation)
        triangle_object.perform_translate(self._board.center, self._board.uniform)

        return triangle_object

    def fillObjects(self):
        self.geoms = [{"name": f"board",
                       "type": "B",
                        "geometry": self._board}]

        it = np.nditer(self._board.board_matrix, flags=['multi_index', "refs_ok"])
        while not it.finished:
            if it[0].item().detail_type != 'o':
                rotation = it[0].item().rotation
                entry = {"name": f"object_{it.multi_index}",
                         "type": it[0].item().detail_type,
                         "geometry": self.createObject(it[0].item().detail_type, it.multi_index, rotation)}
                self.geoms.append(entry)

            is_not_finished = it.iternext()

        #self.definePath()
        return self.geoms

    def definePath(self):
        forwards = [obj['geometry'] for obj in self.geoms if obj['geometry'].index == 'F']
        boundaries = [obj for obj in forwards if obj.matrix_position[0] == 0 or obj.matrix_position[1] == 0]

        selected_objects = random.sample(boundaries, 2)
        for s_object in selected_objects:
            s_object.set_color(np.array(Color.GREEN.value))

    def getBoard(self):
        return self._board

    def getGeoms(self):
        return [entry['geometry'].get_mesh() for entry in self.geoms]
