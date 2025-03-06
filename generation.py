import random

import numpy as np
from enum import Enum

from objects import SceneObject, Board, GeneralObject


class Color(Enum):
    PALE = [255, 231, 166]


class Settings(Enum):
    B = dict({'path': "/Users/vika/Desktop/objects/exported_one/board/Board.ply", 'idx': 'B', 'color': Color.PALE.value, 'rotate': (-np.pi / 2, 0, -np.pi / 2)})
    F = dict({'path': "/Users/vika/Desktop/objects/exported_one/forward/forward.obj", 'idx': '#'})
    S = dict({'path': "/Users/vika/Desktop/objects/exported_one/start/start.obj", 'idx': 'S'})
    E = dict({'path': "/Users/vika/Desktop/objects/exported_one/end/end.obj", 'idx': 'E'})
    L = dict({'path': "/Users/vika/Desktop/objects/exported_one/turn_left/turn_left.obj", 'idx': 'L'})
    R = dict({'path': "/Users/vika/Desktop/objects/exported_one/turn_right/turn_right.obj", 'idx': 'R'})
    C = dict({'path': "/Users/vika/Desktop/objects/exported_one/cross/cross.obj", 'idx': 'C'})


class Scene:
    def __init__(self):
        self._board = None
        self.geoms = None

    def generateBoard(self, config=Settings.B.value, size=(14, 17)):
        board = Board(config['path'], size)
        board.create_mesh()
        board.set_color(np.array(config['color']))
        board.rotate(config['rotate'])
        board.map_fill()
        #board.random_fill()

        board.get_center()
        board.uniformBlockSize()

        self._board = board
        print(f'BOARD: \n{self._board.board_matrix}')

    def createObject(self, config, matrix_position, rotation):
        triangle_object = SceneObject(config['idx'], config['path'], matrix_position)
        triangle_object.create_mesh()
        triangle_object.rotate(rotation)
        triangle_object.perform_translate(self._board.center, self._board.uniform)

        return triangle_object

    def fillObjects(self):
        self.geoms = [{"name": f"board",
                        "geometry": self._board}]

        it = np.nditer(self._board.board_matrix, flags=['multi_index', "refs_ok"])
        while not it.finished:
            if it[0].item().detail_type != 'o':
                rotation = it[0].item().rotation
                entry = {"name": f"object_{it.multi_index}",
                         "geometry": self.createObject(Settings[it[0].item().detail_type].value, it.multi_index, rotation)}
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
