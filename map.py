import heapq
import random
import numpy as np


def manhattan_distance(start, goal):
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])


class MapObject:
    def __init__(self, detail_type, rotation=(0, 0, 0)):
        self.detail_type = detail_type
        self.rotation = rotation

    def __repr__(self):
        return str(self.detail_type)


class Map:
    def __init__(self, shape=(14, 17)):
        self.plane_shape = shape
        self.plane = np.full(shape, MapObject('o'), dtype=MapObject)
        self.occupied_points = []

    def generate_start(self):
        point = None
        while point is None or point in self.occupied_points:
            start_row = random.randint(0, self.plane_shape[0] - 1)
            start_col = random.randint(0, self.plane_shape[1] - 1)
            point = (start_row, start_col)
        return point

    def generate_end(self, start_point, path_length):
        row, col = start_point
        possible_positions = []

        for _ in range(100):
            new_row = random.randint(max(0, row - path_length), min(self.plane_shape[0] - 1, row + path_length))
            new_col = random.randint(max(0, col - path_length), min(self.plane_shape[1] - 1, col + path_length))
            if abs(new_row - row) + abs(new_col - col) == path_length:
                possible_positions.append((new_row, new_col))

        possible_positions = [point for point in possible_positions if point not in self.occupied_points]
        return random.choice(possible_positions) if possible_positions else start_point

    def astar(self, start, goal):
        open_list = []
        heapq.heappush(open_list, (0, start))

        g_costs = {start: 0}

        came_from = {start: None}

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while open_list:
            current_cost, current_pos = heapq.heappop(open_list)

            if current_pos == goal:
                path = []
                while current_pos is not None:
                    path.append(current_pos)
                    current_pos = came_from[current_pos]
                return path[::-1]

            for direction in directions:
                neighbor = (current_pos[0] + direction[0], current_pos[1] + direction[1])
                if ((0 <= neighbor[0] < np.shape(self.plane)[0] and 0 <= neighbor[1] < np.shape(self.plane)[1])
                        and self.plane[neighbor[0]][neighbor[1]].detail_type in ['F', 'o']):

                    tentative_g_cost = g_costs[current_pos] + 1

                    if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                        g_costs[neighbor] = tentative_g_cost
                        priority = tentative_g_cost + manhattan_distance(neighbor, goal)
                        heapq.heappush(open_list, (priority, neighbor))
                        came_from[neighbor] = current_pos
        return None

    def write_path(self, route):
        for i, point in enumerate(route):
            self.occupied_points.append(point)

    def define_turn(self, previous_direction, future_direction):
        left_rotations = [np.pi, np.pi / 2, 0, -np.pi / 2]
        left_turns = [[(1, 0), (0, -1)], [(0, 1), (1, 0)], [(-1, 0), (0, 1)], [(0, -1), (-1, 0)]]
        for i, turn in enumerate(left_turns):
            if previous_direction == turn[0] and future_direction == turn[1]:
                return MapObject('L', (0, left_rotations[i], 0))

        right_rotations = [-np.pi / 2, 0, np.pi / 2, np.pi]
        right_turns = [[(0, 1), (-1, 0)], [(1, 0), (0, 1)], [(0, -1), (1, 0)], [(0, 1), (0, -1)]]
        for i, turn in enumerate(right_turns):
            if previous_direction == turn[0] and future_direction == turn[1]:
                return MapObject('R', (0, right_rotations[i], 0))

    def define_border(self, direction, detail_type):
        border_rotations = [0, -np.pi / 2, np.pi, np.pi / 2]
        start_turns = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        end_turns = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        if detail_type == 'S':
            for i, turn in enumerate(start_turns):
                if direction == turn:
                    # print('inited start')
                    return MapObject('S', (0, border_rotations[i], 0))
        else:
            for i, turn in enumerate(end_turns):
                if direction == turn:
                    # print('inited end')
                    return MapObject('E', (0, border_rotations[i], 0))
        return None


    def check_crossroad(self, point):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for direction in directions:
            neighbour = (point[0] + direction[0], point[1] + direction[1])
            if  not (0 <= neighbour[0] < np.shape(self.plane)[0] and 0 <= neighbour[1] < np.shape(self.plane)[1]):
                return False
            else:
                if self.plane[neighbour].detail_type == 'o':
                    return False

        return True

    def define_crossroad(self, point):
        # left_point_rotation = self.plane[point[0], point[1]-1].rotation[1]
        # up_point_rotation = self.plane[point[0]+1, point[1]].rotation[1]
        return MapObject('C')

    def postprocess_route(self, route):
        print(self.plane)
        for i in range(1, len(route) - 1):  # from 2nd to 1 before last
            previous_direction = (route[i][1] - route[i - 1][1], route[i][0] - route[i - 1][0])
            future_direction = (route[i + 1][1] - route[i][1], route[i + 1][0] - route[i][0])
            if previous_direction != future_direction:
                turn_type = self.define_turn(previous_direction, future_direction)
                #print(f'{route[i]}, {turn_type.detail_type}: pr: {previous_direction}, ft: {future_direction}')
                self.plane[route[i]] = turn_type
            elif previous_direction == future_direction == (1, 0):
                self.plane[route[i]] = MapObject('F', (0, -np.pi / 2, 0))
            elif previous_direction == future_direction == (-1, 0):
                self.plane[route[i]] = MapObject('F', (0, np.pi / 2, 0))
            elif previous_direction == future_direction == (0, 1):
                self.plane[route[i]] = MapObject('F', (0, np.pi, 0))
            else:
                self.plane[route[i]] = MapObject('F')

        start_future_direction = (route[1][1] - route[0][1], route[1][0] - route[0][0])
        self.plane[route[0]] = self.define_border(start_future_direction, 'S')
        end_previous_direction = (route[-1][1] - route[-2][1], route[-1][0] - route[-2][0])
        self.plane[route[-1]] = self.define_border(end_previous_direction, 'E')

        for i in range(1, len(route) - 1):
            if self.plane[route[i]].detail_type != 'o' and self.check_crossroad(route[i]):
                self.plane[route[i]] = self.define_crossroad(route[i])

    def create_plane(self, num, path_length_range=(10, 20)):
        for i in range(num):
            start_point, end_point = None, None
            while start_point == end_point:
                path_length = random.randint(path_length_range[0], path_length_range[1])
                start_point = self.generate_start()
                end_point = self.generate_end(start_point, path_length)
            route = self.astar(start_point, end_point)
            self.write_path(route)
            self.postprocess_route(route)

        return self.plane


def print_plane(plane):
    for row in plane:
        print(" ".join(row))

# if __name__ == "__main__":
#     map = Map()
#     plane = map.create_plane()
#     print_plane(plane)
