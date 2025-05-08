import json
import os
import pathlib

CLASS_2_TYPE = {
    0: "gerade",
    1: "rechtskurve",
    2: "kreuzung",
    5: "weiche"
}

class JsonConverter:
    def __init__(self, save_results_folder='pipeline_results', path_to_base_config: str = 'configs/base_config.json'):
        base_path = pathlib.Path(__file__).resolve().parent.parent.parent
        self.save_results_folder = os.path.join(base_path, save_results_folder)

        self.path_to_base_config = path_to_base_config
        self.base_save_name = os.path.join(self.save_results_folder, 'generated_config.json')

    @staticmethod
    def log_details(filtered_details, cell_shape):
        cell_w, cell_h = cell_shape
        pieces = []
        for obj in filtered_details:
            x1, y1, x2, y2 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]
            cls_i = obj["cls"]

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            col = int(cx // cell_w)
            row = int(cy // cell_h)

            if cls_i == 5:
                w = x2 - x1
                h = y2 - y1
                if w > h:
                    col = int((x1) // cell_w)
                    row = int((y1 + y2) / 2 // cell_h)
                else:
                    col = int((x1 + x2) / 2 // cell_w)
                    row = int((y1) // cell_h)

            pieces.append({
                "type": CLASS_2_TYPE[cls_i],
                "x": col,
                "y": row,
                "color": "#CCCCCC",
                "orientation": 0,
                "state": 0
            })
        return pieces

    def save_json(self, pieces):
        with open(self.path_to_base_config, 'r') as file:
            data = json.load(file)

        data["prefs"]["pattern"] = [None] * 6
        data["visibilities"] = [1] * 32
        data["pieces"] = pieces

        with open(self.base_save_name, 'w', encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def run(self, filtered_details, cell_shape):
        pieces = self.log_details(filtered_details, cell_shape)
        self.save_json(pieces)
        print('Data saved to {}'.format(self.base_save_name))