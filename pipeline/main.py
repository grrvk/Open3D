from pipeline.func.board_segmentor import BoardSegmentor
from pipeline.func.detail_filter import DetailFilter
from pipeline.func.detail_predictor import DetailPredictor
from pipeline.func.json_convert import JsonConverter


def main():
    board_segmentor = BoardSegmentor(model_path='func/models/board_seg_model.pth')
    board = board_segmentor.run(image_path='1.png')

    detail_predictor = DetailPredictor(model_path='func/models/best.pt')
    visualized_details_predicted, predictions = detail_predictor.run(image_obg=board)

    detail_filter = DetailFilter()
    _, filtered_details = detail_filter.run(image_obg=visualized_details_predicted, predictions=predictions)
    _, _, cell_w, cell_h, _, _ = detail_filter.get_cell_shape(visualized_details_predicted)

    converter = JsonConverter(path_to_base_config = 'func/configs/base_config.json')
    converter.run(filtered_details=filtered_details, cell_shape=(cell_w, cell_h))

if __name__ == '__main__':
    main()
