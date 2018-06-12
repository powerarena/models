import os
import cv2
import numpy as np
from PIL import Image
from object_detection.utils import visualization_utils as vis_util
from pa_utils.data.data_utils import generate_output_video, read_video, get_video_info


def label_image(image_np, output_dict, category_index=None, min_score_thresh=0.5, use_normalized_coordinates=True,
                line_thickness=1, box_color='black'):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    # image_np_expanded = np.expand_dims(image_np, axis=0)

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict.get('detection_classes'),
        output_dict.get('detection_scores'),
        category_index or dict(),
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=use_normalized_coordinates,
        line_thickness=line_thickness,
        max_boxes_to_draw=100,
        min_score_thresh=min_score_thresh,
        groundtruth_box_visualization_color=box_color
    )


def add_waterprint(video_path, output_path, waterprint_path=None, waterprint_alpha=0.6):
    if waterprint_path is None:
        base_dir = os.path.dirname(__file__)
        waterprint_path = os.path.join(base_dir, 'data/waterprint.png')
    water_print_img = Image.open(waterprint_path)
    water_print_grey = np.array(water_print_img)[:, :, 3]
    water_print_img = cv2.cvtColor(water_print_grey, cv2.COLOR_GRAY2RGB)

    length, width, height, fps = get_video_info(video_path)
    water_print_img = cv2.resize(water_print_img, (width, height))

    def frames_generator():
        for image in read_video(video_path):
            combined_image = cv2.addWeighted(image, 1, water_print_img, waterprint_alpha, 0)
            yield combined_image
    generate_output_video(frames_generator(), output_path, width, height, output_fps=fps)
