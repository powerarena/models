from object_detection.utils import visualization_utils as vis_util


def label_image(image_np, output_dict, category_index=None, min_score_thresh=0.5):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    # image_np_expanded = np.expand_dims(image_np, axis=0)

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index or dict(),
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=1,
        max_boxes_to_draw=100,
        min_score_thresh=min_score_thresh)