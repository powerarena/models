import cv2
import numpy as np
from object_detection.utils import label_map_util


def resize_image(image, min_dimension, max_dimension):
    image_shape = image.shape
    orig_height = image_shape[0]
    orig_width = image_shape[1]
    num_channels = image_shape[2]
    orig_min_dim = min(orig_height, orig_width)
    # Calculates the larger of the possible sizes
    large_scale_factor = min_dimension / orig_min_dim
    # Scaling orig_(height|width) by large_scale_factor will make the smaller
    # dimension equal to min_dimension, save for floating point rounding errors.
    # For reasonably-sized images, taking the nearest integer will reliably
    # eliminate this error.
    large_height = round(orig_height * large_scale_factor)
    large_width = round(orig_width * large_scale_factor)
    large_size = [large_height, large_width]
    if max_dimension:
        # Calculates the smaller of the possible sizes, use that if the larger
        # is too big.
        orig_max_dim = max(orig_height, orig_width)
        small_scale_factor = max_dimension / orig_max_dim
        # Scaling orig_(height|width) by small_scale_factor will make the larger
        # dimension equal to max_dimension, save for floating point rounding
        # errors. For reasonably-sized images, taking the nearest integer will
        # reliably eliminate this error.
        small_height = round(orig_height * small_scale_factor)
        small_width = round(orig_width * small_scale_factor)
        small_size = [small_height, small_width]
        if max(large_size) > max_dimension:
            new_size = small_size
        else:
            new_size = large_size
    else:
        new_size = large_size
    return cv2.resize(image, (new_size[1], new_size[0]))


def get_video_info(video_path):
    video = cv2.VideoCapture(video_path)
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        length = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.cv.CV_CAP_PROP_FPS))
    else:
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
    print("length: {0}, width: {1}, height: {2}, fps: {3}".format(length, width, height, fps))
    return length, width, height, fps


def read_video(video_path, frame_freq=None):
    video = cv2.VideoCapture(video_path)
    frame_idx = 0
    while (video.isOpened()):
        ret, frame = video.read()
        if ret:
            if frame_freq is None or frame_idx % frame_freq == 0:
                yield frame
        else:
            break
        frame_idx += 1
    video.release()
    cv2.destroyAllWindows()


def create_label_categories(label_path, num_classes):
    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def remove_detection(output_dict, filter_classes=None, min_score_thresh=None):
    output_dict['num_detections'] = output_dict['detection_scores'].shape[0]
    if filter_classes:
        idx_arr = np.where(np.isin(output_dict['detection_classes'], filter_classes))
        output_dict['num_detections'] -= output_dict['detection_classes'][idx_arr].size
        output_dict['detection_classes'] = np.delete(output_dict['detection_classes'], idx_arr, axis=0)
        output_dict['detection_boxes'] = np.delete(output_dict['detection_boxes'],  idx_arr, axis=0)
        output_dict['detection_scores'] = np.delete(output_dict['detection_scores'],  idx_arr, axis=0)
    if min_score_thresh:
        idx_arr = np.where(output_dict['detection_scores'] < min_score_thresh)
        output_dict['num_detections'] -= output_dict['detection_classes'][idx_arr].size
        output_dict['detection_classes'] = np.delete(output_dict['detection_classes'], idx_arr, axis=0)
        output_dict['detection_boxes'] = np.delete(output_dict['detection_boxes'],  idx_arr, axis=0)
        output_dict['detection_scores'] = np.delete(output_dict['detection_scores'],  idx_arr, axis=0)
