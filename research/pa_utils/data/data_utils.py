import os
import cv2
import numpy as np
import logging
import matplotlib
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


def get_video_info(video_path, known_fps=None):

    video = cv2.VideoCapture(video_path)
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver) < 3:
        length = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
    print("length: {0}, width: {1}, height: {2}, fps: {3}, known_fps: {4}".format(length, width, height, fps, known_fps))
    if known_fps:
        fps = known_fps
    else:
        fps = int(fps)
    return length, width, height, fps


def read_video(video_path, frame_freq=None, output_minmax_size=None):
    video = cv2.VideoCapture(video_path)
    frame_idx = 0
    while (video.isOpened()):
        ret, frame = video.read()
        if ret:
            if frame_freq is None or frame_idx % frame_freq < 1:
                if output_minmax_size is None:
                    yield frame
                else:
                    yield resize_image(frame, *output_minmax_size)
        else:
            break
        frame_idx += 1
    video.release()
    cv2.destroyAllWindows()


def get_image_reader(image_dir=None, video_path=None, video_fps=1, max_video_length=0, output_minmax_size=None):
    if image_dir:
        for idx, image_file in enumerate(sorted(os.listdir(image_dir))):
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            if output_minmax_size is None:
                yield image
            else:
                yield resize_image(image, *output_minmax_size)
    elif video_path:
        length, width, height, fps = get_video_info(video_path)
        frame_freq = fps / video_fps
        for idx, image in enumerate(read_video(video_path, frame_freq=frame_freq, output_minmax_size=output_minmax_size)):
            if not max_video_length or idx < video_fps*max_video_length:
                yield image
            else:
                break


def generate_output_video(frames, output_path, width, height, output_fps=1):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
    for frame in frames:
        video_writer.write(frame)
        cv2.imshow('', frame)
        cv2.waitKey(1)
    video_writer.release()


def create_label_categories(label_path, num_classes):
    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return category_index


def remove_detection(output_dict, filter_classes=None, min_score_thresh=None, filter_indice=None, retain_indice=None):
    output_dict['num_detections'] = output_dict['detection_scores'].shape[0]
    if retain_indice:
        output_dict['num_detections'] = len(retain_indice)
        output_dict['detection_classes'] = output_dict['detection_classes'][retain_indice]
        output_dict['detection_boxes'] = output_dict['detection_boxes'][retain_indice]
        output_dict['detection_scores'] = output_dict['detection_scores'][retain_indice]
    elif filter_indice:
        output_dict['num_detections'] -= output_dict['detection_classes'][filter_indice].size
        output_dict['detection_classes'] = np.delete(output_dict['detection_classes'], filter_indice, axis=0)
        output_dict['detection_boxes'] = np.delete(output_dict['detection_boxes'],  filter_indice, axis=0)
        output_dict['detection_scores'] = np.delete(output_dict['detection_scores'],  filter_indice, axis=0)

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


def get_pascal_xml_data(xml_path):
    import tensorflow as tf
    from lxml import etree
    from object_detection.utils import dataset_util

    with tf.gfile.GFile(xml_path, 'r') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    # print(data['object'][0])
    for obj in data['object']:
        bbox = obj['bndbox']
        obj['bndbox'] = dict(xmin=int(bbox['xmin']),
                             ymin=int(bbox['ymin']),
                             xmax=int(bbox['xmax']),
                             ymax=int(bbox['ymax']))
        xmin, ymin, xmax, ymax = obj['bndbox']['xmin'], obj['bndbox']['ymin'], obj['bndbox']['xmax'], obj['bndbox']['ymax']
        center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
        obj['center'] = center
    return data['size'], data['path'], data['object']


def get_color_list(return_rgb=False):
    color_list = sorted(list(matplotlib.colors.cnames.keys()))
    color_list.remove('azure')
    color_list.remove('beige')
    color_list.remove('black')
    # color_list.remove('antiquewhite')
    color_list = [c for idx, c in enumerate(color_list) if idx % 3 == 0]
    if return_rgb:
        color_list = map(lambda x: matplotlib.colors.cnames[x], color_list)
        return list(map(lambda x: tuple(int(x.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)), color_list))
    return color_list

# import json
# import time
# jdata = dict()
# data = get_pascal_xml_data('/app/powerarena-sense-gym/client_dataset/Cerie/cerie_hd-MVI_8580-00865-00.xml')
# jdata = data[0]
# jdata['timestamp'] = int(time.time())
# jdata['detections'] = []
# for ann in data[2]:
#     jdata['detections'].append(dict(label=ann['name'], bbox=[
#         float(ann['bndbox']['ymin'])/float(jdata['height']),
#         float(ann['bndbox']['xmin'])/float(jdata['width']),
#         float(ann['bndbox']['ymax'])/float(jdata['height']),
#         float(ann['bndbox']['xmax'])/float(jdata['width']),
#     ]))
# json.dump(jdata, open('table_detection.json', 'w'))