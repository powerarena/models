import os
import numpy as np
import cv2
import logging
import time
import pickle
from PIL import Image
from pascal_voc_writer import Writer as pv_writer
from pa_utils.model_utils import run_inference_for_single_image
from pa_utils.data.data_utils import create_label_categories, remove_detection, get_color_list
from pa_utils.data.data_utils import resize_image, get_video_info, read_video, get_pascal_xml_data
from object_detection.utils.visualization_utils import draw_bounding_box_on_image


def get_image_diff_percentage(image1, image2, delta_threshold=25):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (21, 21), 0)
    delta = cv2.absdiff(gray1, gray2)
    num_diff_pixels = (delta > delta_threshold).sum()
    return 1.0*num_diff_pixels/delta.size


def generate_from_video(video_path, output_dir, image_prefix='',
                        output_minmax_size=None,
                        known_fps=None,
                        max_output_fps=1.0,
                        show_images=True,
                        skip_n_seconds=0,
                        max_video_length=0,
                        image_diff_threshold=None,
                        image_diff_minimum_second=0,
                        display_only=False,
                        wait_time=1,
                        prefix_exists_error=False
                        ):
    if not display_only and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if any(image_prefix in file_name for file_name in os.listdir(output_dir)):
        print('****Prefix %s already exists in %s!!!****' % (image_prefix, output_dir))
        if prefix_exists_error:
            raise Exception('PrefixExists')
    if skip_n_seconds and max_video_length:
        # video.set(cv2.CAP_PROP_POS_FRAMES, skip_n_seconds*fps)
        # frame_count += skip_n_seconds*fps
        max_video_length += skip_n_seconds
    print(video_path, 'video path exists:', os.path.exists(video_path))
    length, width, height, fps = get_video_info(video_path, known_fps=known_fps)
    last_frame = None
    last_frame_sec = 0
    current_sec = 0
    current_sec_write_count = 0
    start_time = time.time()
    frame_count = skip_n_seconds*fps - 1
    for frame in read_video(video_path, skip_frames=skip_n_seconds*fps):
        frame_count += 1
        sec = frame_count // fps
        if current_sec != sec:
            current_sec = sec
            current_sec_write_count = 0
        # # skip first n seconds
        # if frame_count < skip_n_seconds*fps:
        #     continue
        # max video seconds
        if sec > max_video_length > 0:
            break

        if frame_count % fps == 0:
            logging.info("video at %d sec" % sec)

        # write images
        create_image = False
        show_diff_image = True
        if image_diff_threshold is None:
            if frame_count % (fps/max_output_fps) < 1:
                create_image = True

                # check image diff
                if create_image:
                    if last_frame is not None:
                        print('image diff', get_image_diff_percentage(last_frame, frame))
                    last_frame = frame

        else:
            # #show image diff
            # if last_frame is not None:
            #     print(get_image_diff_percentage(last_frame, frame))
            if last_frame is None or (
                        current_sec_write_count < max_output_fps and
                        get_image_diff_percentage(last_frame, frame) > image_diff_threshold and
                        (image_diff_minimum_second == 0 or (current_sec - last_frame_sec) >= image_diff_minimum_second)
            ):
                create_image = True
                last_frame = frame
                last_frame_sec = sec
                current_sec_write_count += 1
            else:
                show_diff_image = False

        if create_image:
            sec_frame = frame_count % fps
            image_path = os.path.join(output_dir, '%s-%05d-%02d.jpg' % (image_prefix, sec, sec_frame))
            if not display_only:
                if output_minmax_size is not None:
                    cv2.imwrite(image_path, resize_image(frame, *output_minmax_size))
                else:
                    cv2.imwrite(image_path, frame)
            else:
                print('create image',image_path)
            logging.info('wrote frame at sec = %d' % sec)

        if show_images and show_diff_image and create_image:
            cv2.imshow('frame', resize_image(frame, *(540, 960)))
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
        if frame_count % (fps*30) == 0:
            processed_frames = frame_count + 1 - skip_n_seconds*fps
            processed_time = time.time() - start_time
            logging.info('processed fps = %.2f, processed spf = %.3f' % (processed_frames/processed_time, processed_time/processed_frames))


def generate_samples_images(dataset_dir, output_dir, label_classes=None):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    color_list = get_color_list()
    color_dict = dict()
    if label_classes is not None:
        for label_class in label_classes:
            color_dict[label_class] = color_list[len(color_dict)]
    print(color_dict)
    anno_dir = os.path.join(dataset_dir, 'Annotations')
    image_dir = os.path.join(dataset_dir, 'JPEGImages')
    for xml_file in os.listdir(anno_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(anno_dir, xml_file)
            _, _, bbox_objs = get_pascal_xml_data(xml_path)

            if len(bbox_objs) == 0:
                continue
            image_file = xml_file.replace('.xml', '.jpg')
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
            for bbox_obj in bbox_objs:
                label_class = bbox_obj['name']
                if label_class not in color_dict:
                    color_dict[label_class] = color_list[len(color_dict)]
                bbox = bbox_obj['bndbox']
                draw_bounding_box_on_image(image_pil,
                                           bbox['ymin'],
                                           bbox['xmin'],
                                           bbox['ymax'],
                                           bbox['xmax'],
                                           color=color_dict[label_class],
                                           thickness=4,
                                           font_size=20,
                                           display_str_list=(label_class,),
                                           use_normalized_coordinates=False)
            image = np.array(image_pil)
            output_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_path, image)
            # print(output_path, image.shape)


def generate_annotations_from_model(sess, label_path, image_dir, anno_dir, min_score_thresh=0.5):
    category_index = create_label_categories(label_path, 100)

    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)

    with sess:
        for image_file in sorted(os.listdir(image_dir)):
            image = cv2.imread(os.path.join(image_dir, image_file))

            output_dict = run_inference_for_single_image(sess, image)
            remove_detection(output_dict, min_score_thresh=min_score_thresh)
            writer = pv_writer('Annotation/' + image_file, image.shape[1], image.shape[0])
            rescaled_detection_boxes = output_dict['detection_boxes'] * [image.shape[0], image.shape[1], image.shape[0],image.shape[1]]
            sorted_detection_boxes = sorted(enumerate(rescaled_detection_boxes), key=lambda x: x[1][0])
            for idx, bbox in sorted_detection_boxes:
                if output_dict['detection_classes'][idx] in category_index.keys():
                    class_name = category_index[output_dict['detection_classes'][idx]]['name']
                else:
                    class_name = 'N/A'
                writer.addObject(class_name, int(round(bbox[1])), int(round(bbox[0])), int(round(bbox[3])), int(round(bbox[2])))

            annotation_path = os.path.join(anno_dir, image_file.replace('.jpg', '.xml'))
            writer.save(annotation_path)


def generate_detections_pickle(sess, image_reader, output_path):
    with sess:
        image_detections = []
        for idx, image in enumerate(image_reader):
            if idx % 10 == 0:
                print('processing image', idx + 1)
            output_dict = run_inference_for_single_image(sess, image)
            image_detections.append(output_dict)
        with open(output_path, 'wb') as fw:
            pickle.dump(image_detections, fw)


def load_detections_pickle(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def add_lines(image_dir, output_dir, cols=2, rows=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_file in sorted(os.listdir(image_dir)):
        image = cv2.imread(os.path.join(image_dir, image_file))
        for col_idx in range(1, 1+cols):
            point1 = (int(image.shape[1]*col_idx/(cols+1)), 0)
            point2 = (int(image.shape[1]*col_idx/(cols+1)), image.shape[0])
            cv2.line(image, point1, point2, (0, 0, 255))
        for row_idx in range(1, 1+rows):
            point1 = (0, int(image.shape[0]*row_idx/(rows+1)))
            point2 = (image.shape[1], int(image.shape[0]*row_idx/(rows+1)))
            cv2.line(image, point1, point2, (0, 0, 255))

        cv2.imwrite(os.path.join(output_dir, image_file), image)
        cv2.imshow('frame', cv2.resize(image, (1000, 500)))
        cv2.waitKey(1)

