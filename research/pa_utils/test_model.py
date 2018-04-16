import os
import numpy as np
import tensorflow as tf
import cv2
from pa_utils.graph_utils import get_session
from pa_utils.image_utils import label_image
from pa_utils.data.data_utils import get_video_info, read_video, create_label_categories
from pa_utils.model_utils import run_inference_for_single_image


def test_detection_model(sess, label_path, image_dir=None, video_path=None, min_score_thresh=.5):
    category_index = create_label_categories(label_path, 100)

    def _run(image):
        output_dict = run_inference_for_single_image(sess, image)
        label_image(image, output_dict, category_index, min_score_thresh=min_score_thresh)
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    with sess:
        if image_dir:
            for image_file in os.listdir(image_dir):
                image_path = os.path.join(image_dir, image_file)
                image = cv2.imread(image_path)
                _run(image)
        elif video_path:
            for frame in read_video(video_path):
                _run(frame)

    cv2.destroyAllWindows()


def detect_vision(sess, label_path, video_path, output_path, min_score_thresh=.5, output_fps=10):
    category_index = create_label_categories(label_path, 100)
    length, width, height, fps = get_video_info(video_path)
    print('length %s, width %s, height %s, fps %s' % (length, width, height, fps))
    frame_freq = int(fps / output_fps)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    with sess:
        for frame in read_video(video_path, frame_freq=frame_freq):
            output_dict = run_inference_for_single_image(sess, frame)
            label_image(frame, output_dict, category_index, min_score_thresh=min_score_thresh)
            video_writer.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

    video_writer.release()


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    image_dir = '/app/powerarena-sense-gym/people_dataset/ShanghaiTech/part_B/train_data/images'
    video_path = '/home/ma-glass/Downloads/泉塘派出所前园盘X东-白天.mp4'

    base_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'object_detection')
    label_path = os.path.join(base_folder, 'data/person_label_map.pbtxt')

    # checkpoint_dir = '/app/object_detection_app/models'
    # inference_graph_path = os.path.join(checkpoint_dir, 'person_inceptionv2_20180102.pb')

    # model_folder = 'rfcn_resnet101'
    # checkpoint_dir = os.path.join(base_folder, 'output/person/%s/train/' % model_folder)
    # inference_graph_path = os.path.join(checkpoint_dir, 'inference/frozen_inference_graph.pb')

    # sess = get_session(inference_graph_path=inference_graph_path)
    # sess = get_session(checkpoint_dir=checkpoint_dir)
    # test_detection_model(sess, label_path, image_dir=image_dir)

    inference_graph_path = '/Dataset/Pretrained/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb'
    label_path = os.path.join(base_folder, 'data/mscoco_label_map.pbtxt')
    sess = get_session(inference_graph_path=inference_graph_path)
    test_detection_model(sess, label_path, video_path=video_path)
    # output_path = '/home/ma-glass/Downloads/泉塘派出所前园盘X东-白天(VA).mp4'
    # detect_vision(sess, label_path, video_path, output_path)