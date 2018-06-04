import os
import numpy as np
import tensorflow as tf
import cv2
import functools
from pa_utils.graph_utils import get_session
from pa_utils.image_utils import label_image
from pa_utils.data.data_utils import get_video_info, read_video, create_label_categories, remove_detection, resize_image
from pa_utils.model_utils import run_inference_for_single_image
from object_detection.builders import dataset_builder
from object_detection.utils import config_util, dataset_util


def test_detection_model(sess, label_path, image_dir=None, video_path=None, video_fps=1, min_score_thresh=.5, wait_time=1, useRT=False, output_processor=None):
    category_index = create_label_categories(label_path, 100)

    def _run(image, image_title=None):
        if image.shape[1] > 1920:
            image = cv2.resize(image, (1920, 1080))

        output_dict = run_inference_for_single_image(sess, image, useRT=useRT)
        remove_detection(output_dict, min_score_thresh=min_score_thresh)
        if output_processor is not None:
            output_processor(output_dict)
        label_image(image, output_dict, category_index, min_score_thresh=min_score_thresh)
        if image_title:
            cv2.setWindowTitle('frame', image_title)
        cv2.imshow('frame', image)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            return

    with sess:
        if image_dir:
            for idx, image_file in enumerate(sorted(os.listdir(image_dir))):
                image_path = os.path.join(image_dir, image_file)
                image = cv2.imread(image_path)
                _run(image, image_title='%s:%s' % (idx + 1, image_file))
        elif video_path is not None:
            length, width, height, fps = get_video_info(video_path)
            frame_freq = int(fps / video_fps)
            for frame in read_video(video_path, frame_freq=frame_freq):
                _run(frame)

    cv2.destroyAllWindows()


def detect_vision(sess, label_path, video_path, output_path, min_score_thresh=.5, output_fps=10):
    category_index = create_label_categories(label_path, 100)
    length, width, height, fps = get_video_info(video_path)
    print('length %s, width %s, height %s, fps %s' % (length, width, height, fps))
    frame_freq = int(fps / output_fps)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_shape = (width, height)
    if width > 1920:
        new_frame = resize_image(np.zeros((height, width, 3)), 1080, 1920)
        output_video_shape = (new_frame.shape[1], new_frame.shape[0])
        print('reshape frame from', (width, height), 'to', output_video_shape)
    # VideoWriter fps must be integer >= 1
    video_writer = cv2.VideoWriter(output_path, fourcc, 1, output_video_shape)
    with sess:
        for frame in read_video(video_path, frame_freq=frame_freq):
            frame = cv2.resize(frame, output_video_shape)
            output_dict = run_inference_for_single_image(sess, frame)
            label_image(frame, output_dict, category_index, min_score_thresh=min_score_thresh)
            video_writer.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_writer.release()


def loop_input(pipeline_config_path, input_type='eval', wait_time=1):
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('TkAgg')

    # prepare input
    def get_next(config):
        print('config', dataset_builder.build(config))
        return dataset_util.make_initializable_iterator(
            dataset_builder.build(config)).get_next()

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    num_of_val_images = 100000
    if input_type == 'eval':
        input_config = configs['eval_input_config']
        num_of_val_images = configs['eval_config'].num_examples
    else:
        input_config = configs['train_input_config']
    create_input_dict_fn = functools.partial(get_next, input_config)

    print('num_of_val_images', num_of_val_images)
    all_images_ids = set()
    image_persons_count = []
    with tf.Session(config=None) as sess:
        input_tensor_dict = create_input_dict_fn()
        with tf.contrib.slim.queues.QueueRunners(sess):
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            sess.run(tf.tables_initializer())

            for idx in range(num_of_val_images):
                input_dict = sess.run(input_tensor_dict)
                if input_dict['source_id'] in all_images_ids:
                    break
                all_images_ids.add(input_dict['source_id'])
                image_persons_count.append((input_dict['groundtruth_classes'] == 1).sum())
                image = input_dict['image']
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.imshow('frame', image)
                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    break
                # time.sleep(1)

                # print(input_dict['filename'], input_dict['image'].shape)
                if idx % 100 == 0:
                   print('num of all_images_ids', idx + 1, len(all_images_ids))
    print('num of all_images_ids', len(all_images_ids))

    # x = np.array(image_persons_count)
    # n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
    # plt.grid(True)
    # plt.show()


def loop_video(video_path, output_fps=None, wait_time=1):
    length, width, height, fps = get_video_info(video_path)
    print('length %s, width %s, height %s, fps %s' % (length, width, height, fps))
    if output_fps:
        frame_freq = int(fps / output_fps)
    else:
        frame_freq = None
    for frame in read_video(video_path, frame_freq=frame_freq):
        frame = cv2.resize(frame, (1920, 1080))
        cv2.imshow('frame', frame)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break


def get_csv_outputer_lambda(output_path):
    import csv
    csv_writer = csv.DictWriter(open(output_path, 'w'), ['video second', 'number of people'])
    counter = dict(count=0)
    def _output_processor(output_dict, counter=counter):
        print(counter['count'], (output_dict['detection_classes'] == 1).sum())
        csv_writer.writerow({'video second': counter['count'], 'number of people': (output_dict['detection_classes'] == 1).sum()})
        counter['count'] += 1
    return _output_processor


if __name__ == '__main__':
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    useRT = False
    min_score_thresh = .5
    image_dir = '/app/powerarena-sense-gym/people_dataset/ShanghaiTech/part_B/test_data/images'
    image_dir = '/app/powerarena-sense-gym/client_dataset/Cerie/JPEGImages'
    image_dir = '/app/powerarena-sense-gym/models/research/pa_utils/project/aa/test_images'
    video_path = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/Dataset/Video/hku/IMG_2893.MOV'
    video_path = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/Dataset/Video/Cerie/New_HD/MVI_8580.MP4'
    video_path = '/home/ma-glass/Downloads/D1.mp4'
    # video_path = '/home/ma-glass/Downloads/Bird Feeding New 1.MOV'
    # video_path = '/home/ma-glass/Downloads/MK Large Object.mp4'

    base_folder = os.path.dirname(__file__)
    label_path = os.path.join(base_folder, '../object_detection/data/person_label_map.pbtxt')
    # label_path = os.path.join(base_folder, 'data/label_maps/cerie_label_map.pbtxt')
    # label_path = os.path.join(base_folder, 'data/label_maps/person_label_map.pbtxt')

    # Coco model
    # inference_graph_path = '/Dataset/Pretrained/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb'
    # label_path = os.path.join(base_folder, 'data/mscoco_label_map.pbtxt')

    # original person model
    inference_graph_path = '/app/object_detection_app/models/person_inceptionv2/model.pb'

    # train person model
    model_folder = 'faster_rcnn_resnet101'
    train_version = 'train_aa_v3'
    checkpoint_dir = os.path.join(base_folder, 'model_output/person/%s/%s/' % (model_folder, train_version))
    # inference_graph_path = os.path.join(checkpoint_dir, 'inference/frozen_inference_graph.pb')
    # pipeline_config_path = os.path.join(base_folder, 'configs/person/faster_rcnn_inception_resnet_v2_atrous.config')
    pipeline_config_path = None

    # train cerie model
    model_folder = 'faster_rcnn_resnet101'
    train_version = 'train_v3'
    # checkpoint_dir = os.path.join(base_folder, 'model_output/cerie/%s/%s/' % (model_folder, train_version))
    # pipeline_config_path = os.path.join(base_folder, 'configs/cerie/faster_rcnn_resnet101.config')
    # checkpoint_dir = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/DataScience/ObjectDetection/TrainObjectDetection/train_cerie'
    # label_path = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/DataScience/ObjectDetection/TrainObjectDetection/data/cerie/training/pascal_label_map.pbtxt'

    # train fehd model
    model_folder = 'faster_rcnn_resnet101_bird'
    # model_folder = 'faster_rcnn_resnet101_garbage'
    train_version = 'train'
    # checkpoint_dir = os.path.join(base_folder, 'model_output/fehd/%s/%s/' % (model_folder, train_version))
    # inference_graph_path = os.path.join(checkpoint_dir, 'inference/frozen_inference_graph.pb')
    # label_path = os.path.join(base_folder, 'data/label_maps/fehd_garbage.pbtxt')
    # label_path = os.path.join(base_folder, 'data/label_maps/fehd_bird.pbtxt')

    # loop_input(pipeline_config_path=pipeline_config_path)

    # checkpoint_dir = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/DataScience/ObjectDetection/Cerie/train'
    # inference_graph_path = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/DataScience/ObjectDetection/Cerie/frozen_inference_graph.pb'
    # video_path = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/Dataset/Video/Cerie/New/ch20_20171020153610.mp4'
    # label_path = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/DataScience/ObjectDetection/Cerie/cerie_label_map.pbtxt'

    # video_path = 0
    # checkpoint_dir = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/DataScience/ObjectDetection/TrainObjectDetection/train_person_faster'
    # inference_graph_path = '/app/object_detection_app/models/person_renet101_20180423/frozen_inference_graph.pb'

    # session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.4))
    # graph, sess = get_session(inference_graph_path=inference_graph_path, config=None, useRT=useRT)
    graph, sess = get_session(checkpoint_dir=checkpoint_dir, pipeline_config_path=None, config=None)

    # test_detection_model(sess, label_path, image_dir=image_dir, min_score_thresh=min_score_thresh, wait_time=0)
    test_detection_model(sess, label_path, video_path=video_path, video_fps=1/3, min_score_thresh=min_score_thresh, wait_time=1)
    # test_detection_model(sess, label_path, image_dir=image_dir, min_score_thresh=min_score_thresh, wait_time=0)
    # test_detection_model(sess, label_path, video_path=video_path, video_fps=1, min_score_thresh=min_score_thresh, wait_time=1, useRT=useRT, output_processor=None)

    # output_path = '/home/ma-glass/Downloads/泉塘派出所前园盘X东-白天(VA).mp4'
    # detect_vision(sess, label_path, video_path, output_path)
    output_path = '/home/ma-glass/Documents/cerie_MVI_8580(VA).mp4'
    # detect_vision(sess, label_path, video_path, output_path, min_score_thresh=.5, output_fps=1/10)

    # loop_video(video_path)