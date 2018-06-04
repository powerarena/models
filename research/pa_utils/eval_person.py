import functools
import logging
import os
import time
import cv2
import numpy as np
import tensorflow as tf
from pa_utils.model_utils import run_inference_for_single_image
from pa_utils.image_utils import label_image
from pa_utils.data.data_utils import remove_detection
from pa_utils.data.prepare_person_data import get_shanghai_data
from pa_utils.data.data_utils import resize_image
from object_detection.builders import model_builder, dataset_builder
from object_detection.evaluator import _extract_predictions_and_losses, get_evaluators
from object_detection.utils import config_util, dataset_util, label_map_util

logging.basicConfig(format='%(asctime)s %(message)s')
logging.getLogger().setLevel(logging.INFO)


class MAE(object):
    def __init__(self):
        self.n = 0
        self.sum = 0
        self.result = dict()

    def add_single_ground_truth_image_info(self, image_id, groundtruth_dict):
        if image_id not in self.result:
            self.result[image_id] = [None, None]
        if isinstance(groundtruth_dict, int) or isinstance(groundtruth_dict, float):
            self.result[image_id][0] = groundtruth_dict
        else:
            self.result[image_id][0] = (groundtruth_dict['groundtruth_classes']==1).sum()

    def add_single_detected_image_info(self, image_id, detections_dict):
        if image_id not in self.result:
            self.result[image_id] = [None, None]
        if isinstance(detections_dict, int) or isinstance(detections_dict, float):
            self.result[image_id][1] = detections_dict
        else:
            self.result[image_id][1] = (detections_dict['detection_classes']==1).sum()

    def add(self, actual_value, forcast_value):
        self.n += 1
        self.sum += abs(actual_value - forcast_value)

    def evaluate(self):
        for v1, v2 in self.result.values():
            if v1 is not None and v2 is not None:
                self.add(v1, v2)

        mae = 0
        if self.n > 0:
            mae = self.sum/self.n
        return {self.__class__.__name__: mae}

    def clear(self):
        self.sum = 0
        self.n = 0
        self.result = dict()


class MSE(MAE):
    def add(self, actual_value, forcast_value):
        self.n += 1
        self.sum += (actual_value - forcast_value)**2


class SMAPE(MAE):
    def add(self, actual_value, forcast_value):
        if abs(actual_value+forcast_value) > 0:
            self.n += 1
            self.sum += 2 * abs(actual_value - forcast_value)/(actual_value+forcast_value)
        else:
            logging.info('**actual_value+forcast_value %s, %s' % (actual_value, forcast_value))


class MAPE(MAE):
    def add(self, actual_value, forcast_value):
        if actual_value > 0:
            self.n += 1
            self.sum += abs(actual_value - forcast_value)/(actual_value)


def eval_tf_input(pipeline_config_path):
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']
    model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=False)
    model = model_fn()

    def get_next(config):
        return dataset_util.make_initializable_iterator(
            dataset_builder.build(config)).get_next()
    input_config = configs['eval_input_config']
    create_input_dict_fn = functools.partial(get_next, input_config)

    tensor_dict = _extract_predictions_and_losses(model,create_input_dict_fn,ignore_groundtruth=False)

    eval_config = configs['eval_config']

    label_map = label_map_util.load_labelmap(input_config.label_map_path)
    max_num_classes = max([item.id for item in label_map.item])
    categories = label_map_util.convert_label_map_to_categories(
          label_map, max_num_classes)

    num_batches = eval_config.num_examples
    evaluators = get_evaluators(eval_config, categories)

    ### prepare session by checkpoint ###
    variables_to_restore = tf.global_variables()
    global_step = tf.train.get_or_create_global_step()
    variables_to_restore.append(global_step)

    if eval_config.use_moving_averages:
        variable_averages = tf.train.ExponentialMovingAverage(0.0)
        variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    sess = tf.Session(eval_config.eval_master, graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    saver.restore(sess, latest_checkpoint)
    ### prepare session by checkpoint ###

    with tf.contrib.slim.queues.QueueRunners(sess):
        for batch in range(int(num_batches)):
            if (batch + 1) % 10 == 0:
                logging.info('Running eval ops batch %d/%d', batch + 1, num_batches)
            result_dict = sess.run(tensor_dict)
            if not result_dict:
                continue
            for evaluator in evaluators:
                evaluator.add_single_ground_truth_image_info(
                    image_id=batch, groundtruth_dict=result_dict)
                evaluator.add_single_detected_image_info(
                    image_id=batch, detections_dict=result_dict)

        all_evaluator_metrics = dict()
        for evaluator in evaluators:
            metrics = evaluator.evaluate()
            evaluator.clear()
            if any(key in all_evaluator_metrics for key in metrics):
                raise ValueError('Metric names between evaluators must not collide.')
            all_evaluator_metrics.update(metrics)
        print(all_evaluator_metrics)


def eval_inference_graph(sess, pipeline_config_path, output_eval_path, min_score_thresh=.5, wait_ms=1, useRT=False):
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)

    # prepare input
    def get_next(config):
        print('config', dataset_builder.build(config))
        return dataset_util.make_initializable_iterator(
            dataset_builder.build(config)).get_next()
    input_config = configs['eval_input_config']
    create_input_dict_fn = functools.partial(get_next, input_config)

    # prepare evaluators
    label_map = label_map_util.load_labelmap(input_config.label_map_path)
    max_num_classes = max([item.id for item in label_map.item])
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes)
    category_index = label_map_util.create_category_index(categories)
    eval_config = configs['eval_config']
    print(type(eval_config), eval_config)
    evaluators = get_evaluators(eval_config, categories)
    evaluators.extend([MAE(), MSE(), SMAPE(), MAPE()])
    accumulate_time = 0
    num_of_images = 0
    num_of_val_images = eval_config.num_examples
    validate_image_ids = set()
    with sess:
      with tf.Session() as input_sess:
        input_tensor_dict = create_input_dict_fn()
        with tf.contrib.slim.queues.QueueRunners(input_sess):
            input_sess.run(tf.global_variables_initializer())
            input_sess.run(tf.local_variables_initializer())
            input_sess.run(tf.tables_initializer())

            for idx in range(num_of_val_images):
                if idx % 10 == 0:
                    logging.info('processing %s/%s' % (idx, num_of_val_images))
                input_dict = input_sess.run(input_tensor_dict)

                image = input_dict['image']

                # Run inference
                start_time = time.time()
                output_dict = run_inference_for_single_image(sess, image, useRT=useRT)
                accumulate_time += time.time() - start_time
                num_of_images += 1

                remove_detection(output_dict, min_score_thresh=min_score_thresh)
                for evaluator in evaluators:
                    evaluator.add_single_ground_truth_image_info(
                        image_id=input_dict['source_id'], groundtruth_dict=input_dict)
                    evaluator.add_single_detected_image_info(
                        image_id=input_dict['source_id'], detections_dict=output_dict)
                print('num_detections', output_dict['num_detections'], (input_dict['groundtruth_classes'] == 1).sum())

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                label_image(image, output_dict, category_index, min_score_thresh=min_score_thresh)
                cv2.imshow('frame', image)
                if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                    break
                # validate_image_ids.add(input_dict['source_id'])
    all_evaluator_metrics = dict()
    for evaluator in evaluators:
        metrics = evaluator.evaluate()
        try:
            if isinstance(evaluator, MAE) and output_eval_path:
                with open(output_eval_path, 'w') as fw:
                    for k in sorted(evaluator.result.keys()):
                        fw.write('%s\t%s\t%s\n' % (k, evaluator.result[k][0], evaluator.result[k][1]))
        except:
            logging.error('error while output eval')
        # evaluator.clear()
        if any(key in all_evaluator_metrics for key in metrics):
            raise ValueError('Metric names between evaluators must not collide.')
        all_evaluator_metrics.update(metrics)
    print(all_evaluator_metrics)
    print('Second/Image = %s/%s = %s' % (accumulate_time, num_of_images, accumulate_time/num_of_images))
    print('validate_image_ids', len(validate_image_ids))
    print('\t'.join([str(all_evaluator_metrics[x]) for x in ['MAE', 'MSE', 'SMAPE', 'MAPE', 'PascalBoxes_PerformanceByCategory/AP@0.5IOU/person']]))


def eval_shanghaitech(sess, eval_result_path, show_image=False, max_people_count=0, min_score_thresh=.4, wait_ms=1):
    # prepare evaluators
    label_map_path = '/app/powerarena-sense-gym/models/research/object_detection/data/person_label_map.pbtxt'
    label_map = label_map_util.load_labelmap(label_map_path)
    max_num_classes = max([item.id for item in label_map.item])
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes)
    category_index = label_map_util.create_category_index(categories)

    fw = None
    if eval_result_path:
        fw = open(eval_result_path, 'w')

    accumulate_time = 0
    num_of_images = 0
    sum_gt_person = 0
    evaluators = [MAE(), MSE(), SMAPE(), MAPE()]
    with sess:
        for idx, (image, points) in enumerate(get_shanghai_data()):
            if max_people_count > 0 and points.shape[0] > max_people_count:
                continue
            sum_gt_person += points.shape[0]
            num_of_images += 1
            start = time.time()
            # image = image[:, :400, :]
            output_dict = run_inference_for_single_image(sess, resize_image(image, 600, 1024))
            accumulate_time += time.time() - start
            # #### only person
            # output_dict['num_detections'] = max(output_dict['num_detections'], output_dict['detection_scores'].shape[0])
            # person_indices = np.where(output_dict['detection_classes'] != 1)
            # output_dict['num_detections'] -= output_dict['detection_classes'][person_indices].size
            # output_dict['detection_classes'] = np.delete(output_dict['detection_classes'], person_indices, axis=0)
            # output_dict['detection_boxes'] = np.delete(output_dict['detection_boxes'], person_indices, axis=0)
            # output_dict['detection_scores'] = np.delete(output_dict['detection_scores'], person_indices, axis=0)
            # ####
            remove_detection(output_dict, min_score_thresh=min_score_thresh)
            predicted_count = float((output_dict['detection_classes'] == 1).sum())
            time_elasped = time.time() - start
            print(time_elasped, predicted_count, points.shape[0])
            for evaluator in evaluators:
                evaluator.add_single_ground_truth_image_info(str(idx), points.shape[0])
                evaluator.add_single_detected_image_info(str(idx), predicted_count)
            if fw:
                fw.write('%.4f\t%.1f\t%.1f\n' % (time_elasped, predicted_count, points.shape[0]))
                fw.flush()
            if show_image:
                label_image(image, output_dict, category_index, min_score_thresh=min_score_thresh)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imshow('frame', image)
                key_pressed = cv2.waitKey(wait_ms) & 0xFF
                if key_pressed == ord('q'):
                    break
                elif key_pressed == ord('w'):
                    input('Press any key in the console to continue.')
                    # time.sleep(5)

    all_evaluator_metrics = dict()
    for evaluator in evaluators:
        metrics = evaluator.evaluate()
        all_evaluator_metrics.update(metrics)
    print(all_evaluator_metrics)
    print('\t'.join([str(all_evaluator_metrics[x]) for x in ['MAE', 'MSE', 'SMAPE', 'MAPE']]))
    print('Second/Image = %s/%s = %s' % (accumulate_time, num_of_images, accumulate_time / num_of_images))
    print('Avg Person = %s' % (sum_gt_person/num_of_images))


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    from pa_utils.graph_utils import get_session

    useRT = True
    min_score_thresh = .5
    base_folder = os.path.dirname(__file__)

    # loop_input(pipeline_config_path, input_type='eval', wait_time=1)

    # original person model
    model_folder = 'original_inception_resnet'
    inference_graph_path = '/app/object_detection_app/models/person_inceptionv2_20180102.pb'
    pipeline_config_path = os.path.join(base_folder, 'configs/person/faster_rcnn_inception_resnet_v2_atrous.config')

    # train person model
    model_folder = 'faster_rcnn_resnet101'
    train_version = 'train_aa_v3'
    checkpoint_dir = os.path.join(base_folder, 'model_output/person/%s/%s/' % (model_folder, train_version))
    inference_graph_path = os.path.join(checkpoint_dir, 'inference/frozen_inference_graph.pb')

    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    graph, sess = get_session(inference_graph_path=inference_graph_path, config=session_config, useRT=useRT)
    # graph, sess = get_session(checkpoint_dir=checkpoint_dir)

    output_eval_path = os.path.join(os.path.dirname(__file__), 'eval/person_voc2012_%s.txt' % model_folder)
    eval_inference_graph(sess, pipeline_config_path, output_eval_path, min_score_thresh=min_score_thresh, wait_ms=1, useRT=useRT)

    # file_path = os.path.join(os.path.dirname(__file__), 'eval/person_shanghai_%s.txt' % model_folder)
    # eval_shanghaitech(sess, file_path, max_people_count=60, show_image=True, min_score_thresh=min_score_thresh, wait_ms=1)
