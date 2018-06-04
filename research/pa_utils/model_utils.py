import time
import numpy as np
import tensorflow as tf
from object_detection.utils import ops as utils_ops


def get_key(key, useRT=False):
    if useRT:
        return 'import/' + key
    return key


def run_inference_for_single_image(sess, image, graph=None, detect_mask=False, feature_map=False, useRT=False):
    if graph is None:
        graph = tf.get_default_graph()

    # Get handles to input and output tensors
    ops = graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    tensor_keys = [
      'num_detections', 'detection_boxes', 'detection_scores',
      'detection_classes'
    ]
    if detect_mask:
        tensor_keys.append('detection_masks')
    if feature_map:
        feature_map_op = 'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block3/unit_23/bottleneck_v1/Relu'
        tensor_keys.append(feature_map_op)

    for key in tensor_keys:
        tensor_name = key + ':0'
        if get_key(tensor_name, useRT) in all_tensor_names:
            tensor_dict[key] = graph.get_tensor_by_name(get_key(tensor_name, useRT))
        else:
            print('tensor %s not exists' % get_key(tensor_name, useRT))
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Fol`low the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = graph.get_tensor_by_name(get_key('image_tensor:0', useRT))

    # Run inference
    output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: np.expand_dims(image, 0)})
    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


