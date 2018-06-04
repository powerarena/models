import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.exporter import _image_tensor_input_placeholder, _add_output_tensor_nodes


def get_session(inference_graph_path=None, checkpoint_dir=None, pipeline_config_path=None, config=None, gpu_device=0, useRT=False):
    if config is None:
        config = tf.ConfigProto(allow_soft_placement=True)
    device = "/gpu:%d" % gpu_device if gpu_device >= 0 else "/cpu:0"

    if inference_graph_path:
        if useRT:
            graph = create_tensorrt_inference_graph(inference_graph_path)
        else:
            graph = create_inference_graph(inference_graph_path, device)
        return graph, tf.Session(config=config, graph=graph)
    else:
        if pipeline_config_path is None:
            pipeline_config_path = os.path.join(checkpoint_dir, 'pipeline.config')
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            return detection_graph, get_checkpoint_session(pipeline_config_path, checkpoint_dir, config=config)


def get_checkpoint_session(pipeline_config_path, checkpoint_dir, config=None):
    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    ### create input & output tensor ###
    output_collection_name = 'inference_op'
    _, input_tensors = _image_tensor_input_placeholder()
    inputs = tf.to_float(input_tensors)
    preprocessed_inputs, true_image_shapes = detection_model.preprocess(inputs)
    output_tensors = detection_model.predict(
        preprocessed_inputs, true_image_shapes)
    postprocessed_tensors = detection_model.postprocess(
        output_tensors, true_image_shapes)
    outputs = _add_output_tensor_nodes(postprocessed_tensors,
                                       output_collection_name)

    eval_config = configs['eval_config']

    ### prepare session by checkpoint ###
    variables_to_restore = tf.global_variables()
    global_step = tf.train.get_or_create_global_step()
    variables_to_restore.append(global_step)

    if eval_config.use_moving_averages:
        variable_averages = tf.train.ExponentialMovingAverage(0.0)
        variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    sess = tf.Session(eval_config.eval_master, graph=tf.get_default_graph(), config=config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    saver.restore(sess, latest_checkpoint)
    return sess


def create_inference_graph(inference_graph_path, device):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.device(device):
            with tf.gfile.GFile(inference_graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    return detection_graph


def create_tensorrt_inference_graph(inference_graph_path):
    from tensorflow.contrib import tensorrt as trt

    with tf.gfile.GFile(inference_graph_path, 'rb') as fid:
        od_graph_def = tf.GraphDef()
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
    output_nodes = ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']
    trt_graph = trt.create_inference_graph(od_graph_def, output_nodes,
                                           max_batch_size=300,
                                           max_workspace_size_bytes=1 << 32,
                                           precision_mode="FP32")

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        result = tf.import_graph_def(graph_def=trt_graph,
                            # input_map={'image_tensor:0': iterator.get_next()},
                            return_elements=output_nodes)
        print(result)
    return detection_graph
