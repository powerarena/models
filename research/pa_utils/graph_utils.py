import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.exporter import _image_tensor_input_placeholder, _add_output_tensor_nodes


def get_session(inference_graph_path=None, checkpoint_dir=None):
    if inference_graph_path:
        graph = create_inference_graph(inference_graph_path)
        with graph.as_default():
            return tf.Session()
    else:
        pipeline_config_path = os.path.join(checkpoint_dir, 'pipeline.config')
        return get_checkpoint_session(pipeline_config_path, checkpoint_dir)


def get_checkpoint_session(pipeline_config_path, checkpoint_dir):
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

    sess = tf.Session(eval_config.eval_master, graph=tf.get_default_graph())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())

    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    saver.restore(sess, latest_checkpoint)
    return sess


def create_inference_graph(inference_graph_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph
