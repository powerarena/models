import os
from pa_utils.graph_utils import get_session
from pa_utils.data.prepare_training_data import generate_detections_pickle, load_detections_pickle
from pa_utils.data.data_utils import get_image_reader


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    import logging
    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    video_path = ''

    inference_graph_path = '/app/object_detection_app/models/person_inceptionv2/model.pb'
    model_folder = 'faster_rcnn_resnet101'
    train_version = 'train_aa_v3'
    checkpoint_dir = os.path.join('/app/powerarena-sense-gym/models/research/pa_utils', 'model_output/person/%s/%s/' % (model_folder, train_version))

    label_path = os.path.join('label_maps/person_label_map.pbtxt')
    graph, sess = get_session(inference_graph_path=inference_graph_path)
    # graph, sess = get_session(checkpoint_dir=checkpoint_dir)

    image_reader = get_image_reader(video_path=video_path, video_fps=5, max_video_length=0, output_minmax_size=(1080, 1920))
    pickle_path = '/app/powerarena-sense-gym/models/research/pa_utils/project/aa/d1_detections_fps5_bgr.pkl'
    generate_detections_pickle(sess, image_reader, pickle_path)
    detections = load_detections_pickle(pickle_path)
    print(len(detections), detections[0])
