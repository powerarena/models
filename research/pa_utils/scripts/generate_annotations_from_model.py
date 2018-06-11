import os
from pa_utils.graph_utils import get_session
from pa_utils.data.prepare_training_data import generate_annotations_from_model


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    import logging
    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)


    inference_graph_path = '/app/object_detection_app/models/person_inceptionv2/model.pb'
    model_folder = 'faster_rcnn_resnet101'
    train_version = 'train_aa_v3'
    checkpoint_dir = os.path.join('/app/powerarena-sense-gym/models/research/pa_utils', 'model_output/person/%s/%s/' % (model_folder, train_version))

    label_path = os.path.join('label_maps/person_label_map.pbtxt')
    graph, sess = get_session(inference_graph_path=inference_graph_path)
    # graph, sess = get_session(checkpoint_dir=checkpoint_dir)

    image_dir = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/Dataset/Project/HKU/JPEGImages'
    anno_dir = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/Dataset/Project/HKU/Annotations'
    generate_annotations_from_model(sess, label_path, image_dir, anno_dir)
