from pa_utils.data.prepare_training_data import generate_samples_images


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    dataset_dir = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/Dataset/Project/FEHD'
    output_dir = dataset_dir + '/Samples'
    label_classes = ['foam_boxes', 'cartons', 'furniture', 'debris', 'others', 'rubbish_bin']
    generate_samples_images(dataset_dir, output_dir, label_classes=label_classes)
