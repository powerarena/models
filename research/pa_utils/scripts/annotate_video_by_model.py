import cv2
import numpy as np
from pa_utils.graph_utils import get_session
from pa_utils.image_utils import label_image
from pa_utils.data.data_utils import get_video_info, read_video, create_label_categories, resize_image
from pa_utils.model_utils import run_inference_for_single_image


def detect_vision(sess, label_path, video_path, output_path, min_score_thresh=.5, output_fps=10):
    category_index = create_label_categories(label_path, 100)
    length, width, height, fps = get_video_info(video_path)
    print('length %s, width %s, height %s, fps %s' % (length, width, height, fps))
    if output_fps > 0:
        frame_freq = fps / output_fps
    else:
        frame_freq = None
        output_fps = fps
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video_shape = (width, height)
    if width > 1920:
        new_frame = resize_image(np.zeros((height, width, 3)), 1080, 1920)
        output_video_shape = (new_frame.shape[1], new_frame.shape[0])
        print('reshape frame from', (width, height), 'to', output_video_shape)
    # VideoWriter fps must be integer >= 1
    video_writer = cv2.VideoWriter(output_path, fourcc, output_fps, output_video_shape)
    with sess:
        for idx, frame in enumerate(read_video(video_path, frame_freq=frame_freq)):
            print(idx)
            frame = cv2.resize(frame, output_video_shape)
            output_dict = run_inference_for_single_image(sess, frame)
            label_image(frame, output_dict, category_index, min_score_thresh=min_score_thresh)
            video_writer.write(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_writer.release()


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    base_folder = os.path.dirname(os.path.dirname(__file__))
    label_path = os.path.join(base_folder, 'data/label_maps/person_label_map.pbtxt')

    video_path = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/Downloads/18_4_2016 6_04_12 PM (UTC+08_00).avi'
    output_path = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/Downloads/aa_case7_apm.mp4'
    inference_graph_path = '/app/object_detection_app/models/person_inceptionv2/model.pb'
    graph, sess = get_session(inference_graph_path=inference_graph_path)
    detect_vision(sess, label_path, video_path, output_path, output_fps=0)
