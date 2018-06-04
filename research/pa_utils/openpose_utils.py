import PyOpenPose as OP
import numpy as np
import time
import cv2
import os

# OPENPOSE_ROOT = os.environ["OPENPOSE_ROOT"]
OPENPOSE_ROOT = '/usr/local/include/openpose'


def get_openpose_model(pose_network=(-1, 368), face_hand_network=(240, 240), output_resolution=(-1, -1), with_face=False, with_hand=False):
    model_type = 'COCO'
    model_folder = OPENPOSE_ROOT + os.sep + "models" + os.sep
    log_level = 0
    download_heatmaps = False

    op = OP.OpenPose(pose_network, face_hand_network, output_resolution, model_type, model_folder,
                     log_level, download_heatmaps, OP.OpenPose.ScaleMode.ZeroToOne, with_face, with_hand)
    return op


        # op.detectPose(frame)
        # op.detectFace(rgb)
        # op.detectHands(rgb)
        # res = op.render(rgb)
        # persons = op.getKeypoints(op.KeypointType.POSE)[0]


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    from pa_utils.data.data_utils import resize_image

    openpose = get_openpose_model(pose_network=(-1, 368))
    image_path = '/app/powerarena-sense-gym/models/research/pa_utils/data/image_samples/defond/defond-00166-20.jpg'
    image_folder = '/app/powerarena-sense-gym/models/research/pa_utils/data/image_samples/defond'
    time_wait = initial_time_wait = 100
    is_pause = False

    # AA
    image_dir = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/Dataset/Project/AA/D1/JPEGImages'
    image_path = os.path.join(image_dir, 'aa-D1-00300-00.jpg')

    image = cv2.imread(image_path)
    image = resize_image(image, 600, 1024)
    start = time.time()
    openpose.detectPose(image)
    print(time.time() - start)
    people_pose = openpose.getKeypoints(openpose.KeypointType.POSE)[0]
    print(people_pose[:, 1])
    print(people_pose[:, 6])
    print(people_pose[:, 7])
    image = openpose.render(image)
    cv2.imshow('', image)
    cv2.waitKey()

    def waitKey(is_pause, time_wait):
        key_pressed = cv2.waitKey(time_wait) & 0xFF
        time_wait = initial_time_wait
        if key_pressed == ord('p'):
            is_pause = not is_pause
            if is_pause:
                time_wait = 0
        return is_pause, time_wait

    # for image_file in list(sorted(os.listdir(image_folder))):
    #     if image_file.endswith('.jpg'):
    #         image_path = os.path.join(image_folder, image_file)
    #         image = cv2.imread(image_path)
    #         # image = image[int(image.shape[0]/2):, 400:600, :]
    #         # print(image.shape)
    #         frame = cv2.resize(image, (int(image.shape[1]), int(image.shape[0])))
    #         start = time.time()
    #         openpose.detectPose(frame)
    #         print('Time cost:', time.time() - start)
    #         # op.detectFace(frame)
    #         # op.detectHands(frame)
    #         # cv2.imshow('', image)
    #         # is_pause, time_wait = waitKey(is_pause, time_wait)
    #         # rendered_frame = openpose.render(frame)
    #         # cv2.imshow('', cv2.resize(rendered_frame, (image.shape[1], image.shape[0])))
    #         # is_pause, time_wait = waitKey(is_pause, time_wait)
    #         people_pose = openpose.getKeypoints(openpose.KeypointType.POSE)[0]
    #         # print(type(people_pose), people_pose.shape)
    #         if people_pose is None:
    #             continue
    #         print(image_file, 'number of people %d' % people_pose.shape[0])
    #         for i in range(people_pose.shape[0]):
    #             print('person %s' % i, people_pose[i,4,:], people_pose[i,7,:])
    #             cv2.circle(frame, tuple(map(int, people_pose[i,1,0:2])), 3, (255,255,255))
    #             cv2.circle(frame, tuple(map(int, people_pose[i,4,0:2])), 3, (0,0,255))
    #             cv2.circle(frame, tuple(map(int, people_pose[i,3,0:2])), 3, (0,0,120))
    #             cv2.circle(frame, tuple(map(int, people_pose[i,7,0:2])), 3, (0,255,0))
    #             cv2.circle(frame, tuple(map(int, people_pose[i,6,0:2])), 3, (0,120,0))
    #         cv2.circle(frame, (20, 150), 5, (255, 0, 0))
    #         print(people_pose[0,7,0], people_pose[0,7,0] < 25)
    #         if people_pose[0,7,1] > 0.5 and people_pose[0,7,0] <= 25 and people_pose[0,7,1] <= 160:
    #             time_wait = 0
    #
    #         cv2.imshow('', cv2.resize(frame, (image.shape[1], image.shape[0])))
    #         is_pause, time_wait = waitKey(is_pause, time_wait)


    # people_pose.shape = (N, K, 3), N = number of people, K = keypoints(18), 3 = (x, y, score)

    # POSE_COCO_BODY_PARTS
    # {
    #  {0, "Nose"},
    #  {1, "Neck"},
    #  {2, "RShoulder"},
    #  {3, "RElbow"},
    #  {4, "RWrist"},
    #  {5, "LShoulder"},
    #  {6, "LElbow"},
    #  {7, "LWrist"},
    #  {8, "RHip"},
    #  {9, "RKnee"},
    #  {10, "RAnkle"},
    #  {11, "LHip"},
    #  {12, "LKnee"},
    #  {13, "LAnkle"},
    #  {14, "REye"},
    #  {15, "LEye"},
    #  {16, "REar"},
    #  {17, "LEar"},
    #  {18, "Background"},
    # }

    # tray 5
    # tray 4 [(150, 320), (240, 430)]
    # tray 3 [(380, 360), (460, 450)]
    # tray 2 [(620, 355), (690, 445)]
    # tray 1 [(805, 380), (895, 445)]
