import os

import cv2
import numpy as np
from pa_utils.data.data_utils import resize_image, read_video
from pa_utils.openpose_utils import get_openpose_model

FPS = 5
tray_positions = [
    [(150, 320), (215, 435)],
    [(375, 360), (455, 450)],
    [(620, 355), (700, 445)],
    [(800, 380), (920, 440)],
]
person_positions = [
    (260, 485),
    (495, 525),
    (735, 510),
    (1020, 470),
]

# only consider right person's hand now
tray_hand_bottom_distances = [[] for _ in tray_positions]
tray_hand_right_distances = [[] for _ in tray_positions]
MAX_MA_RANGE = 180*FPS  # 1 minutes frame
MIN_MA_RANGE = 40*FPS  # need > 10 seconds before use to determine


def get_center(point1, point2):
    return ((point1[0] + point2[0])/2, (point1[1] + point2[1])/2)


def label_threshold(image):
    for tray_position in tray_positions:
        min_tray_x = tray_position[0][0]
        max_tray_x = tray_position[0][0] + int((tray_position[1][0] - tray_position[0][0])*.85)
        min_tray_y = tray_position[0][1]
        max_tray_y = tray_position[0][1] + int((tray_position[1][1] - tray_position[0][1])) #+ 20
        threshold_box = [(min_tray_x, min_tray_y), (min_tray_x, max_tray_y),
                         (max_tray_x, max_tray_y), (max_tray_x, min_tray_y)]
        image = cv2.drawContours(image, [np.array(threshold_box).reshape(-1, 1, 2)], 0, (0, 0, 255), 1)
    return image


def get_tray_operators(people_pose):
    operator_neck_positions = people_pose[:, 1,: 2].tolist()
    operator_left_shoulder_positions = people_pose[:, 5,: 2].tolist()
    # only consider right operator of each tray
    tray_operators = [[None, None] for _ in tray_positions]
    for keypoint_positions in [operator_neck_positions, operator_left_shoulder_positions]:
        operator_neck_map = dict((tuple(pos), idx) for idx, pos in enumerate(keypoint_positions))
        sorted_neck_positions = sorted(keypoint_positions, key=lambda x: x[0])
        last_operator_idx = -1
        for tray_idx, tray_position in enumerate(tray_positions):
            tray_center = get_center(*tray_position)
            next_tray_center = None if tray_idx >= len(tray_positions) - 1 else get_center(*tray_positions[tray_idx + 1])
            for operator_idx, operator_neck_position in enumerate(sorted_neck_positions):
                if operator_idx <= last_operator_idx:
                    continue
                if operator_neck_position[1] < tray_center[1]:
                    continue
                if operator_neck_position[0] > tray_center[0] and (next_tray_center is None or next_tray_center[0] > operator_neck_position[0]):
                    if tray_operators[tray_idx][1] is None:
                        tray_operators[tray_idx][1] = operator_neck_map[tuple(operator_neck_position)]
                    last_operator_idx = operator_idx
                    break
    return tray_operators


def check_if_operator_reach_tray(tray_position, operator_pose_position, left_hand=True, right_hand=False):
    if left_hand:
        left_hand_idx = 7
        left_hand_position = operator_pose_position[left_hand_idx, :2]
        min_tray_x = tray_position[0][0]
        max_tray_x = tray_position[0][0] + (tray_position[1][0] - tray_position[0][0])*.85
        min_tray_y = tray_position[0][1]
        max_tray_y = tray_position[0][1] + (tray_position[1][1] - tray_position[0][1]) #+ 20
        projected_hand_posision_x = hand_posision_x = left_hand_position[0]
        projected_hand_posision_y = hand_posision_y = left_hand_position[1]
        left_elbow_idx = 6
        if operator_pose_position[left_hand_idx, 2] > 0 and operator_pose_position[left_elbow_idx, 2] > 0:
            left_elbow_position = operator_pose_position[left_elbow_idx, :2]
            projected_hand_posision_x = hand_posision_x - 1/3*(left_elbow_position[0] - hand_posision_x)
            projected_hand_posision_y = hand_posision_y - 1/3*(left_elbow_position[1] - hand_posision_y)
            if min_tray_x < projected_hand_posision_x < max_tray_x and min_tray_y < projected_hand_posision_y < max_tray_y:
                return True, (projected_hand_posision_x, projected_hand_posision_y)
        if min_tray_x < hand_posision_x < max_tray_x and min_tray_y < hand_posision_y < max_tray_y:
            return True, (projected_hand_posision_x, projected_hand_posision_y)
        return False, (projected_hand_posision_x, projected_hand_posision_y)
    if right_hand:
        right_hand_idx = 4
        left_hand_position = operator_pose_position[right_hand_idx, :2]
        if tray_position[0][0] < left_hand_position[0] < tray_position[1][0] and left_hand_position[1] < tray_position[1][1]:
            return True, None
    return False, None


from keras.models import load_model
keras_model = None
tray_frame_datas = [[] for _ in range(len(tray_positions))]
def check_by_lstm(tray_idx, operator_pose_position):
    global keras_model
    if keras_model is None:
        keras_model = load_model('defond_keras_lstm.h5')

    from pa_utils.project.defond.prepare_training_data import preprocess
    tray_position = tray_positions[tray_idx]

    neck_position = operator_pose_position[1]
    left_hand_idx = 7
    left_hand_pose = operator_pose_position[left_hand_idx]
    right_hand_idx = 4
    right_hand_pose = operator_pose_position[right_hand_idx]

    current_frame_data = preprocess(tray_idx, neck_position, left_hand_pose, right_hand_pose)
    if current_frame_data is None:
        return
    tray_frame_datas[tray_idx].append(current_frame_data)
    if len(tray_frame_datas[tray_idx]) > 20:
        tray_frame_datas[tray_idx].pop(0)
    if len(tray_frame_datas[tray_idx]) == 20:
        input_data = np.array(tray_frame_datas[tray_idx]).reshape((1, -1)).clip(min=0)
        result = (keras_model.predict(input_data.reshape(-1, 20, 4))>0.5).astype(np.int).reshape(-1)
        result = result[0]
        # print(result, input_data)
        return result


def check_image_movement(openpose, frame, show_reach_tray=True, show_keypoints=True, draw_hand_position=True, wait_time=100):
    reach_tray_right_operators = []
    image = frame
    openpose.detectPose(image)
    people_pose = openpose.getKeypoints(openpose.KeypointType.POSE)[0]
    reach_trays = False
    if people_pose is not None:
        if show_keypoints:
            image = openpose.render(image)
        people_pose = people_pose[np.where(np.logical_and(people_pose[:, 7, 2] > 0, np.logical_or(people_pose[:, 1, 2] > 0, people_pose[:, 5, 2] > 0)))]
        trays_operators = get_tray_operators(people_pose)
        for tray_idx, (_, right_operator_idx) in enumerate(trays_operators):
            if right_operator_idx is None:
                continue
            is_reach, hand_position = check_if_operator_reach_tray(tray_positions[tray_idx], people_pose[right_operator_idx])
            if draw_hand_position and hand_position is not None:
                image = cv2.circle(image, tuple(map(int, hand_position)), 3, (255, 0, 0), thickness=5)
            if is_reach:
            # if check_by_lstm(tray_idx, people_pose[right_operator_idx]):
                reach_tray_right_operators.append((tray_idx, right_operator_idx))
                reach_trays = True
                if show_reach_tray:
                    draw_hand(image, people_pose[right_operator_idx, 7, :2])
        cv2.imshow('', image)
        if show_reach_tray and reach_trays:
            cv2.waitKey(0)
        else:
            cv2.waitKey(wait_time)
    return reach_tray_right_operators


def draw_hand(image, hand_position):
    cv2.circle(image, tuple(map(int, hand_position)), 3, (0, 0, 255))


def check_movement(image_dir=None, video_path=None, max_duration=10,
                   show_reach_tray=False, show_keypoints=False, video_writer=None):
    def _run(image, skip_tray_counter):
        tray_status = np.zeros((len(tray_positions),))
        image = resize_image(image, 600, 1024)
        reach_tray_right_operators = check_image_movement(openpose, image, wait_time=1, show_reach_tray=show_reach_tray,
                                                          show_keypoints=show_keypoints)
        tray_status[list(map(lambda x: x[0], reach_tray_right_operators))] = 2
        # print('\t'.join([image_file]+list(map(lambda x: str(int(x)), tray_status))))
        wait_time = 1
        trays_color = [(0, 0, 255) for _ in range(len(tray_positions))]
        if len(reach_tray_right_operators) > 0:
            for tray_idx, _ in reach_tray_right_operators:
                if skip_tray_counter[tray_idx] <= 0:
                    skip_tray_counter[tray_idx] = skip_count
                else:
                    continue
                tray_accumulate_counts[tray_idx] += 1
                # tray_position = tray_positions[tray_idx]
                # cv2.putText(image, str(tray_accumulate_counts[tray_idx]), tray_position[0], cv2.FONT_HERSHEY_SIMPLEX,
                #             2, (0, 255, 0), 2, cv2.LINE_AA)
                # wait_time = 1000
                trays_color[tray_idx] = (0, 255, 0)
                # if wait_time >= 1000:
                #     image = openpose.render(image)
        image = openpose.render(image)
        skip_tray_counter -= 1
        for tray_idx, count in enumerate(tray_accumulate_counts):
            cv2.putText(image, str(tray_accumulate_counts[tray_idx]), tray_positions[tray_idx][0],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2, trays_color[tray_idx], 2, cv2.LINE_AA)
        # image = label_threshold(image)
        # cv2.imwrite('images/%s' % image_file, image)
        if video_writer is not None:
            video_writer.write(image)
        cv2.imshow('', image)
        cv2.waitKey(wait_time)

    tray_accumulate_counts = [0 for _ in tray_positions]
    # skip 5 seconds before next count, assume 5 frames per seconds, so we will skill next 25 frames
    skip_count = 5*FPS
    skip_tray_counter = np.zeros((len(tray_positions), ))
    openpose = get_openpose_model(pose_network=(-1, 368))
    if image_dir is not None:
        for image_file in list(sorted(os.listdir(image_dir))):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(image_dir, image_file)
                image = cv2.imread(image_path)
                _run(image, skip_tray_counter)
    elif video_path is not None:
        fps = 25
        output_fps = 5
        frame_freq = int(fps / output_fps)
        for frame_idx, frame in enumerate(read_video(video_path, frame_freq=frame_freq)):
            if frame_idx > output_fps*max_duration:
                break
            _run(frame, skip_tray_counter)


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    openpose = get_openpose_model(pose_network=(-1, 368))
    image_path = '/app/powerarena-sense-gym/models/research/pa_utils/data/image_samples/defond/shenzhen-00007-20.jpg'

    image = cv2.imread(image_path)
    # check_image_movement(openpose, image, wait_time=10000)

    image_folder = '/app/powerarena-sense-gym/models/research/pa_utils/data/image_samples/defond'

    video_path = '/home/ma-glass/Downloads/20180413(2).mp4'
    video_output_path = 'defond(VA).mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, 5, (1024, 576))
    check_movement(video_path=video_path, video_writer=video_writer, max_duration=30*60)
    video_writer.release()

