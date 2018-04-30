import csv
import os

import cv2
import numpy as np
from pa_utils.data.data_utils import resize_image
from pa_utils.openpose_utils import get_openpose_model
from pa_utils.project.defond.count_movement import get_tray_operators, tray_positions, get_center, person_positions

feature_number = 4
def preprocess(tray_idx, neck_pose, left_hand_pose, right_hand_pose):
    tray_position = tray_positions[tray_idx]
    person_position = person_positions[tray_idx]
    tray_x, tray_y = get_center(*tray_position)
    tray_width = tray_position[1][0] - tray_position[0][0]
    tray_height = tray_position[1][1] - tray_position[0][1]
    distance_x = abs(person_position[0] - tray_x)
    distance_y = abs(person_position[1] - tray_y)
    # if neck_pose[2] < .1:
    #     return
    # person_tray_distance = np.linalg.norm((tray_x - neck_pose[0], tray_y - neck_pose[1]))

    data = [(left_hand_pose[0] - tray_x)/distance_x, (left_hand_pose[1] - tray_y)/distance_y,
            (right_hand_pose[0] - tray_x)/distance_x, (right_hand_pose[1] - tray_y)/distance_y]
    return [1.0*d for d in data]


def get_training_data(window_size=20):
    images_ids = []
    persons_status = [[] for _ in range(5)]
    number_labeled_frames = 0
    with open('training_data.txt') as fr:
        reader = csv.DictReader(fr, fieldnames=['ImageID'] + ['Person %d' % i for i in range(5, 0, -1)], delimiter='\t')
        columns = next(reader)
        print(columns)
        for row in reader:
            if row['ImageID'] is None or row['ImageID'].strip() == '':
                break
            images_ids.append(row['ImageID'])
            number_labeled_frames += 1
            for person_idx in range(1, 6):
                if row['Person %d' % person_idx] is not None and '2' in row['Person %d' % person_idx]:
                    persons_status[person_idx - 1].append([row['ImageID'], 1])
                else:
                    persons_status[person_idx - 1].append([row['ImageID'], 0])

    with open('image_openpose_result.txt') as fr:
        for idx, line in enumerate(fr):
            image_idx = int(idx/4)
            components = line.strip().split()
            tray_idx = int(components[1])
            person_idx = 4 - tray_idx
            if image_idx >= number_labeled_frames:
                print('image_idx >= number_labeled_frames')
                break
            person_status_record = persons_status[person_idx - 1][image_idx]
            if person_status_record[0] != components[0]:
                raise Exception('image_id dont match!')
            if len(components) == 2:
                person_status_record.append(None)
                continue
            neck_pose = list(map(float, components[2:5]))
            left_hand_pose = list(map(float, components[5:8]))
            right_hand_pose = list(map(float, components[8:11]))
            data = preprocess(tray_idx, neck_pose, left_hand_pose, right_hand_pose)
            if data is None:
                person_status_record.append(None)
            else:
                person_status_record.extend(data)
            # print(person_idx, person_status_record)
    x = []
    y = []
    for frame_idx in range(0, len(persons_status[0]) - window_size):
        for person_idx in range(0, 4):
            x_i = []
            y_i = False
            frame_size = 0
            idx = -1
            while frame_size < window_size:
                idx += 1
                if frame_idx + idx >= len(persons_status[person_idx]):
                    break
                if persons_status[person_idx][frame_idx + idx][2] is None:
                    continue
                frame_size += 1
                x_i.extend(persons_status[person_idx][frame_idx + idx][2:])
                if persons_status[person_idx][frame_idx + idx][1] == 1:
                    y_i = True
            if len(x_i) == window_size * feature_number:
                x.append(x_i)
                y.append(y_i)

    return np.array(x).clip(min=0), np.array(y)


def generate_openpose_positions(image_dir, output_path):

    with open(output_path, 'w') as fw:
        openpose = get_openpose_model(pose_network=(-1, 368))
        for image_file in sorted(os.listdir(image_dir)):
            if not image_file.endswith('.jpg'):
                continue
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)
            image = resize_image(image, 600, 1024)

            openpose.detectPose(image)
            image = openpose.render(image)
            cv2.imshow('', image)
            cv2.waitKey(1)
            people_pose = openpose.getKeypoints(openpose.KeypointType.POSE)[0]
            if people_pose is not None:
                people_pose = people_pose[np.where(np.logical_or(people_pose[:, 1, 2] > 0, people_pose[:, 5, 2] > 0))]
                trays_operators = get_tray_operators(people_pose)
                for tray_idx, (_, right_operator_idx) in enumerate(trays_operators):
                    if right_operator_idx is None:
                        fw.write('%s %s\n' % (image_file, tray_idx))
                        continue
                    neck_position = people_pose[right_operator_idx, 1]
                    left_hand_position = people_pose[right_operator_idx, 7]
                    right_hand_position = people_pose[right_operator_idx, 4]
                    print(neck_position, left_hand_position, right_hand_position)
                    fw.write('%s %s %s %s %s %s %s %s %s %s %s\n' % (image_file, tray_idx,
                                                                  neck_position[0], neck_position[1], neck_position[2],
                                                       left_hand_position[0], left_hand_position[1], left_hand_position[2],
                                                       right_hand_position[0], right_hand_position[1], right_hand_position[2]))


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    image_dir = '/app/powerarena-sense-gym/models/research/pa_utils/data/image_samples/defond'
    output_path = 'image_openpose_result.txt'
    # generate_openpose_positions(image_dir, output_path)

    x, y = get_training_data()
    print(x.shape, y.shape)
    print(x[0])

    from keras.models import Sequential, load_model
    from keras.layers import LSTM, Dense, Dropout

    print(x.shape)
    model = Sequential()
    model.add(LSTM(100, input_shape=(20, feature_number,), return_sequences=True))
    model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
    model.add(LSTM(32))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(x.reshape(-1, 20, feature_number), y, epochs=50, batch_size=10, verbose=1)
    model.save('defond_keras_lstm.h5')

    model = load_model('defond_keras_lstm.h5')
    print(((model.predict(x.reshape(-1, 20, feature_number))>0.5).astype(np.int).reshape(-1) == y).sum(), y.size)