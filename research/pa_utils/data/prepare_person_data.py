import os
import cv2
import numpy as np
import scipy.io
from PIL import Image


def get_shanghai_data(data_dir='/app/powerarena-sense-gym/people_dataset/ShanghaiTech'):
    image_dir = os.path.join(data_dir, 'part_B/test_data/images')
    mat_dir = os.path.join(data_dir, 'part_B/test_data/ground-truth')
    for mat_file in sorted(os.listdir(mat_dir)):
        mat_path = os.path.join(mat_dir, mat_file)
        mat = scipy.io.loadmat(mat_path)
        annotations = mat['image_info'][0][0][0][0]['location']
        image_path = os.path.join(image_dir, mat_file.replace('GT_', '').replace('.mat', '.jpg'))
        image = Image.open(image_path)
        image = np.array(image)
        # image = cv2.imread(image_path)
        yield image, annotations

if __name__ == '__main__':
    print(next(get_shanghai_data())[0][10:15,10,:])
