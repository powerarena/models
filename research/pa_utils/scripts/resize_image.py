import cv2
from pa_utils.data.data_utils import resize_image


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    image_path = '/home/ma-glass/Pictures/Screenshot from D1.mp4 - 2.png'
    output_image_path = '/home/ma-glass/Pictures/D1_resize-2.jpg'

    image = cv2.imread(image_path)
    resized_image = resize_image(image, 1080, 1920)
    cv2.imwrite(output_image_path, resized_image)
