import os
import numpy as np
import cv2
import datetime
import time
from pa_utils.data.data_utils import resize_image
import logging
logging.basicConfig(format='%(asctime)s %(message)s')
logging.getLogger().setLevel(logging.INFO)


def generate_from_video(video_path, output_dir, image_prefix='',
                        output_minmax_size=None,
                        output_fps=1.0,
                        show_images=True,
                        skip_n_seconds=0,
                        max_video_length=0,
                        ):
    video = cv2.VideoCapture(video_path)
    (major_ver, minor_ver, subminor_ver)= (cv2.__version__).split('.')
    if int(major_ver) < 3:
        length = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.cv.CV_CAP_PROP_FPS))
    else:
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        try:
            fps = int(video.get(cv2.CAP_PROP_FPS))
        except:
            logging.exception('fail to parse fps %s' % video.get(cv2.CAP_PROP_FPS))
            fps = 30
    print("length: {0}, width: {1}, height: {2}, fps: {3}".format(length, width, height, fps))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    frame_count = -1
    if skip_n_seconds:
        # video.set(cv2.CAP_PROP_POS_FRAMES, skip_n_seconds*fps)
        # frame_count += skip_n_seconds*fps
        max_video_length += skip_n_seconds
    while video.isOpened():
        frame_count += 1
        sec = frame_count // fps
        if sec > max_video_length > 0:
            break
        ret, frame = video.read()
        if not ret:
            break

        if frame_count < skip_n_seconds*fps:
            continue

        if frame_count % fps == 0:
            logging.info("video at %s sec" % (frame_count // fps))
        # write the frame to image
        if frame_count % (fps/output_fps) < 1:
            sec_frame = frame_count % fps
            image_path = os.path.join(output_dir, '%s-%05d-%02d.jpg' % (image_prefix, sec, sec_frame))
            if output_minmax_size is not None:
                cv2.imwrite(image_path, resize_image(frame, *output_minmax_size))
            else:
                cv2.imwrite(image_path, frame)
            logging.info('wrote frame at sec = %d' % sec)
        if show_images:
            if frame.shape[0] > 960:
                cv2.imshow('frame', resize_image(frame, *(540, 960)))
            else:
                cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # if frame_count >= fps*60*15:
        #     break
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = '/home/ma-glass/Downloads/20180413(2).mp4'
    output_dir = 'image_samples/defond'
    image_prefix = 'defond'
    output_fps = 5
    show_images = True
    max_video_length = 300

    generate_from_video(video_path, output_dir, image_prefix=image_prefix,
                        output_fps=output_fps,
                        output_minmax_size=(600, 1024),
                        show_images=show_images,
                        skip_n_seconds=0,
                        max_video_length=max_video_length)