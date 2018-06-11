import os
from pa_utils.data.prepare_training_data import generate_from_video, add_lines


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    output_dir = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/Dataset/Project/FEHD/JPEGImages'
    video_file = 'NVR02 (Site 13 - 23)_S16-MK-WOS1_20180606181400_20180606204600.avi'
    video_path = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/Dataset/Video/FEHD/' + video_file
    image_prefix = 'FEHD' + '-' + os.path.splitext(os.path.basename(video_path))[0][21:32] + '-' + os.path.splitext(os.path.basename(video_path))[0][:20]
    output_fps = 1/10
    show_images = True
    known_fps = None
    skip_n_seconds = 0
    max_video_length = 0
    image_diff_threshold = None  #0.12
    image_diff_minimum_second = 10
    display_only = True
    # display_only = False
    prefix_exists_error = not display_only and False
    wait_time = 300
    start_time = 0 * 60 + 50
    end_time = 2 * 60 + 0
    if start_time > 0:
        skip_n_seconds = start_time
    if end_time > 0:
        max_video_length = end_time - start_time

    generate_from_video(video_path, output_dir, image_prefix=image_prefix,
                        output_minmax_size=(1080, 1920),
                        known_fps=known_fps,
                        max_output_fps=output_fps,
                        max_video_length=max_video_length,
                        show_images=show_images,
                        skip_n_seconds=skip_n_seconds,
                        image_diff_threshold=image_diff_threshold,
                        image_diff_minimum_second=image_diff_minimum_second,
                        display_only=display_only,
                        wait_time=wait_time,
                        prefix_exists_error=prefix_exists_error)


    # # add grid lines to images
    # output_dir = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/Dataset/Project/HKU/JPEGImages-lines'
    # add_lines(image_dir, output_dir)
