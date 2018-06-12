from pa_utils.image_utils import add_waterprint


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.getLogger().setLevel(logging.INFO)

    video_path = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/Downloads/C0015(VA).mp4'
    output_path = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/Downloads/C0015(VA)_wp.mp4'
    add_waterprint(video_path, output_path)