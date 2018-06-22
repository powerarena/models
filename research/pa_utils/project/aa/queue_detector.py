import os
import time
import cv2
import numpy as np
import collections
import threading
import queue
from functools import partial
from pa_utils.data.data_utils import remove_detection, get_image_reader
from pa_utils.data.prepare_training_data import load_detections_pickle
from pa_utils.image_utils import label_image
# drawing
from pa_utils.project.aa.queue_utils import draw_counter_regions, draw_counter_tables, draw_grid_lines, \
    draw_queue_stats, draw_roi_masks, get_and_draw_baisc_queue_lines, get_and_draw_basic_queue_regions, \
    get_and_draw_dynamic_queue_lines, get_and_draw_queues, get_and_draw_service_regions, process_and_draw_track_lines, \
    get_and_draw_queue_regions, draw_queue_time_result
# utils
from pa_utils.project.aa.queue_utils import bbox_center, get_nearest_grid_id, maximize_counts, process_raw_demo_queues_count, \
    filter_detections_inside_regions, detect_image_by_worker, get_desk_positions, get_output_processor
# parameters
from pa_utils.project.aa.queue_utils import PROCESS_FPS, MAX_TRACK_SECONDS, IMAGE_HEIGHT, IMAGE_WIDTH, \
    MAX_QUEUE_NUMBER, MIN_SCORE_THRESHOLD, MAX_QUEUE_GAP, COUNTER_TRACK_SECONDS, QUEUE_COUNT_MAXIMIZE_WINDOW
# counters
from pa_utils.project.aa.queue_utils import detect_counters, get_counter_staff_indices, get_service_region_parameter, \
    estimate_queue_time_based_on_queues_count, create_demo_queues_time


DEMO = False
BOOTSTRAP_SEC = 30
DEFAULT_SKIP_SECONDS_FOR_AVAILABLE_COUNTERS = 18


def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def track_iou(tracks_active, detections, track_id_start,
              sigma_l=0, sigma_h=0.5, sigma_iou=0.5, t_min=2,
              deque_size=PROCESS_FPS*MAX_TRACK_SECONDS):
    """
    Simple IOU based tracker.
    See "High-Speed Tracking-by-Detection Without Using Image Information by E. Bochinski, V. Eiselein, T. Sikora" for
    more information.
    Args:
         detections (list): list of detections per frame, usually generated by util.load_mot
         sigma_l (float): low detection threshold.
         sigma_h (float): high detection threshold.
         sigma_iou (float): IOU threshold.
         t_min (float): minimum track length in frames.
    Returns:
        list: list of tracks.
    """

    # tracks_active = []
    tracks_finished = []

    # apply low threshold to detections
    dets = detections.tolist()

    updated_tracks = []
    for track in tracks_active:
        if len(dets) > 0:
            # get det with highest iou
            best_match = max(dets, key=lambda x: iou(track['bboxes'][-1], x))
            if iou(track['bboxes'][-1], best_match) >= sigma_iou:
                track['bboxes'].append(best_match)
                track['center'].append(bbox_center(best_match))

                updated_tracks.append(track)

                # remove from best matching detection from detections
                del dets[dets.index(best_match)]

        # if track was not updated
        if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
            # finish track when the conditions are met
            if len(track['bboxes']) >= t_min:
                tracks_finished.append(track)

    # create new tracks
    new_tracks = [{'track_id': track_id_start + idx,
                   'bboxes': collections.deque([det], maxlen=deque_size),
                   'center': collections.deque([bbox_center(det)], maxlen=deque_size)
                   } for idx, det in enumerate(dets)]
    new_tracks_active = updated_tracks + new_tracks

    return new_tracks_active, tracks_finished


def track_grids(grid_detection_history, tracks_active, frame_idx):
    grid_detection = collections.defaultdict(int)
    for track in tracks_active:
        grid_detection[get_nearest_grid_id(*track['center'][-1])] += 1
    for grid_id, count in grid_detection.items():
        grid_detection_history[grid_id].append((frame_idx, count))


def track_detections(detections, image_reader_bootstrap, image_reader, detection_roi_masks, desk_positions, table_face_slope,
                     queue_time_output_path,
                     avilable_counters_skip_seconds=DEFAULT_SKIP_SECONDS_FOR_AVAILABLE_COUNTERS,
                     wait_time=3, max_queue_gap=IMAGE_WIDTH/4,
                     known_available_counters=None,
                     output_processor=None):
    print('DEMO', DEMO)
    draw_staff = True
    draw_queue_region = True
    draw_queue_people = True
    draw_roi = False
    draw_grid = False
    draw_flow = False
    draw_counter = False # queue region
    draw_counter_region = False
    draw_service_region = False
    draw_queue_line = False
    draw_non_queue_people = False
    # basic
    draw_basic_queue_line = False
    draw_basic_queue_region = False

    counter_show_up_threshold = COUNTER_TRACK_SECONDS*PROCESS_FPS
    all_counters = dict((desk_idx, collections.deque(maxlen=counter_show_up_threshold)) for desk_idx in range(len(desk_positions)))
    all_tracks = dict()
    grid_detection_history = collections.defaultdict(partial(collections.deque, maxlen=MAX_TRACK_SECONDS*PROCESS_FPS))
    tracks_active = []
    track_id_start = 1
    service_region_parameters = get_service_region_parameter(desk_positions)
    queues_count_history = collections.defaultdict(partial(collections.deque, maxlen=QUEUE_COUNT_MAXIMIZE_WINDOW))
    maximized_queues_counts = collections.defaultdict(list)

    def _process_people_flow(image_idx, output_dict, all_counters, tracks_active, track_id_start, grid_detection_history, draw_staff=False):
        # remove counter staff detection
        detection_index_to_counter_map = get_counter_staff_indices(output_dict['detection_boxes'], desk_positions)
        if draw_staff:
            label_image(image, dict(detection_boxes=output_dict['detection_boxes'][list(detection_index_to_counter_map.keys())]),
                        min_score_thresh=MIN_SCORE_THRESHOLD, use_normalized_coordinates=False,
                        line_thickness=2)
        remove_detection(output_dict, filter_indice=list(detection_index_to_counter_map.keys()))

        for counter_idx in set(detection_index_to_counter_map.values()):
            all_counters[counter_idx].append(image_idx)

        tracks_active, _ = track_iou(tracks_active, output_dict['detection_boxes'], track_id_start, sigma_iou=.3)
        track_grids(grid_detection_history, tracks_active, image_idx)
        return tracks_active

    boostrap_detections = dict()
    if detections is None:
        input_q = queue.Queue()
        output_q = queue.Queue()
        for gpu_idx in range(2):
            worker = threading.Thread(target=detect_image_by_worker, args=(input_q, output_q, gpu_idx, boostrap_detections))
            worker.daemon = True
            worker.start()

    # bootstrap 30s
    ### single thread ###
    # for image_idx, image in enumerate(image_reader_bootstrap):
    #     if detections is None:
    #         output_dict = run_inference_for_single_image(sess, image)
    #         boostrap_detections[image_idx] = output_dict
    #     else:
    #         output_dict = detections[image_idx]
    ### single thread ###

    finish_loop_input = False
    enum_bootstrap_reader = enumerate(image_reader_bootstrap)
    while True:
        if detections is None:
            ### worker ###
            if finish_loop_input and input_q.qsize() == 0 and output_q.qsize() == 0:
                # skip last 2 frame result
                break

            if not finish_loop_input and input_q.qsize() < PROCESS_FPS:
                try:
                    input_image_idx, input_image = next(enum_bootstrap_reader)
                    input_q.put((input_image_idx, input_image))
                except StopIteration:
                    finish_loop_input = True
                continue

            image_idx, image, output_dict = output_q.get()
            ### worker ###
        else:
            try:
                image_idx, image = next(enum_bootstrap_reader)
            except StopIteration:
                break
            output_dict = detections[image_idx]

        boostrap_detections[image_idx] = output_dict

        output_dict = output_dict.copy()

        remove_detection(output_dict, min_score_thresh=MIN_SCORE_THRESHOLD)
        output_dict['detection_boxes'] = np.rint(output_dict['detection_boxes'] * (image.shape[0], image.shape[1], image.shape[0], image.shape[1])).astype(np.int32)
        output_dict['detection_boxes'][:,2] = np.minimum(output_dict['detection_boxes'][:,2], image.shape[0] - 1)
        output_dict['detection_boxes'][:,3] = np.minimum(output_dict['detection_boxes'][:,3], image.shape[1] - 1)

        if detection_roi_masks and len(detection_roi_masks) > 0:
            roi_detection_indice = filter_detections_inside_regions(output_dict['detection_boxes'], detection_roi_masks)
            remove_detection(output_dict, retain_indice=roi_detection_indice)

        image_idx -= BOOTSTRAP_SEC*PROCESS_FPS
        tracks_active = _process_people_flow(image_idx, output_dict, all_counters, tracks_active, track_id_start, grid_detection_history)
        print('bootstrapping', image_idx)

    start = time.time()

    # for image_idx, image in enumerate(image_reader):
    #     if detections is None:
    #         if image_idx not in boostrap_detections:
    #             output_dict = run_inference_for_single_image(sess, image)
    #         else:
    #             output_dict = boostrap_detections[image_idx]
    #     else:
    #         output_dict = detections[image_idx]

    finish_loop_input = False
    enum_reader = enumerate(image_reader)
    current_sec = 0
    image = None
    while True:
        if detections is None:
            ### worker ###
            if finish_loop_input and input_q.qsize() == 0 and output_q.qsize() == 0:
                # skip last 2 frame result
                break

            if not finish_loop_input and input_q.qsize() < PROCESS_FPS:
                try:
                    input_image_idx, input_image = next(enum_reader)
                    input_q.put((input_image_idx, input_image))
                except StopIteration:
                    finish_loop_input = True
                continue

            image_idx, image, output_dict = output_q.get()
            ### worker ###
        else:
            try:
                image_idx, image = next(enum_reader)
            except StopIteration:
                break
            output_dict = detections[image_idx]

        current_sec = image_idx // PROCESS_FPS

        remove_detection(output_dict, min_score_thresh=MIN_SCORE_THRESHOLD)
        output_dict['detection_boxes'] = np.rint(output_dict['detection_boxes'] * (image.shape[0], image.shape[1], image.shape[0], image.shape[1])).astype(np.int32)
        output_dict['detection_boxes'][:,2] = np.minimum(output_dict['detection_boxes'][:,2], image.shape[0] - 1)
        output_dict['detection_boxes'][:,3] = np.minimum(output_dict['detection_boxes'][:,3], image.shape[1] - 1)

        if detection_roi_masks and len(detection_roi_masks) > 0:
            roi_detection_indice = filter_detections_inside_regions(output_dict['detection_boxes'], detection_roi_masks)
            remove_detection(output_dict, retain_indice=roi_detection_indice)

        tracks_active = _process_people_flow(image_idx, output_dict, all_counters, tracks_active, track_id_start, grid_detection_history, draw_staff=draw_staff)

        if known_available_counters is None or (image_idx // PROCESS_FPS) <= avilable_counters_skip_seconds:
            available_counters = detect_counters(all_counters, image_idx)
        else:
            available_counters = known_available_counters

        for track in tracks_active:
            track['last_frame_idx'] = image_idx
            all_tracks[track['track_id']] = track
        if len(tracks_active) > 0:
            track_id_start = max(track_id_start, 1 + tracks_active[-1]['track_id'])

        if image_idx % PROCESS_FPS == 0:
            draw_roi_masks(image, detection_roi_masks, draw_roi)
            draw_grid_lines(image, draw_grid)

            # # draw counter tables
            draw_counter_tables(image, desk_positions, draw_counter=draw_counter)
            draw_counter_regions(image, desk_positions, draw_counter_region=draw_counter_region)

            basic_queue_lines = get_and_draw_baisc_queue_lines(available_counters, desk_positions, image, draw_queue_line=draw_basic_queue_line)
            service_regions, service_regions_grids = get_and_draw_service_regions(basic_queue_lines, image, service_region_parameters, draw_service_region=draw_service_region)
            basic_queue_regions, queues_upper_limit_line = get_and_draw_basic_queue_regions(basic_queue_lines, desk_positions, image=image, draw_queue_region=draw_basic_queue_region)
            basic_queue_detections = get_and_draw_queues(basic_queue_regions, output_dict.copy(), image, image_idx,
                                                   grid_detection_history, service_regions, max_gap=max_queue_gap/3*4,
                                                   draw_queue_people=False,
                                                   draw_non_queue_people=False)
            grids_weight = process_and_draw_track_lines(image, grid_detection_history, image_idx, draw_flow=draw_flow)
            dynamic_queue_lines = get_and_draw_dynamic_queue_lines(grids_weight, service_regions_grids,
                                                                   basic_queue_lines, basic_queue_detections,
                                                                   table_face_slope, service_region_parameters,
                                                                   image=image, draw_queue_line=draw_queue_line)
            dynamic_queue_regions = get_and_draw_queue_regions(dynamic_queue_lines, queues_upper_limit_line, desk_positions, IMAGE_WIDTH, IMAGE_HEIGHT, image=image,
                                              draw_queue_region=draw_queue_region)
            queue_detections = get_and_draw_queues(dynamic_queue_regions, output_dict, image, image_idx,
                                                   grid_detection_history, service_regions, max_gap=max_queue_gap,
                                                   draw_queue_people=draw_queue_people,
                                                   draw_non_queue_people=draw_non_queue_people)

            for queue_idx in range(MAX_QUEUE_NUMBER):
                queues_count_history[queue_idx].append(len(queue_detections.get(queue_idx, [])))
            queues_count = maximize_counts(queues_count_history)
            if DEMO:
                queues_count = process_raw_demo_queues_count(current_sec, queues_count)

            for queue_idx in queues_count:
                maximized_queues_counts[queue_idx].append(queues_count[queue_idx])

            # visual stat
            draw_queue_stats(current_sec, image, queues_count, desk_positions)

            cv2.imshow('frame', image)
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break

            if output_processor:
                output_processor(image_idx, queues_count, queue_detections, image=image)

        print(image_idx, time.time() - start)
        start = time.time()

    if DEMO:
        estimated_queues_time = create_demo_queues_time(queue_time_output_path, maximized_queues_counts, current_sec)
    else:
        estimated_queues_time = estimate_queue_time_based_on_queues_count(queue_time_output_path, maximized_queues_counts)

    if image is not None and len(estimated_queues_time) > 0:
        draw_queue_time_result(image, estimated_queues_time, desk_positions)
        sleep_sec_count = 0
        while sleep_sec_count <= 5:
            output_processor(None, None, None, image=image)
            cv2.imshow('', image)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                break
            sleep_sec_count += 1

    cv2.destroyAllWindows()


def get_configs(video_path):
    avilable_counters_skip_seconds = DEFAULT_SKIP_SECONDS_FOR_AVAILABLE_COUNTERS

    # DEMO
    if video_path.lower().endswith('D1.mp4'.lower()):
        # D1
        table_face_direction = [(460, 703), (1404, 816)]
        table_service_direction = [(213, 530), (1484, 144)]
        table_horizontal_direction = [(207, 721), (1652, 207)]
        desk_positions = [
             [173, 213, 323, ],
             [409, 449, 540, ],
             [549, 588, 668, ],
             [744, 776, 846, ],
             [847, 887, 953, ],
             [1000, 1037, 1098, ],
             [1099, 1130, 1188, ],
             [1229, 1258, 1310, ],
             [1308, 1335, 1384, ],
             [1410, 1442, 1486, ],
        ]
        initial_known_acs = [0, 1, 2]
    elif video_path.lower().endswith('TODO.mp4'.lower()):
        # Step 1. edit above line, e.g. TODO.mp4 => D2.mp4

        # Step 2. Assume the image is 1920x1080, please resize the image before find coordinates
        # use pa_utils/scripts/resize_image.py to resize the captured screen

        # Step 3. find 2 points of the "vertical line",  i.e. the line perpendicular to counters
        table_face_direction = [(327, 684), (1412, 814)]

        # Step 4. find 2 points of "upper horizontal line", i.e. the line connect the counters
        table_service_direction = [(218, 523), (1493, 137)]

        # Step 5. find 2 points of the "lower horizontal line", i.e. the line where counters touch the ground
        table_horizontal_direction = [(219, 722), (1485, 266)]

        # Step 6. find x coordinate for each counter, each counter should have 3 x coordinates
        # the y coordinate will be calculated based on the x coordinate and the "upper horizontal line"
        desk_positions = [
            [177, 217, 319],
            [403, 447, 542],
            [549, 586, 668],
            [730, 773, 845],
            [851, 885, 954],
            [999, 1037, 1097],
            [1099, 1129, 1187],
            [1222, 1256, 1308],
            [1306, 1335, 1384],
            [1409, 1441, 1485],
        ]

        # Step 7. you could provide available cs, index start from 0.
        # if cannot determined, then set it to be None
        initial_known_acs = [0, 1, 2]
    else:
        print('No such configs')
        return

    return table_face_direction, table_service_direction, table_horizontal_direction, desk_positions, \
           initial_known_acs, avilable_counters_skip_seconds


if __name__ == '__main__':
    # queue detection flow:
    # 1. detect counters by
    #   a. counter staff, region after counter table
    #   b. serving people, region before counter table ( not implemented )
    # Assume we can transform the video into designed angle.
    # 1. use basic queue region to check the queue length
    # 2a. if two nearby queues have comparable queue length, then we will use the weighted flow to determine the queue lines.
    # 2b. otherwise, if one queue is too short, compared with another, we use default queue region instead.

    # BOOTSTRAP_SEC = 1

    # step 1. video path
    video_path = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/Downloads/D1_TODO.mp4'

    if video_path.endswith('D1.mp4'):
        DEMO = True

    # step 2. (Optional) crop region, default = None
    # if image size is too large & roi less than 50%, crop instead of mask, y1, x1, y2, x2
    CROP_REGION = None

    image_reader_bootstrap = get_image_reader(video_path=video_path, video_fps=PROCESS_FPS, max_video_length=BOOTSTRAP_SEC,
                                              crop_region=CROP_REGION, output_minmax_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    image_reader = get_image_reader(video_path=video_path, video_fps=PROCESS_FPS, max_video_length=0,
                                    crop_region=CROP_REGION, output_minmax_size=(IMAGE_HEIGHT, IMAGE_WIDTH))

    # step 3. (Optional) preload detections, default = None
    # pickle_path = '/app/powerarena-sense-gym/models/research/pa_utils/project/aa/d1_detections_fps5_bgr.pkl'
    # frames_detections = load_detections_pickle(pickle_path)
    frames_detections = None

    # step 4. (Optional) roi, default empty list, default = []
    # Use roi masks if not whole area could contains queue people.
    # detection_roi_masks = [[(0, 0), (1250, 0), (1250, 800), (0, 800)]]
    detection_roi_masks = []

    # step 5. calibrate coordinates, in the sense of (1920 x 1080 pixels)
    # Please IMPLEMENT get_configs
    table_face_direction, table_service_direction, table_horizontal_direction, desk_positions, known_acs, avilable_counters_skip_seconds = get_configs(video_path)
    table_face_slope, desk_positions = get_desk_positions(table_face_direction, table_service_direction, table_horizontal_direction, desk_positions)

    output_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = '%s_queue_result' % output_name
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_video_path = os.path.join(output_folder, '%s(VA).mp4' % output_name)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 1, (IMAGE_WIDTH, IMAGE_HEIGHT))
    csv_path = os.path.join(output_folder, '%s_queue_info.csv' % output_name)
    queue_time_output_path = os.path.join(output_folder, '%s_queue_time.txt' % output_name)
    output_processor = get_output_processor(csv_path=csv_path, video_writer=video_writer)
    # output_processor = None

    track_detections(frames_detections, image_reader_bootstrap, image_reader,
                     detection_roi_masks, desk_positions, table_face_slope, queue_time_output_path,
                     avilable_counters_skip_seconds=avilable_counters_skip_seconds,
                     known_available_counters=known_acs, max_queue_gap=MAX_QUEUE_GAP,
                     wait_time=5, output_processor=output_processor)
    video_writer.release()

