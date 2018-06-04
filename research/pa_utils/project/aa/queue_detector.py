import os
import time
import cv2
import numpy as np
import math
import collections
import threading
import queue
from functools import partial
from matplotlib import pyplot as plt
from pa_utils.graph_utils import get_session
from pa_utils.data.data_utils import get_video_info, read_video, create_label_categories, remove_detection, get_image_reader, get_color_list
from pa_utils.model_utils import run_inference_for_single_image
from pa_utils.data.prepare_training_data import load_detections_pickle
from pa_utils.image_utils import label_image
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon


base_folder = os.path.join(os.path.dirname(__file__), '../../')
label_path = os.path.join(base_folder, 'data/label_maps/person_label_map.pbtxt')
category_index = create_label_categories(label_path, 100)

BOOTSTRAP_SEC = 30
GRID_SIZE = 20
PROCESS_FPS = 5
MAX_TRACK_SECONDS = 120
IMAGE_WIDTH, IMAGE_HEIGHT = (1920, 1080)

SERVICE_REGION_PIXEL = 150
SERVICE_GRID_NUM = max(3, math.ceil(SERVICE_REGION_PIXEL/GRID_SIZE))
def prepare_service_grips():
    # rows = np.arange(5)
    # cols = np.arange(5)
    # x, y = np.meshgrid(rows, cols)
    # service_grids = np.stack((x, y), axis=-1)
    num_grid = SERVICE_GRID_NUM
    center_grid = round(num_grid*3/5)
    service_grids = np.array([(j, i) for i in range(0, num_grid) for j in range(0, num_grid)]).reshape(num_grid*num_grid,2) - (center_grid,center_grid)
    # column_grids = service_grids.reshape(num_grid, num_grid, 2)[:,0,1]
    # column_weight = pow((num_grid - abs(column_grids))/num_grid, 2)
    column_grids = np.arange(int(num_grid/2)*2+1) - int(num_grid/2)
    column_weight = pow((column_grids.shape[0] - abs(column_grids)) / column_grids.shape[0], 2)
    return service_grids, column_grids, column_weight

SERVICE_REGION_GRIDS, SERVICE_REGION_COLUMNS, SERVICE_REGION_COLUMNS_WEIGHT = prepare_service_grips()
print(SERVICE_REGION_COLUMNS, SERVICE_REGION_COLUMNS_WEIGHT)

COLOR_LIST = get_color_list(return_rgb=True)
COLOR_NAME_LIST = get_color_list()

print('PROCESS_FPS:', PROCESS_FPS)


def transformation_trial():
    image_path = ''
    img = cv2.imread('/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/Dataset/Project/AA/D1/JPEGImages/aa-D1-00430-00.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rows,cols,ch = img.shape

    pts1 = np.float32([[207,717],[1333,316],[1163,838],[1918,357]])
    pts2 = np.float32([[0,0],[1000,0],[0,1000],[1000,1000]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(1000,1000))


    # pts1 = np.float32([[207,717],[1333,316],[1163,838]])
    # pts2 = np.float32([[0,0],[1000,0],[0,1000]])
    # M = cv2.getAffineTransform(pts1,pts2)
    # dst = cv2.warpAffine(img,M,(cols,rows))

    point1 = pts1[0]
    point2 = pts1[2]
    # anti clockwise
    counter_line_angle_from_horizontal = math.atan((point2[1] - point1[1])/(point2[0] - point1[0]))/math.pi*180
    rotation_anlge_horizontal = counter_line_angle_from_horizontal
    rotation_anlge_vertical = counter_line_angle_from_horizontal - 90
    M = cv2.getRotationMatrix2D((cols/2,rows/2),rotation_anlge_horizontal,1)
    dst = cv2.warpAffine(img,M,(cols,rows))

    plt.subplot(121),plt.imshow(img),plt.title('Input')
    plt.subplot(122),plt.imshow(dst),plt.title('Output')
    plt.show()
    dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
    cv2.imwrite('vertical_aa.jpg', dst)


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


def center(bbox, y1x1=True, head_center=False):
    if y1x1:
        y1, x1, y2, x2 = bbox
    else:
        x1, y1, x2, y2 = bbox

    if head_center:
        return (int((x1+x2)/2), int(y1))
    return (int((x1+x2)/2), int((y1+y2)/2))


def get_line_y_by_2point(point1, point2, x):
    x1, y1 = point1
    x2, y2 = point2
    if x1 == x2:
        print('vertical line', point1, point2)
        return 0
    slope = (y2 - y1)/(x2 - x1)
    # slope = (y-y1)/(x-x1)
    y = slope*(x-x1) + y1
    return y


def get_line_y_by_slope(slope, point1, x):
    x1, y1 = point1
    if slope == 0:
        return y1
    y = slope*(x-x1) + y1
    return y


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
                track['center'].append(center(best_match))

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
                   'center': collections.deque([center(det)], maxlen=deque_size)
                   } for idx, det in enumerate(dets)]
    new_tracks_active = updated_tracks + new_tracks

    return new_tracks_active, tracks_finished


def track_grids(grid_detection_history, tracks_active, frame_idx):
    grid_detection = collections.defaultdict(int)
    for track in tracks_active:
        grid_detection[get_nearest_grid_id(*track['center'][-1])] += 1
    for grid_id, count in grid_detection.items():
        grid_detection_history[grid_id].append((frame_idx, count))


def track_detections(detections, image_reader_bootstrap, image_reader, desk_positions, table_face_slope,
                     sess=None, min_score_thresh=.5, wait_time=1,
                     output_processor=None):
    counter_show_up_threshold = 60*PROCESS_FPS
    all_counters = dict((desk_idx, collections.deque(maxlen=counter_show_up_threshold)) for desk_idx in range(len(desk_positions)))
    all_tracks = dict()
    grid_detection_history = collections.defaultdict(partial(collections.deque, maxlen=MAX_TRACK_SECONDS*PROCESS_FPS))
    tracks_active = []
    track_id_start = 1
    available_counters = []

    def _process_people_flow(image_idx, image, output_dict, all_counters, tracks_active, track_id_start, grid_detection_history):
        # remove counter staff detection
        detection_index_to_counter_map = get_counter_staff_indices(output_dict['detection_boxes'], desk_positions, image.shape)
        remove_detection(output_dict, filter_indice=list(detection_index_to_counter_map.keys()))
        for counter_idx in set(detection_index_to_counter_map.values()):
            all_counters[counter_idx].append(image_idx)

        tracks_active, _ = track_iou(tracks_active, output_dict['detection_boxes'], track_id_start)
        track_grids(grid_detection_history, tracks_active, image_idx)
        return tracks_active

    boostrap_detections = dict()
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

    ### worker ###
    finish_loop_input = False
    enum_bootstrap_reader = enumerate(image_reader_bootstrap)
    while True:
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
        boostrap_detections[image_idx] = output_dict
    ### worker ###

        output_dict = output_dict.copy()

        remove_detection(output_dict, min_score_thresh=min_score_thresh)
        output_dict['detection_boxes'] = np.rint(output_dict['detection_boxes'] * (image.shape[0], image.shape[1], image.shape[0], image.shape[1])).astype(np.int32)
        output_dict['detection_boxes'][:,2] = np.minimum(output_dict['detection_boxes'][:,2], image.shape[0] - 1)
        output_dict['detection_boxes'][:,3] = np.minimum(output_dict['detection_boxes'][:,3], image.shape[1] - 1)

        image_idx -= BOOTSTRAP_SEC*PROCESS_FPS
        tracks_active = _process_people_flow(image_idx, image, output_dict, all_counters, tracks_active, track_id_start, grid_detection_history)
        print('bootstrapping', image_idx)

    draw_grid = False
    draw_flow = False
    draw_counter = False
    draw_service_region = False
    draw_queue_line = False
    draw_queue_region = True
    draw_queue_people = True
    draw_non_queue_people = False
    # basic
    draw_basic_queue_line = False
    draw_basic_queue_region = True
    start = time.time()

    # for image_idx, image in enumerate(image_reader):
    #     if detections is None:
    #         if image_idx not in boostrap_detections:
    #             output_dict = run_inference_for_single_image(sess, image)
    #         else:
    #             output_dict = boostrap_detections[image_idx]
    #     else:
    #         output_dict = detections[image_idx]

    ### worker ###
    finish_loop_input = False
    enum_reader = enumerate(image_reader)
    while True:
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

        draw = image_idx % PROCESS_FPS == 0

        remove_detection(output_dict, min_score_thresh=min_score_thresh)
        output_dict['detection_boxes'] = np.rint(output_dict['detection_boxes'] * (image.shape[0], image.shape[1], image.shape[0], image.shape[1])).astype(np.int32)
        output_dict['detection_boxes'][:,2] = np.minimum(output_dict['detection_boxes'][:,2], image.shape[0] - 1)
        output_dict['detection_boxes'][:,3] = np.minimum(output_dict['detection_boxes'][:,3], image.shape[1] - 1)

        tracks_active = _process_people_flow(image_idx, image, output_dict, all_counters, tracks_active, track_id_start, grid_detection_history)

        available_counters = detect_counters(all_counters, image_idx)
        # available_counters = list(range(0, len(desk_positions)))
        for track in tracks_active:
            track['last_frame_idx'] = image_idx
            all_tracks[track['track_id']] = track
        if len(tracks_active) > 0:
            track_id_start = 1 + tracks_active[-1]['track_id']
        grids_weight = process_and_draw_track_lines(image, grid_detection_history, image_idx, draw_flow=draw and draw_flow)


        # # draw counter tables
        draw_counter_tables(image, desk_positions, draw_counter=draw and draw_counter)

        if draw:
            if image_idx // PROCESS_FPS == 74:
                print(image_idx // PROCESS_FPS)
            draw_grid_lines(image, draw_grid)
            basic_queue_lines = get_baisc_queue_lines(available_counters, desk_positions, grid_detection_history, image, draw_queue_line=draw and draw_basic_queue_line)
            basic_queue_regions, queues_upper_limit_line = get_basic_queue_regions(basic_queue_lines, desk_positions, image=image, draw_queue_region=draw and draw_basic_queue_region)
            service_regions, service_regions_grids = get_and_draw_service_regions(basic_queue_lines, image, draw_service_region=draw_service_region)
            basic_queue_detections = get_and_draw_queues(basic_queue_regions, output_dict.copy(), image, image_idx,
                                                   grid_detection_history, service_regions, max_gap=IMAGE_WIDTH/3,
                                                   draw_queue_people=draw and draw_queue_people,
                                                   draw_non_queue_people=draw and draw_non_queue_people)
            # queue_detections = basic_queue_detections
            dynamic_queue_lines = get_and_draw_dynamic_queue_lines(grids_weight, service_regions_grids,
                                                                            basic_queue_lines, basic_queue_detections,
                                                                            table_face_slope, image=image,
                                                                            draw_queue_line=draw and draw_queue_line)
            dynamic_queue_regions = get_queue_regions(dynamic_queue_lines, queues_upper_limit_line, desk_positions, IMAGE_WIDTH, IMAGE_HEIGHT, image=image,
                                              draw_queue_region=draw and draw_queue_region)
            queue_detections = get_and_draw_queues(dynamic_queue_regions, output_dict, image, image_idx,
                                                   grid_detection_history, service_regions,
                                                   draw_queue_people=draw and draw_queue_people,
                                                   draw_non_queue_people=draw and draw_non_queue_people)

            # visual stat
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, 'Time: %d' % (image_idx // PROCESS_FPS), (100, 100), font, 2, (137, 244, 66), 2, cv2.LINE_AA)
            # cv2.putText(image, '%s' % available_counters, (100, 250), font, 3, (0, 0, 255), 3, cv2.LINE_AA)
            for queue_idx in available_counters:
                if queue_idx in queue_detections:
                    text_position = (desk_positions[queue_idx][0][0] - 30, desk_positions[queue_idx][0][1] - 100)
                    cv2.putText(image, '%d' % len(queue_detections[queue_idx]), text_position, font, 3,
                                COLOR_LIST[queue_idx], 2, cv2.LINE_AA)
            cv2.imshow('frame', image)
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                return

            if output_processor:
                output_processor(image_idx, queue_detections, image=image)

        print(image_idx, time.time() - start)
        start = time.time()
    # time.sleep(100)
    cv2.destroyAllWindows()

def get_nearest_grid_id(x, y):
    return (x//GRID_SIZE, y//GRID_SIZE)
def get_grid_center(grid_x, grid_y):
    return (grid_x*GRID_SIZE+GRID_SIZE//2, grid_y*GRID_SIZE+GRID_SIZE//2)


def extend_line(point1, point2, width, height, max_extend_ratio=0):
    if point1[0] == point2[0]:
        return (point2[0], height if point2[1] >= point1[1] else 0)
    elif point1[1] == point2[1]:
        return (width if point2[0] >= point1[0] else 0, point2[1])
    else:
        slope = (point2[1] - point1[1])/(point2[0] - point1[0])
        if point2[0] > point1[0]:
            if max_extend_ratio > 0:
                x = min(width, point2[0] + (point2[0] - point1[0])*max_extend_ratio)
            else:
                x = width
        else:
            if max_extend_ratio > 0:
                x = max(0, point2[0] + (point2[0] - point1[0])*max_extend_ratio)
            else:
                x = 0
        y = slope * (x - point2[0]) + point2[1]
        return (int(x), int(y))


def process_and_draw_track_lines(image, grid_detection_history, current_frame_idx, show_last_n_frames=PROCESS_FPS*MAX_TRACK_SECONDS, draw_flow=False):
    grids_weight = np.zeros((IMAGE_WIDTH//GRID_SIZE, IMAGE_HEIGHT//GRID_SIZE))
    for grid_id, grid_counts in grid_detection_history.items():
        grid_weight = 0
        for frame_idx, count in reversed(grid_counts):
            if current_frame_idx - frame_idx > show_last_n_frames:
                break
            grid_weight += max(0, ((5 - 5*(current_frame_idx-frame_idx)//show_last_n_frames) / 5))
        if draw_flow:
            cv2.circle(image, get_grid_center(*grid_id), min(30, math.floor(grid_weight / 10)), (0, 0, 255), -1)
        grids_weight[grid_id] = grid_weight
    return grids_weight
    # grid_count = collections.defaultdict(int)
    # grid_connect = collections.defaultdict(int)
    # for track_id, track in tracks_active.items():
    #     last_n_frame = show_last_n_frames - (current_frame_idx - track['last_frame_idx'])
    #
    #     # for i in range(max(1, len(track['bboxes']) - last_n_frame), len(track['bboxes'])):
    #     #     cv2.line(image, track['center'][i - 1], track['center'][i], (0, 0, 255))
    #
    #     for i in range(max(0, len(track['bboxes']) - last_n_frame), len(track['bboxes'])):
    #         grid_count[get_nearest_grid_id(*track['center'][i])] += 1
    #
    #     last_n_frame = 3*show_last_n_frames - (current_frame_idx - track['last_frame_idx'])
    #     for i in range(max(1, len(track['bboxes']) - last_n_frame), len(track['bboxes'])):
    #         # only consider x axis movement and move towards counter
    #         if get_nearest_grid_id(*track['center'][i]) != get_nearest_grid_id(*track['center'][i-1]) :
    #         #         get_nearest_grid_id(*track['center'][i])[0] < get_nearest_grid_id(*track['center'][i-1])[0]:
    #             grid_connect[(get_nearest_grid_id(*track['center'][i-1]), get_nearest_grid_id(*track['center'][i]))] += 1
    #
    # for grid_id, grid_count in grid_count.items():
    #     if math.floor(grid_count/20) > 1:
    #         cv2.circle(image, get_grid_center(*grid_id), min(30, math.floor(grid_count/10)), (0, 0, 255), -1)
    # for grid_id_tuple, grid_count in grid_connect.items():
    #     if grid_count > 2:
    #         cv2.line(image, get_grid_center(*grid_id_tuple[0]), get_grid_center(*grid_id_tuple[1]), (0, 255, 0), math.ceil(grid_count/3))


def filter_detections_inside_regions(detections, masks):
    polygons = [Polygon(mask) for mask in masks]
    filtered_detection_indice = []
    for idx, detection in enumerate(detections):
        if any(polygon.contains(Point(*center(detection))) for polygon in polygons):
            filtered_detection_indice.append(idx)
    return filtered_detection_indice


def filter_detection_by_prolong_grid(region_indice, detections, grid_detection_history, current_frame_idx):
    def is_prolong(grid_id, last_n_seconds=5):
        if grid_id not in grid_detection_history:
            return False
        grid_history = grid_detection_history[grid_id]
        showup_count = 0
        for frame_idx, count in reversed(grid_history):
            if current_frame_idx == frame_idx:
                continue
            if current_frame_idx - frame_idx > 10*last_n_seconds*PROCESS_FPS:
                break
            showup_count += count * max(0, ((10 - math.floor((current_frame_idx-frame_idx)/(last_n_seconds*PROCESS_FPS))) / 5))
            if showup_count >= last_n_seconds*PROCESS_FPS*0.1:
                return True
    prolong_indice = []
    for idx in region_indice:
        detection = detections[idx]
        grid_id = get_nearest_grid_id(*center(detection))
        if is_prolong(grid_id):
            prolong_indice.append(idx)
    return prolong_indice


def filter_detection_by_far_detection(region_indice, detections, service_region, max_gap):
    def sort_detection_by_x_axis():
        return sorted([(idx, detection) for idx, detection in zip(region_indice, detections[region_indice])], key=lambda x: center(x[1])[0])
    sorted_detections = sort_detection_by_x_axis()
    filtered_indice = []
    if len(sorted_detections) > 0:
        first_detection = sorted_detections[0]
        if abs(first_detection[1][1] - get_grid_center(*service_region[1])[0]) >= IMAGE_WIDTH/6:
            return []
        filtered_indice.append(first_detection[0])
        for i in range(1, len(sorted_detections)):
            if abs(sorted_detections[i][1][1] - sorted_detections[i-1][1][1]) >= max_gap:
                break
            else:
                filtered_indice.append(sorted_detections[i][0])
        return sorted(filtered_indice)
    else:
        return region_indice


def get_counter_staff_indices(detections, desk_positions, image_shape):
    detection_index_to_counter_map = dict()
    height, width = image_shape[:2]

    for desk_idx, desk_position in enumerate(desk_positions):
        desk_region = [extend_line(desk_position[1], desk_position[0], width, height),
                       extend_line(desk_position[1], desk_position[0], width, height, max_extend_ratio=1),
                       extend_line(desk_position[2], desk_position[3], width, height, max_extend_ratio=1),
                       extend_line(desk_position[2], desk_position[3], width, height)]
        desk_polygon = Polygon(desk_region)
        for detection_idx, detection in enumerate(detections):
            if detection_idx in detection_index_to_counter_map:
                continue
            detection_center_point = Point(*center(detection))
            if desk_polygon.contains(detection_center_point):
                detection_index_to_counter_map[detection_idx] = desk_idx
    return detection_index_to_counter_map


def detect_counters(counters, current_frame_idx):
    available_counters = []
    for counter_idx in sorted(counters.keys()):
        counter_frames = counters[counter_idx]
        frame_count = len(counter_frames)
        if frame_count > PROCESS_FPS*15*0.8 and current_frame_idx - counter_frames[-1] <= 60*PROCESS_FPS and \
            (frame_count / (current_frame_idx - counter_frames[0] + 1)) > 0.05:
            available_counters.append(counter_idx)
        # elif frame_count > 0:
        #     print(current_frame_idx//PROCESS_FPS, ':', counter_idx, counter_frames[0], counter_frames[-1], frame_count, current_frame_idx)
    return available_counters


def get_baisc_queue_lines(available_counters, desk_positions, grid_detection_history, image, draw_queue_line=False):
    queue_lines = dict()
    for counter_idx in available_counters:
        desk_position = desk_positions[counter_idx]
        queue_line_points = get_desk_center_line(desk_position, image.shape[1], image.shape[0])
        # cv2.line(image, queue_line_points[0], queue_line_points[1], (255, 0, 255))
        queue_lines[counter_idx] = [queue_line_points[0], queue_line_points[1]]
        if draw_queue_line:
            cv2.line(image, *queue_lines[counter_idx], COLOR_LIST[counter_idx])
    return queue_lines


def get_basic_queue_regions(queue_lines, desk_positions, image=None, draw_queue_region=False):
    queue_regions = dict()
    queues_upper_limit_line = dict()
    last_queue_line = None
    available_queue_counters = sorted(queue_lines.keys())
    for process_idx in range(len(available_queue_counters)):
        counter_idx = available_queue_counters[process_idx]
        queue_region_points = []
        if last_queue_line is None:
            last_queue_line = []
            queue_region_points.append(desk_positions[counter_idx][4])
            point1 = center([*desk_positions[counter_idx][0], *desk_positions[counter_idx][3]], y1x1=False)
            point2 = center([*desk_positions[counter_idx][1], *desk_positions[counter_idx][2]], y1x1=False)
            last_queue_line.append(point1)
            last_queue_line.append(extend_line(point1, point2, IMAGE_WIDTH, IMAGE_HEIGHT))
            queue_region_points.extend(last_queue_line)
            queue_region_points.append(extend_line(desk_positions[counter_idx][4], desk_positions[counter_idx][5], IMAGE_WIDTH, IMAGE_HEIGHT))
            # last_queue_line.append(desk_positions[counter_idx][3])
            # last_queue_line.append(extend_line(desk_positions[counter_idx][3], desk_positions[counter_idx][2], width, height))
        else:
            queue_region_points.extend(last_queue_line)
            last_queue_line = []
            if process_idx + 1 < len(available_queue_counters):
                next_counter_id = available_queue_counters[process_idx + 1]
                line_head = center([*desk_positions[counter_idx][3], *desk_positions[next_counter_id][3]], y1x1=False)
                last_queue_line.append(line_head)
                line_end_1 = extend_line(desk_positions[counter_idx][3],
                                         desk_positions[counter_idx][2], IMAGE_WIDTH, IMAGE_HEIGHT)
                line_end_2 = extend_line(desk_positions[next_counter_id][3],
                                         desk_positions[next_counter_id][2], IMAGE_WIDTH, IMAGE_HEIGHT)
                line_end = center([*line_end_1, *line_end_2], y1x1=False)
                last_queue_line.append(line_end)
            else:
                extended_counter_idx = counter_idx
                if (counter_idx + 1) not in queue_lines and (counter_idx + 1) < len(desk_positions):
                    if (counter_idx + 2) not in queue_lines and (counter_idx + 2) < len(desk_positions):
                        extended_counter_idx = counter_idx + 2
                    else:
                        extended_counter_idx = counter_idx + 1
                last_queue_line.append(desk_positions[extended_counter_idx][3])
                last_queue_line.append(extend_line(desk_positions[extended_counter_idx][3], desk_positions[extended_counter_idx][2], IMAGE_WIDTH, IMAGE_HEIGHT))
            queue_region_points.extend(last_queue_line)
        queue_regions[counter_idx] = queue_region_points
        queues_upper_limit_line[counter_idx] = list(last_queue_line)
        last_queue_line.reverse()
        if draw_queue_region and image is not None:
            cv2.polylines(image, [np.array(queue_region_points)], True, COLOR_LIST[counter_idx])
    return queue_regions, queues_upper_limit_line


def get_queue_regions(queue_lines, queues_upper_limit_line, desk_positions, width, height, image=None, draw_queue_region=False):
    queue_regions = dict()
    last_queue_line = None
    available_queue_counters = sorted(queue_lines.keys())
    BASIC_QUEUE_REGION_MARGIN = 30
    for process_idx in range(len(available_queue_counters)):
        counter_idx = available_queue_counters[process_idx]
        queue_upper_limit_points = queues_upper_limit_line[counter_idx]
        queue_region_points = []
        if last_queue_line is None:
            last_queue_line = []
            for grid_id_x, grid_id_y in queue_lines[counter_idx]:
                x, y = get_grid_center(grid_id_x, grid_id_y)
                if process_idx + 1 < len(available_queue_counters):
                    y_margin = min(BASIC_QUEUE_REGION_MARGIN*3, BASIC_QUEUE_REGION_MARGIN*(available_queue_counters[process_idx + 1] - counter_idx))
                else:
                    y_margin = BASIC_QUEUE_REGION_MARGIN*3
                y = int(y) - y_margin
                # dont exceed upper limit by margin, here upper limit means the lower the y value, the upper
                y_upper_limit = get_line_y_by_2point(*queue_upper_limit_points, x)
                if y < y_upper_limit - BASIC_QUEUE_REGION_MARGIN:
                    y = int(y_upper_limit - BASIC_QUEUE_REGION_MARGIN)
                last_queue_line.append((x, y))

            margin = np.array([0, -GRID_SIZE])
            queue_region_points.append(desk_positions[counter_idx][4])
            queue_region_points.append(margin+desk_positions[counter_idx][0])
            queue_region_points.append(margin+desk_positions[counter_idx][3])
            queue_region_points.append(desk_positions[counter_idx][2])
            queue_region_points.extend(last_queue_line)
            queue_region_points.append(extend_line(desk_positions[counter_idx][4], desk_positions[counter_idx][5], width, height))
        else:
            queue_region_points.extend(last_queue_line)
            queue_region_points.append(margin+desk_positions[counter_idx][0])
            queue_region_points.append(margin+desk_positions[counter_idx][3])
            queue_region_points.append(desk_positions[counter_idx][2])
            last_queue_line = []
            if process_idx + 1 >= len(available_queue_counters):
                extended_counter_idx = counter_idx
                if (counter_idx + 1) not in queue_lines and (counter_idx + 1) < len(desk_positions):
                    if (counter_idx + 2) not in queue_lines and (counter_idx + 2) < len(desk_positions):
                        extended_counter_idx = counter_idx + 2
                    else:
                        extended_counter_idx = counter_idx + 1
                queue_region_points.append(margin+desk_positions[extended_counter_idx][3])
                queue_region_points.append(extend_line(desk_positions[extended_counter_idx][3], desk_positions[extended_counter_idx][2], width, height))
            else:
                next_counter_idx = available_queue_counters[process_idx + 1]
                next_queue_line_dict = dict(queue_lines[next_counter_idx])
                for grid_id_x, grid_id_y in queue_lines[counter_idx]:
                    if grid_id_x not in next_queue_line_dict:
                        x, y = get_grid_center(grid_id_x, grid_id_y)
                        y_margin = 90
                        y -= y_margin
                    else:
                        avg_grid_id_y = (grid_id_y + next_queue_line_dict[grid_id_x])/2
                        x, y = get_grid_center(grid_id_x, avg_grid_id_y)

                    # dont exceed upper limit by margin, here upper limit means the lower the y value, the upper
                    y_upper_limit = get_line_y_by_2point(*queue_upper_limit_points, x)
                    if y < y_upper_limit - BASIC_QUEUE_REGION_MARGIN:
                        y = int(y_upper_limit - BASIC_QUEUE_REGION_MARGIN)
                    last_queue_line.append((x, int(y)))
                queue_region_points.extend(last_queue_line)
        queue_regions[counter_idx] = queue_region_points
        last_queue_line.reverse()
        if draw_queue_region and image is not None:
            cv2.polylines(image, [np.array(queue_region_points)], True, COLOR_LIST[counter_idx])
    return queue_regions


def get_and_draw_queues(queue_regions, output_dict, image, current_frame_idx, grid_detection_history, service_regions,
                        max_gap=IMAGE_WIDTH/4, draw_queue_people=True, draw_non_queue_people=False):
    queue_detections = dict()
    for queue_idx in queue_regions:
        region_indice = filter_detections_inside_regions(output_dict['detection_boxes'], [queue_regions[queue_idx]])
        region_indice = filter_detection_by_prolong_grid(region_indice, output_dict['detection_boxes'], grid_detection_history, current_frame_idx)
        # filter if too far away from service region / last detected person in terms of x axis pixel.
        region_indice = filter_detection_by_far_detection(region_indice, output_dict['detection_boxes'], service_regions[queue_idx], max_gap)
        queue_detections[queue_idx] = output_dict['detection_boxes'][region_indice]
        if draw_queue_people:
            label_image(image, dict(detection_boxes=output_dict['detection_boxes'][region_indice]),
                    category_index, min_score_thresh=min_score_thresh, use_normalized_coordinates=False,
                    box_color=COLOR_NAME_LIST[queue_idx])
        remove_detection(output_dict, filter_indice=region_indice)
    if draw_non_queue_people:
        label_image(image, output_dict, category_index, min_score_thresh=min_score_thresh, use_normalized_coordinates=False)
    return queue_detections


def get_desk_center_line(desk_position, width, height):
    point1 = center((desk_position[1][0], desk_position[1][1], desk_position[6][0], desk_position[6][1]), y1x1=False)
    point2 = (point1[0] + (desk_position[1][0] - desk_position[0][0]),
              point1[1] + (desk_position[1][1] - desk_position[0][1]))
    point2 = extend_line(point1, point2, width, height)
    return point1, point2


def draw_counter_tables(image, desk_positions, draw_counter=False):
    if draw_counter:
        for desk_idx, desk_position in enumerate(desk_positions):
            # table 2d
            # print(desk_position)
            cv2.polylines(image, [np.array(desk_position[:4])], True, COLOR_LIST[desk_idx])
            # table 3d
            cv2.line(image, desk_position[1], desk_position[5], COLOR_LIST[desk_idx])
            cv2.line(image, desk_position[0], extend_line(desk_position[0], desk_position[1], image.shape[1], image.shape[0]), COLOR_LIST[desk_idx])
            cv2.line(image, desk_position[3], extend_line(desk_position[3], desk_position[2], image.shape[1], image.shape[0]), COLOR_LIST[desk_idx])
            cv2.line(image, desk_position[4], extend_line(desk_position[4], desk_position[5], image.shape[1], image.shape[0]), COLOR_LIST[desk_idx])


def draw_grid_lines(image, draw_grid):
    if draw_grid:
        vertical_lines = [np.array([(i*GRID_SIZE, 0), (i*GRID_SIZE, IMAGE_HEIGHT)]) for i in range(0, IMAGE_WIDTH//GRID_SIZE)]
        horizontal_lines = [np.array([(0, i*GRID_SIZE), (IMAGE_WIDTH, i*GRID_SIZE)]) for i in range(0, IMAGE_WIDTH//GRID_SIZE)]
        vertical_lines.extend(horizontal_lines)
        cv2.polylines(image, vertical_lines, True, (0, 255, 255))


def get_and_draw_service_regions(basic_queue_lines, image, draw_service_region=False):
    service_regions = dict()
    service_regions_grids = dict()
    for counter_idx in sorted(basic_queue_lines.keys()):
        point1, _ = basic_queue_lines[counter_idx]
        service_grid_to_search = SERVICE_REGION_GRIDS + get_nearest_grid_id(*point1)
        # if service_grid_to_search[0][0] < 0:
        #     service_grid_to_search += (-service_grid_to_search[0][0], 0)
        # elif service_grid_to_search[-1][0] >= IMAGE_WIDTH // GRID_SIZE:
        #     service_grid_to_search -= (service_grid_to_search[-1][0] - (IMAGE_WIDTH // GRID_SIZE) + 1, 0)
        # if service_grid_to_search[0][1] < 0:
        #     service_grid_to_search += (0, -service_grid_to_search[0][1])
        # elif service_grid_to_search[-1][1] >= IMAGE_WIDTH // GRID_SIZE:
        #     service_grid_to_search -= (0, service_grid_to_search[-1][1] - (IMAGE_WIDTH // GRID_SIZE) + 1)
        service_regions[counter_idx] = [service_grid_to_search[0], service_grid_to_search[-1]]
        service_regions_grids[counter_idx] = service_grid_to_search
        if draw_service_region:
            cv2.rectangle(image, get_grid_center(*service_grid_to_search[0]),
                          get_grid_center(*service_grid_to_search[-1]), COLOR_LIST[counter_idx], thickness=2)
    return service_regions, service_regions_grids


def get_and_draw_dynamic_queue_lines(grids_weight, service_regions_grids,
                                 basic_queue_lines, basic_queue_detections, table_face_slope,
                                 image=None, draw_queue_line=False):
    BASIC_QUEUE_REGION_MARGIN = 30
    queue_lines = dict()
    grid_x_occupy = dict()
    # service_regions = dict()
    available_queue_counters = sorted(basic_queue_lines.keys())
    for process_idx in range(len(available_queue_counters)):
        counter_idx = available_queue_counters[process_idx]
        point1, _ = basic_queue_lines[counter_idx]
        queue_line = []
        if counter_idx not in basic_queue_detections or len(basic_queue_detections[counter_idx]) == 0:
            # give 30 pixel margin
            # queue_lines[counter_idx] = [(x//GRID_SIZE, y/GRID_SIZE - BASIC_QUEUE_REGION_MARGIN/GRID_SIZE) for x, y in basic_queue_lines[counter_idx]]

            basic_queue_head = basic_queue_lines[counter_idx][0]
            queue_line.append((basic_queue_head[0]//GRID_SIZE, basic_queue_head[1]/GRID_SIZE - BASIC_QUEUE_REGION_MARGIN/GRID_SIZE))
            last_grid_x, last_grid_y = queue_line[0]
            while last_grid_x < IMAGE_WIDTH // GRID_SIZE - 1:
                grid_x = last_grid_x + 1
                grid_y = get_line_y_by_slope(table_face_slope, (last_grid_x, last_grid_y), grid_x)
                grid_x_occupy[grid_x] = grid_y
                queue_line.append((grid_x, grid_y))
                last_grid_x = grid_x
                last_grid_y = grid_y
            queue_lines[counter_idx] = queue_line

            continue
        queue_end_x_grid_id = math.ceil(sorted(basic_queue_detections[counter_idx], key=lambda x: x[3])[-1][3]/GRID_SIZE)
        service_region_grids = service_regions_grids[counter_idx]
        service_region_weights = grids_weight[service_region_grids[:, 0], service_region_grids[:, 1]].reshape((SERVICE_GRID_NUM, SERVICE_GRID_NUM))
        service_region_weight_sum = service_region_weights.sum()
        if service_region_weight_sum > 0:
            # max_weight_idx = service_region_weights.argmax()
            # max_weight_grid_id = service_region_grids[max_weight_idx]
            # max_weight = grids_weight[tuple(max_weight_grid_id)]
            first_grid_x = 1.0*(service_region_weights*range(service_region_weights.shape[1])).sum()/service_region_weight_sum
            y_points = np.arange(service_region_weights.shape[0]).reshape((service_region_weights.shape[0], 1))
            first_grid_y = 1.0*(service_region_weights*y_points).sum()/service_region_weight_sum
            first_grid_id = service_region_grids[0] + (first_grid_x, first_grid_y)
        else:
            # max_grid_weight = 0
            first_grid_id = get_nearest_grid_id(*point1)
        last_grid_x = int(round(first_grid_id[0]))
        last_grid_y = first_grid_id[1]
        queue_line.append((int(round(last_grid_x)), int(round(last_grid_y))))

        queue_line_weighted_sum = collections.deque(maxlen=IMAGE_WIDTH//GRID_SIZE//8)
        # basic queue end
        # cv2.circle(image, get_grid_center(queue_end_x_grid_id, int(round(get_line_y_by_slope(table_face_slope, (last_grid_x, last_grid_y), queue_end_x_grid_id)))), 100, COLOR_LIST[counter_idx], thickness=2)
        # queue_upper_limit_points = queues_upper_limit_line[counter_idx]
        while last_grid_x < IMAGE_WIDTH // GRID_SIZE - 1:
            if last_grid_x <= queue_end_x_grid_id:
                grid_x = last_grid_x + 1
                last_grid_y += table_face_slope # add 1 grid_x unit, increase slope grid y unit
                round_grid_y = int(round(last_grid_y))
                y_grid_to_search = SERVICE_REGION_COLUMNS + round_grid_y
                service_column_weigted_grids = grids_weight[grid_x, y_grid_to_search] * SERVICE_REGION_COLUMNS_WEIGHT
                service_column_weigted_grids_sum = service_column_weigted_grids.sum()
                # print(service_column_weigted_grids_sum)
                if service_column_weigted_grids_sum > 0:
                    grid_y_weighted = 1.0 * (service_column_weigted_grids * np.arange(SERVICE_REGION_COLUMNS_WEIGHT.shape[0])).sum() / service_column_weigted_grids_sum
                    grid_y_shift = SERVICE_REGION_COLUMNS[0] + grid_y_weighted
                    grid_y_shift = min(1/3, max(-1/3, grid_y_shift*service_column_weigted_grids_sum/(SERVICE_REGION_PIXEL/2)))
                    grid_y = last_grid_y + grid_y_shift
                else:
                    grid_y = round_grid_y
                if grid_x in grid_x_occupy:
                    current_min_grid_y = grid_x_occupy[grid_x]
                    if current_min_grid_y - grid_y <= SERVICE_REGION_PIXEL/GRID_SIZE/2:
                        grid_y = current_min_grid_y - SERVICE_REGION_PIXEL/GRID_SIZE/2
                basic_predicted_grid_y = (grid_x - first_grid_id[0])*table_face_slope + first_grid_id[1]
                if grid_y - basic_predicted_grid_y >= 2:
                    grid_y = basic_predicted_grid_y + 2
                elif basic_predicted_grid_y - grid_y >= 2:
                    grid_y = basic_predicted_grid_y - 2

                queue_line_weighted_sum.append(service_column_weigted_grids_sum)
                if len(queue_line_weighted_sum) >= IMAGE_WIDTH//GRID_SIZE//8 and sum(queue_line_weighted_sum) < 300:
                    grid_y = basic_predicted_grid_y

                # # dont exceed upper limit by margin, here upper limit means the lower the y value, the upper
                # upper_limit = get_line_y_by_2point(*queue_upper_limit_points, grid_x*GRID_SIZE)
                # if grid_y*GRID_SIZE < upper_limit - BASIC_QUEUE_REGION_MARGIN:
                #     grid_y = (upper_limit - BASIC_QUEUE_REGION_MARGIN)/GRID_SIZE
                grid_x_occupy[grid_x] = grid_y
                queue_line.append((grid_x, grid_y))
                last_grid_x = grid_x
                last_grid_y = grid_y
            else:
                grid_x = last_grid_x + 1
                grid_y = get_line_y_by_slope(table_face_slope, (last_grid_x, last_grid_y), grid_x)
                grid_x_occupy[grid_x] = grid_y
                queue_line.append((grid_x, grid_y))
                last_grid_x = grid_x
                last_grid_y = grid_y

        queue_lines[counter_idx] = queue_line
        if draw_queue_line:
            for grid_x, grid_y in queue_line:
                cv2.circle(image, get_grid_center(grid_x, int(round(grid_y))), 50, COLOR_LIST[counter_idx], thickness=2)
    return queue_lines


def output_processor1():
    fw = open('d1_hard_code_region_count/D1_count.csv', 'w')

    masks = [
        [(382,486), (773, 357), (775, 518), (383, 657)],
        [(383, 657), (774, 355), (1920, 461), (1920, 832)],
    ]

    def _run(frame_idx, image, output_dict):
        detections = output_dict['detection_boxes']
        current_sec = frame_idx//PROCESS_FPS
        if frame_idx % PROCESS_FPS == 0:
            retain_indice = filter_detections_inside_regions(detections, masks)
            fw.write('%d,%d\n' % (current_sec, len(retain_indice)))
            if current_sec % 15 == 0:
                output_dict_copy = output_dict.copy()
                file_path = 'd1_hard_code_region_count/%d_original.jpg' % current_sec
                image_copy = image.copy()
                label_image(image_copy, output_dict_copy, category_index, min_score_thresh=min_score_thresh, use_normalized_coordinates=False)
                cv2.imwrite(file_path, image_copy)

                output_dict_copy = output_dict.copy()
                remove_detection(output_dict_copy, retain_indice=retain_indice)
                file_path = 'd1_hard_code_region_count/%d_region.jpg' % current_sec
                image_copy = image.copy()
                label_image(image_copy, output_dict_copy, category_index, min_score_thresh=min_score_thresh, use_normalized_coordinates=False)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image_copy, '%d' % len(retain_indice), (1720, 100), font, 4, (0, 0, 255), 4, cv2.LINE_AA)

                for mask in masks:
                    pts = np.array(mask, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(image_copy, [pts], True, (0, 255, 255))

                cv2.imwrite(file_path, image_copy)
    return _run


def get_output_processor(max_queue=10, csv_path=None, video_writer=None, folder_path=None):
    if csv_path:
        fw = open(csv_path, 'w')
        fw.write('time,%s\n' % ','.join(['queue %s count, queue %s length' % (i,i) for i in range(1, max_queue+1)]))
    else:
        fw = None

    if folder_path:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def _run(frame_idx, queues_detections, image=None):
        current_sec = frame_idx // PROCESS_FPS
        if frame_idx % PROCESS_FPS == 0:
            if fw is not None:
                queues_info = []
                for queue_idx in range(max_queue):
                    if queue_idx in queues_detections and len(queues_detections[queue_idx]) > 0:
                        queue_detections = queues_detections[queue_idx]
                        queue_x_start = sorted(queue_detections, key=lambda x: x[1])[0][1]
                        queue_x_end = sorted(queue_detections, key=lambda x: x[3])[-1][3]
                        queues_info.append(str(len(queue_detections)) + ',' + str(queue_x_end - queue_x_start))
                    else:
                        queues_info.append('0,0')
                queue_count_str = ','.join(queues_info)
                fw.write('%d,%s\n' % (current_sec, queue_count_str))

            if video_writer is not None:
                video_writer.write(image)

            # if folder_path is not None and image is not None and current_sec % 15 == 0:
            #     file_path = os.path.join(folder_path, '%d.jpg' % current_sec)
            #     cv2.imwrite(file_path, image)
    return _run


def get_default_desk_positions():
    ground_direction = [(460, 703), (1404, 816)]
    ground_slope = (ground_direction[1][1] - ground_direction[0][1])/(ground_direction[1][0] - ground_direction[0][0])
    desk_positions = [
        [#(172, 524),
         (208,532), (323, 497), (208, 722)],
        [#(400, 450),
         (446,458), (544, 430), (446, 636)],
        [#(548, 410),
         (584, 416), (671, 392), (583, 590)],
        [#(730, 356),
         (772, 363), (849, 339), (772, 515)],
    ]
    table_width = 40
    # generate points
    for desk_position in desk_positions:
        # deduce point A by ground slope
        desk_position.insert(0, (desk_position[0][0] - table_width, int(desk_position[0][1] - ground_slope*table_width)))
        desk_position.insert(3, (desk_position[0][0] + (desk_position[2][0] - desk_position[1][0]),
                                 desk_position[0][1] + (desk_position[2][1] - desk_position[1][1])))
        desk_position.insert(4, (desk_position[0][0] + (desk_position[4][0] - desk_position[1][0]),
                                 desk_position[0][1] + (desk_position[4][1] - desk_position[1][1])))
        desk_position.append((desk_position[5][0] + (desk_position[2][0] - desk_position[1][0]),
                                 desk_position[5][1] + (desk_position[2][1] - desk_position[1][1])))
    return desk_positions


def get_slope(point1, point2):
    if abs(point2[0] - point1[0]) > 0:
        return (point2[1] - point1[1]) / (point2[0] - point1[0])
    else:
        return 0


def get_desk_positions(table_face_direction, table_horizontal_direction, desk_positions, table_width_ratio=4/10):
    desk_full_positions = []
    vertical_slope = get_slope(*table_face_direction)
    horizontal_line = LineString([extend_line(table_horizontal_direction[1], table_horizontal_direction[0], IMAGE_WIDTH, IMAGE_HEIGHT),
                       extend_line(table_horizontal_direction[0], table_horizontal_direction[1], IMAGE_WIDTH, IMAGE_HEIGHT)])
    for desk_position in desk_positions:
        point2 = desk_position[0]
        point3 = desk_position[1]
        table_width = abs(table_width_ratio*(point3[0] - point2[0]))
        point1 = (int(point2[0] - table_width), int(desk_position[0][1] - vertical_slope*table_width))
        point4 = (point1[0] + (point3[0] - point2[0]),
                  point1[1] + (point3[1] - point2[1]))
        point6 = horizontal_line.interpolate(horizontal_line.project(Point(point2)))
        point6 = (int(point6.x), int(point6.y))
        point7 = horizontal_line.interpolate(horizontal_line.project(Point(point3)))
        point7 = (int(point7.x), int(point7.y))
        point5 = (point1[0] + (point6[0] - point2[0]),
                  point1[1] + (point6[1] - point2[1]))
        desk_full_positions.append([point1, point2, point3, point4, point5, point6, point7])
    return vertical_slope, desk_full_positions


def detect_image_by_worker(input_q, output_q, gpu_device=0, boostrap_detections=None):
    import tensorflow as tf

    inference_graph_path = '/app/object_detection_app/models/person_inceptionv2/model.pb'
    graph, sess = get_session(inference_graph_path=inference_graph_path, gpu_device=gpu_device)

    # with graph.as_default():
    #     with tf.device(device):
    while True:
        image_idx, image = input_q.get()
        if boostrap_detections is not None and image_idx in boostrap_detections:
            output_dict = boostrap_detections[image_idx]
        else:
            output_dict = run_inference_for_single_image(sess, image, graph=graph)
        output_q.put((image_idx, image, output_dict))


if __name__ == '__main__':
    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # so the IDs match nvidia-smi
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

    useRT = False
    min_score_thresh = .5
    image_dir = '/app/powerarena-sense-gym/client_dataset/Cerie/JPEGImages'
    video_path = '/home/ma-glass/Downloads/D1.mp4'
    # video_path = '/mnt/2630afa8-db60-478d-ac09-0af3b44bead6/Downloads/IMGP1138.MOV'

    inference_graph_path = '/app/object_detection_app/models/person_inceptionv2/model.pb'

    # train person model
    model_folder = 'faster_rcnn_resnet101'
    train_version = 'train_aa_v3'
    checkpoint_dir = os.path.join(base_folder, 'model_output/person/%s/%s/' % (model_folder, train_version))

    # sess = get_session(inference_graph_path=inference_graph_path)
    # sess = get_session(checkpoint_dir=checkpoint_dir)
    sess = None

    # BOOTSTRAP_SEC = 1
    image_reader_bootstrap = get_image_reader(video_path=video_path, video_fps=PROCESS_FPS, max_video_length=BOOTSTRAP_SEC,
                                    output_minmax_size=(1080, 1920))
    image_reader = get_image_reader(video_path=video_path, video_fps=PROCESS_FPS, max_video_length=0,
                                    output_minmax_size=(1080, 1920))

    pickle_path = '/app/powerarena-sense-gym/models/research/pa_utils/project/aa/d1_detections_fps5.pkl'
    # pickle_path = '/app/powerarena-sense-gym/models/research/pa_utils/project/aa/d1_detections_fps10_resnet101.pkl'
    # pickle_path = '/app/powerarena-sense-gym/models/research/pa_utils/project/aa/IMGP1138_detections_fps5_resnet101_v0.pkl'
    frames_detections = load_detections_pickle(pickle_path)
    print('frames_detections length', len(frames_detections))
    # frames_detections = None

    # IMGP1138
    table_face_direction = [(420, 812), (1579, 962)]
    table_horizontal_direction = [(393, 749), (1579, 453)]
    desk_positions = [
        [(176, 520), (386, 481)],
        [(601, 440), (755, 409)],
        [(839, 393), (982, 364)],
        [(1172, 327), (1303, 305)],
        [(1361, 290), (1498, 263)],
        [(1562, 246), (1731, 219)],
        [(1780, 207), (1892, 190)],
    ]

    # D1
    table_face_direction = [(460, 703), (1404, 816)]
    table_horizontal_direction = [(207, 721), (1652, 207)]
    desk_positions = [
         [(208, 532), (323, 497)],
         [(446, 458), (544, 430)],
         [(584, 416), (671, 392)],
         [(772, 363), (849, 339)],
         [(887, 325), (957, 305)],
         [(1037, 279), (1100, 259)],
         [(1133, 250), (1194, 232)],
         [(1256, 212), (1312, 195)],
         [(1335, 188), (1386, 171)],
         [(1442, 156), (1491, 141)],
    ]

    table_face_slope, desk_positions = get_desk_positions(table_face_direction, table_horizontal_direction, desk_positions)
    # desk_positions = get_default_desk_positions()

    # output_video_path = 'D1(VA)2.mp4'
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # video_writer = cv2.VideoWriter(output_video_path, fourcc, 1, (IMAGE_WIDTH, IMAGE_HEIGHT))
    # csv_path = 'D1_count_v1_queue_length2.csv'
    # output_processor = get_output_processor(csv_path=csv_path, video_writer=video_writer)
    output_processor = None

    track_detections(frames_detections, image_reader_bootstrap, image_reader, sess=sess,
                     desk_positions=desk_positions, table_face_slope=table_face_slope,
                     min_score_thresh=min_score_thresh, wait_time=1, output_processor=output_processor)
    # video_writer.release()

    # queue detection flow:
    # Assume we can transform the video into designed angle.
    # 1. use basic queue region to check the queue length
    # 2a. if two nearby queues have comparable queue length, then we will use the weighted flow to determine the queue lines.
    # 2b. otherwise, if one queue is too short, compared with another, we use default queue region instead.
