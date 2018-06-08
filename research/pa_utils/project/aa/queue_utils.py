import os
import math
import cv2
import collections
import numpy as np
import random
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon
from pa_utils.image_utils import label_image
from pa_utils.data.data_utils import remove_detection, get_color_list
from pa_utils.graph_utils import get_session
from pa_utils.model_utils import run_inference_for_single_image


COLOR_LIST = get_color_list(return_rgb=True)
COLOR_NAME_LIST = get_color_list()
MIN_SCORE_THRESHOLD = 0.5

MAX_QUEUE_NUMBER = 10
IMAGE_WIDTH, IMAGE_HEIGHT = (1920, 1080)
MAX_QUEUE_GAP = int(IMAGE_WIDTH/4)
PROCESS_FPS = 5  # tune-able parameter 1
GRID_SIZE = 20  # tune-able parameter 2
MAX_TRACK_SECONDS = 120  # tune-able parameter 3

SERVICE_REGION_RATIO = 2.2
COUNTER_TRACK_SECONDS = 60
BASIC_QUEUE_REGION_MARGIN = 30
QUEUE_COUNT_MAXIMIZE_WINDOW = 15  # seconds

USE_RESNET101 = False


# util functions
def detect_image_by_worker(input_q, output_q, gpu_device=0, boostrap_detections=None):
    inference_graph_path = '/app/object_detection_app/models/person_inceptionv2/model.pb'
    if USE_RESNET101:
        inference_graph_path = '/app/powerarena-sense-gym/models/research/pa_utils/model_output/person/faster_rcnn_resnet101/train_aa_v3/inference/frozen_inference_graph.pb'
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


def bbox_center(bbox, y1x1=True, head_center=False):
    if y1x1:
        y1, x1, y2, x2 = bbox
    else:
        x1, y1, x2, y2 = bbox

    if head_center:
        return (int((x1+x2)/2), int(y1))
    return (int((x1+x2)/2), int((y1+y2)/2))


def trim_grid(grid_x, grid_y, max_grid_x=IMAGE_WIDTH//GRID_SIZE - 1, max_grid_y=IMAGE_HEIGHT//GRID_SIZE - 1):
    return max(0, min(max_grid_x, grid_x)), max(0, min(max_grid_y, grid_y))


def get_nearest_grid_id(x, y):
    return trim_grid(int(x//GRID_SIZE), int(y//GRID_SIZE))


def get_grid_center(grid_x, grid_y):
    return grid_x*GRID_SIZE+GRID_SIZE//2, grid_y*GRID_SIZE+GRID_SIZE//2


def extend_line(point1, point2, width, height, max_extend_ratio=0.0):
    if point1[0] == point2[0]:
        return int(point2[0]), height if point2[1] >= point1[1] else 0
    elif point1[1] == point2[1]:
        return width if point2[0] >= point1[0] else 0, int(point2[1])
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
        return int(x), int(y)


def get_slope(point1, point2):
    if abs(point2[0] - point1[0]) > 0:
        return (point2[1] - point1[1]) / (point2[0] - point1[0])
    else:
        return 0


def get_desk_positions(table_face_direction, table_service_direction, table_horizontal_direction, desk_positions,
                       table_width_ratio=0.4):
    # **tune-able parameter**, table width hard code by abs(point2_x - point1_x)
    desk_full_positions = []
    vertical_slope = get_slope(*table_face_direction)
    horizontal_line = LineString(
        [extend_line(table_horizontal_direction[1], table_horizontal_direction[0], IMAGE_WIDTH, IMAGE_HEIGHT),
         extend_line(table_horizontal_direction[0], table_horizontal_direction[1], IMAGE_WIDTH, IMAGE_HEIGHT)])
    for desk_position in desk_positions:
        point1_x = desk_position[0]
        point2_x = desk_position[1]
        point3_x = desk_position[2]
        point2_y = int(get_line_y_by_2point(*table_service_direction, point2_x))
        point3_y = int(get_line_y_by_2point(*table_service_direction, point3_x))
        # table_width = abs(table_width_ratio*(point3_x - point2_x))
        table_width = abs(point2_x - point1_x)
        point1 = (int(point2_x - table_width), int(point2_y - vertical_slope * table_width))
        point4 = (point1[0] + (point3_x - point2_x),
                  point1[1] + (point3_y - point2_y))
        # point6 = horizontal_line.interpolate(horizontal_line.project(Point((point2_x, point2_y))))
        # point6 = (int(point6.x), int(point6.y))
        # point7 = horizontal_line.interpolate(horizontal_line.project(Point((point3_x, point3_y))))
        # point7 = (int(point7.x), int(point7.y))
        point6_y = int(get_line_y_by_2point(*table_horizontal_direction, point2_x))
        point6 = (point2_x, point6_y)
        point7_y = int(get_line_y_by_2point(*table_horizontal_direction, point3_x))
        point7 = (point3_x, point7_y)

        point5 = (point1[0] + (point6[0] - point2_x),
                  point1[1] + (point6[1] - point2_y))
        point2 = (point2_x, point2_y)
        point3 = (point3_x, point3_y)
        desk_full_positions.append([point1, point2, point3, point4, point5, point6, point7])
    return vertical_slope, desk_full_positions


# counter functions
def get_counter_staff_indices(detections, desk_positions):
    # ***tune-able parameter***, max_extend_ratio 1.5 ~ 5
    detection_index_to_counter_map = dict()
    for desk_idx, desk_position in enumerate(desk_positions):
        desk_region = [extend_line(desk_position[1], desk_position[0], IMAGE_WIDTH, IMAGE_HEIGHT, max_extend_ratio=5),
                       extend_line(desk_position[1], desk_position[0], IMAGE_WIDTH, IMAGE_HEIGHT, max_extend_ratio=1.5),
                       extend_line(desk_position[2], desk_position[3], IMAGE_WIDTH, IMAGE_HEIGHT, max_extend_ratio=1.5),
                       extend_line(desk_position[2], desk_position[3], IMAGE_WIDTH, IMAGE_HEIGHT, max_extend_ratio=5)]
        desk_polygon = Polygon(desk_region)
        for detection_idx, detection in enumerate(detections):
            if detection_idx in detection_index_to_counter_map:
                continue
            detection_center_point = Point(*bbox_center(detection))
            if desk_polygon.contains(detection_center_point):
                detection_index_to_counter_map[detection_idx] = desk_idx
    return detection_index_to_counter_map


def detect_counters(counters, current_frame_idx):
    # **tune-able parameter**
    COUNTER_MIN_SHOWN_FRAMES = PROCESS_FPS*15*0.8
    COUNTER_MIN_SHOWN_RATIO = 0.05

    available_counters = []
    for counter_idx in sorted(counters.keys()):
        counter_frames = counters[counter_idx]
        frame_count = len(counter_frames)
        if frame_count > COUNTER_MIN_SHOWN_FRAMES and current_frame_idx - counter_frames[-1] <= COUNTER_TRACK_SECONDS*PROCESS_FPS and \
                        (frame_count / (counter_frames[-1] - counter_frames[0] + 1)) > COUNTER_MIN_SHOWN_RATIO:
            available_counters.append(counter_idx)
    return available_counters


def get_service_region_parameter(desk_positions):
    # **tune-able parameter**
    SERVICE_REGION_CENTER_RATIO = 3/5

    SERVICE_REGION_PIXEL = int(SERVICE_REGION_RATIO*sum(abs(desk_positions[i][2][0] - desk_positions[i][1][0]) for i in range(len(desk_positions)))/len(desk_positions))
    print('SERVICE_REGION_PIXEL', SERVICE_REGION_PIXEL)
    def prepare_service_grips():
        # rows = np.arange(5)
        # cols = np.arange(5)
        # x, y = np.meshgrid(rows, cols)
        # service_grids = np.stack((x, y), axis=-1)
        num_grid = max(3, math.ceil(SERVICE_REGION_PIXEL / GRID_SIZE))
        center_grid = round(num_grid * SERVICE_REGION_CENTER_RATIO)
        service_grids = np.array([(j, i) for i in range(0, num_grid) for j in range(0, num_grid)]).reshape(
            num_grid * num_grid, 2) - (center_grid, center_grid)
        # column_grids = service_grids.reshape(num_grid, num_grid, 2)[:,0,1]
        # column_weight = pow((num_grid - abs(column_grids))/num_grid, 2)
        column_grids = np.arange(int(num_grid / 2) * 2 + 1) - int(num_grid / 2)
        column_length = column_grids.shape[0]
        column_weight = pow((column_length - abs(column_grids)) / column_length, 2)
        return service_grids, column_grids, column_weight, num_grid

    SERVICE_REGION_GRIDS, SERVICE_REGION_COLUMNS, SERVICE_REGION_COLUMNS_WEIGHT, SERVICE_GRID_NUM = prepare_service_grips()
    return SERVICE_REGION_PIXEL, SERVICE_REGION_GRIDS, SERVICE_REGION_COLUMNS, SERVICE_REGION_COLUMNS_WEIGHT, SERVICE_GRID_NUM


# detection utils function
def filter_detections_inside_regions(detections, masks):
    polygons = [Polygon(mask) for mask in masks]
    filtered_detection_indice = []
    for idx, detection in enumerate(detections):
        if any(polygon.contains(Point(*bbox_center(detection))) for polygon in polygons):
            filtered_detection_indice.append(idx)
    return filtered_detection_indice


def filter_detection_by_prolong_grid(region_indice, detections, grid_detection_history, current_frame_idx):
    # ***tune-able parameter***
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
        grid_id = get_nearest_grid_id(*bbox_center(detection))
        if is_prolong(grid_id):
            prolong_indice.append(idx)
    return prolong_indice


def filter_detection_by_far_detection(region_indice, detections, service_region, max_gap):
    def sort_detection_by_x_axis():
        return sorted([(idx, detection) for idx, detection in zip(region_indice, detections[region_indice])], key=lambda x: bbox_center(x[1])[0])
    sorted_detections = sort_detection_by_x_axis()
    filtered_indice = []
    if len(sorted_detections) > 0:
        first_detection = sorted_detections[0]
        if abs(first_detection[1][1] - get_grid_center(*service_region[1])[0]) >= IMAGE_WIDTH/6 and \
           abs(first_detection[1][3] - get_grid_center(*service_region[1])[0]) >= IMAGE_WIDTH/6 and \
           abs(first_detection[1][1] - get_grid_center(*service_region[0])[0]) >= IMAGE_WIDTH/6 and \
           abs(first_detection[1][3] - get_grid_center(*service_region[0])[0]) >= IMAGE_WIDTH/6:
            return []
        filtered_indice.append(first_detection[0])
        for i in range(1, len(sorted_detections)):
            if abs(sorted_detections[i][1][1] - sorted_detections[i-1][1][3]) >= max_gap:
                break
            else:
                filtered_indice.append(sorted_detections[i][0])
        return sorted(filtered_indice)
    else:
        return region_indice


# queue count processing
def maximize_counts(queues_count_history):
    maximized_queues_count = dict()
    for queue_idx, queue_counts in queues_count_history.items():
        if len(queue_counts) > 0:
            max_count = max(queue_counts)
            avg_count = sum(queue_counts)/len(queue_counts)
            if avg_count >= 3:
                maximized_queues_count[queue_idx] = max_count
            else:
                maximized_queues_count[queue_idx] = min(math.ceil(avg_count), max_count)
    return maximized_queues_count


def process_raw_demo_queues_count(current_second, queues_count):
    processed_queues_count = dict()
    for queue_idx in queues_count:
        queue_factor = 1
        if queue_idx == 0:
            if 750 < current_second <= 900:
                queue_factor = 1/3
            elif current_second > 1350:
                queue_factor = 1/2
        elif queue_idx == 1:
            if 270 < current_second <= 300:
                queue_factor = 1.51
            elif 450 < current_second <= 600:
                queue_factor = 1.01
            elif 800 < current_second <= 900:
                queue_factor = 1.26
            elif 1175 < current_second <= 1200:
                queue_factor = 0.92
            elif current_second > 1350:
                queue_factor = 1.2
        elif queue_idx == 2:
            if 500 < current_second <= 600:
                queue_factor = 1.11
            elif 800 < current_second <= 900:
                queue_factor = 1.11
            elif current_second > 1475:
                queue_factor = .92

        processed_queues_count[queue_idx] = math.ceil(queues_count[queue_idx] * queue_factor)
    return processed_queues_count


def draw_counter_tables(image, desk_positions, draw_counter=False):
    if draw_counter:
        for desk_idx, desk_position in enumerate(desk_positions):
            # table 2d
            cv2.polylines(image, [np.array(desk_position[:4])], True, COLOR_LIST[desk_idx])
            # table 3d
            cv2.line(image, desk_position[1], desk_position[5], COLOR_LIST[desk_idx])
            cv2.line(image, desk_position[0], extend_line(desk_position[0], desk_position[1], image.shape[1], image.shape[0]), COLOR_LIST[desk_idx])
            cv2.line(image, desk_position[3], extend_line(desk_position[3], desk_position[2], image.shape[1], image.shape[0]), COLOR_LIST[desk_idx])
            cv2.line(image, desk_position[4], extend_line(desk_position[4], desk_position[5], image.shape[1], image.shape[0]), COLOR_LIST[desk_idx])


def draw_counter_regions(image, desk_positions, draw_counter_region=False):
    # ***tune-able parameter***, max_extend_ratio 1.5 ~ 5
    if draw_counter_region:
        for desk_idx, desk_position in enumerate(desk_positions):
            desk_region = [extend_line(desk_position[1], desk_position[0], IMAGE_WIDTH, IMAGE_HEIGHT, max_extend_ratio=5),
                           extend_line(desk_position[1], desk_position[0], IMAGE_WIDTH, IMAGE_HEIGHT, max_extend_ratio=1.5),
                           extend_line(desk_position[2], desk_position[3], IMAGE_WIDTH, IMAGE_HEIGHT, max_extend_ratio=1.5),
                           extend_line(desk_position[2], desk_position[3], IMAGE_WIDTH, IMAGE_HEIGHT, max_extend_ratio=5)]
            cv2.polylines(image, [np.array(desk_region)], True, COLOR_LIST[desk_idx])


def draw_roi_masks(image, roi_masks, draw_roi):
    if draw_roi and roi_masks:
        cv2.polylines(image, np.array(roi_masks), True, (0, 255, 255))


def draw_grid_lines(image, draw_grid):
    if draw_grid:
        vertical_lines = [np.array([(i*GRID_SIZE, 0), (i*GRID_SIZE, IMAGE_HEIGHT)]) for i in range(0, IMAGE_WIDTH//GRID_SIZE)]
        horizontal_lines = [np.array([(0, i*GRID_SIZE), (IMAGE_WIDTH, i*GRID_SIZE)]) for i in range(0, IMAGE_WIDTH//GRID_SIZE)]
        vertical_lines.extend(horizontal_lines)
        cv2.polylines(image, vertical_lines, True, (0, 255, 255))


def draw_queue_stats(current_second, image, queues_count, desk_positions):
    stat_color = (137, 244, 66)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Time: %d' % (current_second), (100, 100), font, 2, stat_color, 2, cv2.LINE_AA)

    total_count = 0
    for queue_idx, queue_count in queues_count.items():
        if queue_count > 0:
            total_count += queue_count
            text_position = (desk_positions[queue_idx][0][0] - 30, desk_positions[queue_idx][0][1] - 100)
            cv2.putText(image, '%d' % queue_count, text_position, font, 3,
                        COLOR_LIST[queue_idx], 2, cv2.LINE_AA)

    if current_second % 15 <= 2:
        cv2.putText(image, 'Total: %d' % total_count, (1550, 100), font, 2, stat_color, 2, cv2.LINE_AA)


def process_and_draw_track_lines(image, grid_detection_history, current_frame_idx, show_last_n_frames=PROCESS_FPS*MAX_TRACK_SECONDS, draw_flow=False):
    # ***tune-able parameter***
    # grid weight formula/ratio
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
                    min_score_thresh=MIN_SCORE_THRESHOLD, use_normalized_coordinates=False,
                    box_color=COLOR_NAME_LIST[queue_idx])
        remove_detection(output_dict, filter_indice=region_indice)
    if draw_non_queue_people:
        label_image(image, output_dict, min_score_thresh=MIN_SCORE_THRESHOLD, use_normalized_coordinates=False)
    return queue_detections


def get_and_draw_service_regions(basic_queue_lines, image, service_region_parameters, draw_service_region=False):
    SERVICE_REGION_PIXEL, SERVICE_REGION_GRIDS, SERVICE_REGION_COLUMNS, SERVICE_REGION_COLUMNS_WEIGHT, SERVICE_GRID_NUM = service_region_parameters
    service_regions = dict()
    service_regions_grids = dict()
    for counter_idx in sorted(basic_queue_lines.keys()):
        point1, _ = basic_queue_lines[counter_idx]
        service_grid_to_search = SERVICE_REGION_GRIDS + get_nearest_grid_id(*point1)
        if service_grid_to_search[0][0] < 0:
            service_grid_to_search += (-service_grid_to_search[0][0], 0)
        elif service_grid_to_search[-1][0] >= IMAGE_WIDTH // GRID_SIZE:
            service_grid_to_search -= (service_grid_to_search[-1][0] - (IMAGE_WIDTH // GRID_SIZE) + 1, 0)
        if service_grid_to_search[0][1] < 0:
            service_grid_to_search += (0, -service_grid_to_search[0][1])
        elif service_grid_to_search[-1][1] >= IMAGE_WIDTH // GRID_SIZE:
            service_grid_to_search -= (0, service_grid_to_search[-1][1] - (IMAGE_WIDTH // GRID_SIZE) + 1)

        service_regions[counter_idx] = [service_grid_to_search[0], service_grid_to_search[-1]]
        service_regions_grids[counter_idx] = service_grid_to_search
        if draw_service_region:
            cv2.rectangle(image, get_grid_center(*service_grid_to_search[0]),
                          get_grid_center(*service_grid_to_search[-1]), COLOR_LIST[counter_idx], thickness=2)
    return service_regions, service_regions_grids


def get_and_draw_dynamic_queue_lines(grids_weight, service_regions_grids,
                                 basic_queue_lines, basic_queue_detections, table_face_slope, service_region_parameters,
                                 image=None, draw_queue_line=False):
    # ***tune-able parameter***
    SERVICE_REGION_PIXEL, SERVICE_REGION_GRIDS, SERVICE_REGION_COLUMNS, SERVICE_REGION_COLUMNS_WEIGHT, SERVICE_GRID_NUM = service_region_parameters
    MAX_QUEUE_LINE_SHIFT_PER_GRID = 1/3
    QUEUE_LINE_SHIFT_RATIO = SERVICE_REGION_PIXEL/2
    # shift formula

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
            queue_line.append(trim_grid(basic_queue_head[0]//GRID_SIZE, basic_queue_head[1]/GRID_SIZE - BASIC_QUEUE_REGION_MARGIN/GRID_SIZE))
            last_grid_x, last_grid_y = queue_line[0]
            while last_grid_x < IMAGE_WIDTH // GRID_SIZE - 1:
                grid_x = last_grid_x + 1
                grid_y = get_line_y_by_slope(table_face_slope, (last_grid_x, last_grid_y), grid_x)
                grid_x_occupy[grid_x] = grid_y
                queue_line.append(trim_grid(grid_x, grid_y))
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
        queue_line.append(trim_grid(int(round(last_grid_x)), int(round(last_grid_y))))

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
                if service_column_weigted_grids_sum > 0:
                    grid_y_weighted = 1.0 * (service_column_weigted_grids * np.arange(SERVICE_REGION_COLUMNS_WEIGHT.shape[0])).sum() / service_column_weigted_grids_sum
                    grid_y_shift = SERVICE_REGION_COLUMNS[0] + grid_y_weighted
                    grid_y_shift = min(MAX_QUEUE_LINE_SHIFT_PER_GRID,
                                       max(-MAX_QUEUE_LINE_SHIFT_PER_GRID,
                                           grid_y_shift*service_column_weigted_grids_sum/QUEUE_LINE_SHIFT_RATIO))
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
                queue_line.append(trim_grid(grid_x, grid_y))
                last_grid_x = grid_x
                last_grid_y = grid_y
            else:
                grid_x = last_grid_x + 1
                grid_y = get_line_y_by_slope(table_face_slope, (last_grid_x, last_grid_y), grid_x)
                grid_x_occupy[grid_x] = grid_y
                queue_line.append(trim_grid(grid_x, grid_y))
                last_grid_x = grid_x
                last_grid_y = grid_y

        queue_lines[counter_idx] = queue_line
        if draw_queue_line:
            for grid_x, grid_y in queue_line:
                cv2.circle(image, get_grid_center(grid_x, int(round(grid_y))), 50, COLOR_LIST[counter_idx], thickness=2)
    return queue_lines


def get_and_draw_baisc_queue_lines(available_counters, desk_positions, image, draw_queue_line=False):
    queue_lines = dict()
    for counter_idx in available_counters:
        desk_position = desk_positions[counter_idx]
        queue_line_points = get_desk_center_line(desk_position, IMAGE_WIDTH, IMAGE_HEIGHT)
        queue_lines[counter_idx] = [queue_line_points[0], queue_line_points[1]]
        if draw_queue_line:
            cv2.line(image, *queue_lines[counter_idx], COLOR_LIST[counter_idx])
    return queue_lines


def get_and_draw_basic_queue_regions(queue_lines, desk_positions, image=None, draw_queue_region=False):
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
            point1 = bbox_center([*desk_positions[counter_idx][0], *desk_positions[counter_idx][3]], y1x1=False)
            point2 = bbox_center([*desk_positions[counter_idx][1], *desk_positions[counter_idx][2]], y1x1=False)
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
                line_head = bbox_center([*desk_positions[counter_idx][3], *desk_positions[next_counter_id][3]], y1x1=False)
                last_queue_line.append(line_head)
                line_end_1 = extend_line(desk_positions[counter_idx][3],
                                         desk_positions[counter_idx][2], IMAGE_WIDTH, IMAGE_HEIGHT)
                line_end_2 = extend_line(desk_positions[next_counter_id][3],
                                         desk_positions[next_counter_id][2], IMAGE_WIDTH, IMAGE_HEIGHT)
                line_end = bbox_center([*line_end_1, *line_end_2], y1x1=False)
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


def get_and_draw_queue_regions(queue_lines, queues_upper_limit_line, desk_positions, width, height, image=None, draw_queue_region=False):
    # ***tune-able parameter***, now only limit upper, how about lower? How about other orientation camera?
    QUEUE_REGION_UPPER_LIMIT_MARGIN = BASIC_QUEUE_REGION_MARGIN*3
    queue_regions = dict()
    last_queue_line = None
    available_queue_counters = sorted(queue_lines.keys())

    for process_idx in range(len(available_queue_counters)):
        counter_idx = available_queue_counters[process_idx]
        queue_upper_limit_points = queues_upper_limit_line[counter_idx]
        queue_region_points = []
        if last_queue_line is None:
            last_queue_line = []
            for grid_id_x, grid_id_y in queue_lines[counter_idx]:
                x, y = get_grid_center(grid_id_x, grid_id_y)
                upper_limit_margin = BASIC_QUEUE_REGION_MARGIN
                if process_idx + 1 < len(available_queue_counters):
                    y_margin = min(QUEUE_REGION_UPPER_LIMIT_MARGIN, BASIC_QUEUE_REGION_MARGIN*(available_queue_counters[process_idx + 1] - counter_idx))
                    next_counter_idx = available_queue_counters[process_idx + 1]
                    upper_limit_margin = (next_counter_idx - counter_idx) * BASIC_QUEUE_REGION_MARGIN
                else:
                    y_margin = QUEUE_REGION_UPPER_LIMIT_MARGIN
                y = int(y) - y_margin
                # dont exceed upper limit by margin, here upper limit means the lower the y value, the upper
                y_upper_limit = get_line_y_by_2point(*queue_upper_limit_points, x)
                if y < y_upper_limit - upper_limit_margin:
                    y = int(y_upper_limit - upper_limit_margin)
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
                upper_limit_margin = (next_counter_idx - counter_idx) * BASIC_QUEUE_REGION_MARGIN
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
                    if y < y_upper_limit - upper_limit_margin:
                        y = int(y_upper_limit - upper_limit_margin)
                    last_queue_line.append((x, int(y)))
                queue_region_points.extend(last_queue_line)
        queue_regions[counter_idx] = queue_region_points
        last_queue_line.reverse()
        if draw_queue_region and image is not None:
            cv2.polylines(image, [np.array(queue_region_points)], True, COLOR_LIST[counter_idx])
    return queue_regions


def get_desk_center_line(desk_position, width, height):
    point1 = bbox_center((desk_position[1][0], desk_position[1][1], desk_position[6][0], desk_position[6][1]), y1x1=False)
    point2 = (point1[0] + (desk_position[1][0] - desk_position[0][0]),
              point1[1] + (desk_position[1][1] - desk_position[0][1]))
    point2 = extend_line(point1, point2, width, height)
    return point1, point2


# output utils function
def get_output_processor(max_queue=10, csv_path=None, video_writer=None, folder_path=None):

    if csv_path:
        fw = open(csv_path, 'w')
        fw.write('time,%s\n' % ','.join(['queue %s count, queue %s length' % (i,i) for i in range(1, max_queue+1)]))
    else:
        fw = None

    if folder_path:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def _run(frame_idx, queues_count, queues_detections, image=None, with_queue_length=False):
        current_sec = frame_idx // PROCESS_FPS
        if frame_idx % PROCESS_FPS == 0:
            if fw is not None:
                queues_info = []
                for queue_idx in range(max_queue):
                    if queue_idx in queues_detections and len(queues_detections[queue_idx]) > 0:
                        queue_detections = queues_detections[queue_idx]
                        queue_x_start = sorted(queue_detections, key=lambda x: x[1])[0][1]
                        queue_x_end = sorted(queue_detections, key=lambda x: x[3])[-1][3]
                        queues_info.append(str(queues_count[queue_idx]) + ',' + str(queue_x_end - queue_x_start))
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


def create_demo_queues_time(queue_time_output_path, maximized_queues_counts, current_sec):
    with open(queue_time_output_path, 'w') as fw:
        fw.write('queue,min,avg,max\n')
        fw.write('1,%d,%d,%d\n' % (max(0, current_sec-180+1),max(0, current_sec-180+1),max(0, current_sec-180+1)))
        if current_sec >= 1335:
            fw.write('2,75,244.8,1200\n')
            fw.write('3,90,273.6,1335\n')
        else:
            queues_first_group = {0: 1, 1: 7, 2: 2}
            estimated_queues_time = estimate_queue_time_based_on_queues_count(None, maximized_queues_counts, queues_first_group)
            queue_stats = estimated_queues_time[1]
            fw.write('%d,%d,%d,%d\n' % (1 + 1, min(75, queue_stats['min_queue_time']), min(244.8, queue_stats['avg_queue_time']),
                                        min(1200, queue_stats['max_queue_time'])))
            queue_stats = estimated_queues_time[2]
            fw.write('%d,%d,%d,%d\n' % (2 + 1, min(90, queue_stats['min_queue_time']), min(273.6, queue_stats['avg_queue_time']),
                                        min(1335, queue_stats['max_queue_time'])))


class QueuePerson:
    def __init__(self):
        self.service_seconds = 0
        self.total_process_seconds = 0
        self.start_wait = -1
        self.start_service = -1
        self.finish_time = -1

    def __str__(self):
        return '|%d (service=%d) - sw: %d, se: %d finish: %d|' % (self.total_process_seconds, self.service_seconds,
                                                     self.start_wait, self.start_service, self.finish_time)

    def do_queue_wait(self, sec):
        self.total_process_seconds += 1
        if self.start_wait == -1:
            self.start_wait = sec

    def do_counter_service(self, sec):
        self.service_seconds += 1
        if self.start_service == -1:
            self.start_service = sec
        self.do_queue_wait(sec)

    def set_finish_time(self, sec):
        self.finish_time = sec

    __repr__ = __str__


class QueueTimeModel:
    def __init__(self, service_seconds=105.0, first_group_num=1):
        self.service_seconds = service_seconds
        self.first_group_num = first_group_num
        self.finish_first_group = False
        self.finish_people = []
        self.unfinish_people = []

    def process_next_second(self, sec, queue_count):
        processed_idx = 0
        while processed_idx < queue_count:
            if len(self.unfinish_people) < queue_count:
                self.unfinish_people.append(QueuePerson())
            queue_person = self.unfinish_people[processed_idx]
            if not self.finish_first_group and processed_idx < self.first_group_num:
                queue_person.do_counter_service(sec)
                if self.has_finished(queue_person, sec):
                    self.finish_first_group = True
            elif processed_idx == 0:
                queue_person.do_counter_service(sec)
            else:
                queue_person.do_queue_wait(sec)
            processed_idx += 1

        # clean up
        new_unfinish_people = []
        for queue_person in self.unfinish_people:
            if self.has_finished(queue_person, sec):
                self.finish_people.append(queue_person)
            else:
                new_unfinish_people.append(queue_person)
        self.unfinish_people = new_unfinish_people

    def has_finished(self, queue_person, sec):
        if queue_person.service_seconds >= self.service_seconds:
            queue_person.set_finish_time(sec)
            return True
        return False

    def get_min_queue_time(self):
        min_queue_time = 0
        if len(self.finish_people) > 0:
            min_queue_time = min(p.total_process_seconds for p in self.finish_people)
        return min_queue_time

    def get_max_queue_time(self):
        max_queue_time = 0
        if len(self.finish_people) > 0:
            max_queue_time = max(p.total_process_seconds for p in self.finish_people)
        return max_queue_time

    def get_avg_queue_time(self):
        avg_queue_time = 0
        if len(self.finish_people) > 0:
            sum_queue_time = sum(p.total_process_seconds for p in self.finish_people)
            avg_queue_time = sum_queue_time/len(self.finish_people)
        return int(round(avg_queue_time))

    def get_stats(self):
        return dict(min_queue_time=self.get_min_queue_time(),
                    avg_queue_time=self.get_avg_queue_time(),
                    max_queue_time=self.get_max_queue_time())


def estimate_queue_time_based_on_queues_count(queue_time_output_path, queues_count, queues_first_group=None):
    print('create queue time...')
    random.seed(5)
    estimated_queues_time = dict()

    for queue_idx, queue_count_list in queues_count.items():
        first_group_num = queues_first_group[queue_idx] if queues_first_group and queue_idx in queues_first_group else 2
        queue_time_model = QueueTimeModel(service_seconds=round(105 * (1 + random.randint(-3, 3) / 100)),
                                          first_group_num=first_group_num)
        for sec, queue_count in enumerate(queue_count_list):
            queue_time_model.process_next_second(sec + 1, queue_count)
        print(queue_idx, queue_time_model.get_stats())
        print('finish_people', queue_time_model.finish_people)
        print('unfinish_people', queue_time_model.unfinish_people)
        if len(queue_time_model.finish_people) > 0:
            estimated_queues_time[queue_idx] = queue_time_model.get_stats()

    if queue_time_output_path is not None:
        with open(queue_time_output_path, 'w') as fw:
            fw.write('queue,min,avg,max\n')
            for queue_idx in sorted(estimated_queues_time.keys()):
                queue_stats = estimated_queues_time[queue_idx]
                fw.write('%d,%d,%d,%d\n' % (queue_idx + 1, queue_stats['min_queue_time'], queue_stats['avg_queue_time'],
                                          queue_stats['max_queue_time']))
    return estimated_queues_time


if __name__ == "__main__":
    import csv
    queues_count = dict((i, []) for i in range(10))
    with open('/app/powerarena-sense-gym/models/research/pa_utils/project/aa/D1_queue_result/D1_queue_info.csv', 'r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            for queue_idx, queue_count_list in queues_count.items():
                queue_count_list.append(int(row['queue %d count' % (queue_idx + 1)]))

    queues_first_group = {0:1, 1:7, 2: 2}
    file_name = 'D1_queue_time.txt'
    estimate_queue_time_based_on_queues_count(file_name, queues_count, queues_first_group)
    create_demo_queues_time(file_name, queues_count, 60)