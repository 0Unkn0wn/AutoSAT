import collections
import math
import os
import time
import json
import csv
import torch
import psutil
from datetime import datetime
from collections import deque

from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from models.common import DetectMultiBackend
import cv2
import numpy as np
from constants import width, height, OBJECT_DOWNSAMPLE_FACTOR, OBJECT_FPS_CAP


def initialize(weights_path, output_dir_base):
    """
    Initialize the object detection model and related components.
    """
    # Base directory for saving outputs
    os.makedirs(output_dir_base, exist_ok=True)
    print("Output Directory Created")

    # Determine the next available directory for saving detections
    detections_dir = find_next_available_dir(output_dir_base, 'detections')
    os.makedirs(detections_dir, exist_ok=True)
    print(f"Detections Directory Created: {detections_dir}")

    # Load model once
    device = select_device('CPU')
    model = load_model(weights_path, device)

    quantized_model = quantize_model(model)
    print("Done optimizing model")

    # Check if GUI is available
    gui_available = True

    return quantized_model, device, gui_available


def traffic_object_detection(frame_queue, state_queue, model, device, stop_event):
    """
    This is the main thread of object detection, it initializes the memory then it has a frame
    queue and a state queue, it constantly takes frames from the frames queue and then appends resulting states
    in the states queue
    """
    prev_frame_time = 0

    # Set distance threshold for red light/traffic sign detection
    red_light_distance_threshold = 4.5  # meters
    speed_sign_distance_threshold = 10  # meters
    person_distance_threshold = 10 # meters
    car_distance_threshold = 10 # meters

    # Memory buffers for red lights and speed signs
    red_light_memory = collections.deque(maxlen=5)
    speed_sign_memory = collections.deque(maxlen=5)
    person_memory = collections.deque(maxlen=10)
    car_memory = collections.deque(maxlen=5)

    saw_red_light = False
    last_speed_limit = 10 #  this sets the initial speed of the car

    initial_person_position = "None"
    current_person_position = "None"

    # Car
    car_spotted = False

    # Data Logger
    # Define the headers for the CSV file
    headers = ["detections", "detect_processing_time", "traffic_processing_time",
               "year", "month", "day", "hour", "minute", "second", "state", "cpu_usage"]


    # Create a new folder named with the current date and time
    now = datetime.now()

    log_root = "logs"
    os.makedirs(log_root, exist_ok=True)
    folder_name = now.strftime("%m_%d_%H")
    folder_path = os.path.join(log_root, folder_name)
    os.makedirs(folder_path, exist_ok=True)


    # Create a new CSV file within the new folder, also named with the current date and time
    file_name = now.strftime("%Y_%m_%d_%H_%M_%S") + "_detection_log.csv"
    file_path = os.path.join(folder_path, file_name)

    # Define the headers for the CSV file
    headers = ["detections", "detect_processing_time", "traffic_processing_time",
               "year", "month", "day", "hour", "minute", "second", "state", "cpu_usage"]

    # Create the CSV file and write the headers
    with open(file_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    print("Data logger Object detection ready")

    while True:
        if frame_queue:
            # Get the newest frame from the queue
            frame = frame_queue.pop()
            if frame is not None:
                # Record the start time for processing
                start_time = time.time()

                # Process the frame and create a new state
                dets = process_single_image(model, device, frame)

                # Visualize detections
                if dets:
                    for det in dets:
                        x1, y1, x2, y2 = map(int, det['bbox'])
                        label = f"{det['class']} ({det['distance']:.1f}m)"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # FPS overlay
                new_frame_time = time.time()
                fps = 1 / (new_frame_time - prev_frame_time + 1e-8)
                prev_frame_time = new_frame_time
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Show frame
                cv2.imshow("Live Detection Feed", frame)
                cv2.waitKey(1)

                # Record the end time for processing and calculate the duration
                end_time = time.time()
                detect_processing_time = end_time - start_time

                # Local loop variables for checking closest traffic lights and signs
                # Speed signs
                bool_saw_speed_sign = False
                closest_seen_speed_limit = 0
                closest_speed_sign_distance = np.inf

                # Traffic lights
                closest_traffic_light_distance = np.inf
                bool_saw_red_light = False

                #People
                closest_person_distance = np.inf

                temp_initial_person_position = "None"
                temp_current_person_position = "None"

                # Cars
                temp_car_spotted = False
                closest_car_distance = np.inf
                #TODO: Implement logic to detect cars
                if dets:
                    for det in dets:
                        # First check for traffic lights
                        if det['class'] == 'Traffic Light Red':
                            closest_traffic_light_distance = min(closest_traffic_light_distance, det['distance'])
                            bool_saw_red_light = True

                        elif det['class'] in ['Speed-limit-10km-h', 'Speed-limit-15km-h', 'Speed-limit-20km-h']:
                            if det['distance'] < closest_speed_sign_distance:
                                closest_speed_sign_distance = det['distance']
                                closest_seen_speed_limit = int(det['class'].split('-')[2].replace('km', ''))
                                bool_saw_speed_sign = True

                        elif det['class'] == 'Person':
                            x_center = (det['bbox'][0] + det['bbox'][2])/2
                            middle_bound_left = 0.40
                            middle_bound_right = 0.60
                            closest_person_distance = min(closest_person_distance, det['distance'])
                            #Initialize person position
                            if initial_person_position == "None":
                                temp_initial_person_position = "Right" if x_center > width // 2 else "Left"
                                temp_current_person_position = "Right" if x_center > width // 2 else "Left"

                            #Update person position if it was right
                            elif initial_person_position == "Right":
                                if (x_center > width * middle_bound_left and  x_center < width * middle_bound_right):
                                    temp_current_person_position = "Middle"
                                elif x_center < width * middle_bound_left:
                                    temp_current_person_position = "Left"
                                else:
                                    temp_current_person_position = "Right"

                            #Update person position if it was left
                            elif initial_person_position == "Left":
                                if (x_center > width * middle_bound_left and  x_center < width * middle_bound_right):
                                    temp_current_person_position = "Middle"
                                elif x_center > width * middle_bound_right:
                                    temp_current_person_position = "Right"
                                else:
                                    temp_current_person_position = "Left"
                        elif det['class'] =='Car':
                            closest_car_distance = min(closest_car_distance, det['distance'])
                            temp_car_spotted = True



                # Update memory buffers
                red_light_memory.append(bool_saw_red_light and closest_traffic_light_distance < red_light_distance_threshold)
                speed_sign_memory.append((bool_saw_speed_sign, closest_seen_speed_limit) if closest_speed_sign_distance < speed_sign_distance_threshold else (False, 0))
                person_memory.append((closest_person_distance < person_distance_threshold, temp_current_person_position) if closest_person_distance < person_distance_threshold else (False, "None"))
                car_memory.append(temp_car_spotted and closest_car_distance < car_distance_threshold)

                # Check the memory buffers
                saw_red_light = all(red_light_memory)
                # Update last speed limit if all detected speed signs agree
                if all(sign[0] for sign in speed_sign_memory) and len(set(sign[1] for sign in speed_sign_memory)) == 1:
                    last_speed_limit = closest_seen_speed_limit

                # Update person positions if all detections agree
                if all(person[0] for person in person_memory):
                    positions = [person[1] for person in person_memory]
                    if len(set(positions)) == 1:
                        current_person_position = positions[0]
                        if initial_person_position == "None":
                            initial_person_position = positions[0]

                # Check if person memory is all false
                if not any(person[0] for person in person_memory):
                    initial_person_position = "None"
                    current_person_position = "None"

                # Memory for car spotting
                car_spotted = all(car_memory)

                new_state = {
                    "spotted_red_light": saw_red_light,
                    "Speed limit": last_speed_limit,
                    "Initial Person Position": initial_person_position,
                    "Current Person Position": current_person_position,
                    "Car Spotted": car_spotted
                }


                state_queue.append(new_state)
                #print(f"Updated shared_state: {new_state}")
                end_time = time.time()

                traffic_detect_processing_time = end_time - start_time

                # Measure CPU usage
                cpu_usage = psutil.cpu_percent(interval=None)

                # Extract the current timestamp and break it into components
                now = datetime.now()
                year, month, day = now.year, now.month, now.day
                hour, minute, second = now.hour, now.minute, now.second


                # Prepare the data for
                detection_info = {
                    "detections": dets,
                    "detect_processing_time": detect_processing_time,
                    "traffic_processing_time": traffic_detect_processing_time,
                    "year": year,
                    "month": month,
                    "day": day,
                    "hour": hour,
                    "minute": minute,
                    "second": second,
                    "state": new_state,
                    "cpu_usage": cpu_usage
                }

                with open(file_path, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        detection_info["detections"],
                        detection_info["detect_processing_time"],
                        detection_info["traffic_processing_time"],
                        detection_info["year"],
                        detection_info["month"],
                        detection_info["day"],
                        detection_info["hour"],
                        detection_info["minute"],
                        detection_info["second"],
                        detection_info["state"],
                        detection_info["cpu_usage"]
                    ])
                # If object detection is running too fast, cap the FPS
                if traffic_detect_processing_time < 1 / OBJECT_FPS_CAP:
                    time.sleep(1 / OBJECT_FPS_CAP - traffic_detect_processing_time)
    print("Object detection stopped...(got killed)")


def adjust_throttle(state_queue, throttle_queue, max_car_speed=20):
    """
    Adjust the throttle based on detected objects and the current state.
    """
    already_found_car = False
    kill_object_detection = False
    while True:
        if state_queue:
            # Get the newest state from the queue
            new_state = state_queue.pop()

            # Adjust the throttle based on the shared state
            throttle_speed = calculate_throttle_based_on_state(new_state, max_car_speed)

            # Update the throttle queue with the most recent throttle speed
            car_in_range = False
            if new_state['Car Spotted']:
                car_in_range = True
                already_found_car = True
            if (new_state["Speed limit"] == 15) and already_found_car:
                kill_object_detection = True
            if len(throttle_queue) >= 1:
                throttle_queue.pop()
            throttle_state = {
                "throttle": throttle_speed,
                "car in range": car_in_range,
                "kill object detection": kill_object_detection
            }

            throttle_queue.append(throttle_state)

            # Log throttle information
            throttle_info = {
                "speed": new_state["Speed limit"],
                "throttle": throttle_speed,
                "saw_red_light": new_state["spotted_red_light"],
                "timestamp": datetime.now().isoformat()
            }
            with open(os.path.join("logs", "throttle_log.json"), "a") as f:
                f.write(json.dumps(throttle_info) + "\n")

        time.sleep(0.03)


def calculate_throttle_based_on_state(state, max_car_speed=20):
    """
    Calculate the throttle value based on the current state.
    """
    if state["spotted_red_light"]:
        return 0  # Stop if red light is spotted
    elif state["Car Spotted"]:
        return 10  # Slow down if a car is spotted
    else:
        car_speed_km_h = min(state["Speed limit"], max_car_speed)
        if car_speed_km_h == 15:
            car_speed_km_h = 10
        return int(car_speed_km_h / max_car_speed * 100)  # Throttle as a percentage of max speed


def load_model(weights, device):
    """
    Load a pretrained YOLO model.
    """
    model = DetectMultiBackend(weights, device=device, fp16=False)
    model.warmup()
    return model


def quantize_model(model):
    """
    Quantize the model to reduce computation load.
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

def run_inference(model, frame, device, stride=32):
    """
    This function takes as input a frame and the model and gives back the predictions in yolo format
    """
    # Resize the image to ensure its dimensions are multiples of the model's stride
    img_size = check_img_size(frame.shape[:2], s=stride)  # Ensure multiple of stride
    img = cv2.resize(frame, (img_size[1], img_size[0]))

    # Convert image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))  # Convert to [3, height, width]
    img = np.expand_dims(img, axis=0)  # Add batch dimension [1, 3, height, width]
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # Normalize to 0.0 - 1.0

    with torch.no_grad():
        pred = model(img)
    return pred, img.shape

def estimate_distance(x1, y1, x2, y2, real_width, real_height, x_offset=424, y_offset=240, image_width=848, image_height=480,fov_based = False):
    """
    estimate_distance:
    takes the box corner positions, the objects real width and height and other image variables to calculate the distance from the center of the
    car to the object
    """
    # Constants for Logitech StreamCam
    camera_fov_h = 67.5  # Horizontal field of view in degrees
    camera_fov_v = 41.2  # Vertical field of view in degrees
    focal_length = 540   # focal length of the camera
    box_width = abs(x2 - x1)
    box_height = abs(y2 - y1)

    #Standard distance offset (distance from front of car to camera)
    camera_offset  = 0.3
    #Assure there is no division by 0
    if box_height <= 0:
        box_height = 1
    if box_width <=0:
        box_width = 1
    #FOV based calculations
    if fov_based:
        # Translate bounding box coordinates to full image coordinates
        x1 += x_offset
        y1 += y_offset
        x2 += x_offset
        y2 += y_offset

        # Calculate the center and dimensions of the bounding box
        box_x_center = (x1 + x2) / 2
        box_y_center = (y1 + y2) / 2


        # Calculate the horizontal and vertical angles relative to the center of the image
        angle_h = (box_x_center - image_width / 2) * (camera_fov_h / image_width)
        angle_v = (box_y_center - image_height / 2) * (camera_fov_v / image_height)

        # Calculate the estimated distances using width and height
        est_w_d = (real_width * focal_length) / box_width
        est_h_d = (real_height * focal_length) / box_height

        # Adjust the estimated distances based on the angles
        adjusted_est_w_d = est_w_d / math.cos(math.radians(angle_h))
        adjusted_est_h_d = est_h_d / math.cos(math.radians(angle_v))

        # Average the distances if the entire object is within the FOV
        if box_width < box_height:
            distance = (adjusted_est_w_d + adjusted_est_h_d) / 2
        else:
            distance = adjusted_est_w_d

        return max(distance -camera_offset,0)
    #focal based calculations
    else:

        est_w_d = (focal_length * real_width)/box_width
        est_h_d = (focal_length * real_height)/box_height

        return max((est_w_d+ est_h_d)/2 - camera_offset,0)

def process_detections(pred, frame, img_shape, conf_thres=0.70, iou_thres=0.45, max_det=1000):
    """
    This functions takes the model predictions, the image and its filter parameters and gives back the detections
    classes
    """
    det = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)[0]
    detections = []

    if det is not None and len(det):
        det[:, :4] = scale_boxes(img_shape[2:], det[:, :4], frame.shape).round()
        for *xyxy, conf, cls in reversed(det):
            x1, y1, x2, y2 = map(int, xyxy)
            confidence = conf.item()
            class_id = int(cls.item())
            if class_id == 0:
                obj_class = 'Car'
            elif class_id == 1:
                obj_class = 'Person'
            elif class_id == 2:
                obj_class = 'Speed-limit-10km-h'
            elif class_id == 3:
                obj_class = 'Speed-limit-15km-h'
            elif class_id == 4:
                obj_class = 'Speed-limit-20km-h'
            elif class_id == 5:
                obj_class = 'Traffic Light Green'
            elif class_id == 6:
                obj_class = 'Traffic Light Red'
            else:
                obj_class = 'Unknown'

            detections.append({
                'class': obj_class,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })

    return detections


def process_single_image(model, device, frame):
    """
    This image takes a single frame and performs object detection on it and returns the detections with the class, bbox and distance
    """

    DOUBLE_SIDE = True
    TRIPLE_SIDE = False
    down_sample_factor = OBJECT_DOWNSAMPLE_FACTOR

    real_widths = {
        'Speed-limit-10km-h': 0.6,
        'Speed-limit-15km-h': 0.6,
        'Speed-limit-20km-h': 0.6,
        'Traffic Light Green': 0.07,
        'Traffic Light Red': 0.07,
        'Car': 1.7,
        'Person': 0.5,
        'Unknown': 0.5
    }
    real_heights = {
        'Speed-limit-10km-h': 0.6,
        'Speed-limit-15km-h': 0.6,
        'Speed-limit-20km-h': 0.6,
        'Traffic Light Green': 0.3,
        'Traffic Light Red': 0.3,
        'Car': 1.5,
        'Person': 2.0,
        'Unknown': 0.5
    }


    if frame is None:
        print(f"Failed to read the image at {frame}. Skipping.")
        return None



    def filter_edges(detection_info, x_value, p_treshold, width):
        max_x = min(x_value + (p_treshold*width),width)
        min_x = max(x_value - (p_treshold*width),0)

        for r_dets in detection_info:
            x1, y1, x2, y2 = r_dets['bbox']
            center_x = (x1+x2)/2
            if (center_x < max_x) and (center_x > min_x):
                detection_info.remove(r_dets)
        return detection_info

    def apply_offsets(detection_info, offset):
        for det in detection_info:
            x1, y1, x2, y2 = det['bbox']
            x1 = x1 * down_sample_factor
            y1 = y1 * down_sample_factor
            x2 = x2 * down_sample_factor
            y2 = y2 * down_sample_factor

            x1 += offset[1]
            y1 += offset[0]
            x2 += offset[1]
            y2 += offset[0]
            det['bbox'] = (x1, y1, x2, y2)
        return detection_info


    # Generate dictionary with detection details
    detection_info = []

    right_detection_info = []
    left_detection_info = []
    middle_detection_info = []

    # Crop the top right portion of the image
    top_right_frame = frame[:height // 2, width // 2:]


    # Ensure the cropped image is resized to the desired dimensions (multiples of model's stride)
    stride = 32
    original_height, original_width = top_right_frame.shape[:2]
    desired_height = (((original_height // stride) + 1) * stride)//down_sample_factor
    desired_width = (((original_width // stride) + 1) * stride) //down_sample_factor
    resized_top_right_frame = cv2.resize(top_right_frame, (desired_width, desired_height))

    # Process the resized cropped frame with YOLOv5 detection
    pred, img_shape = run_inference(model, resized_top_right_frame, device)
    detections = process_detections(pred, resized_top_right_frame, img_shape)





    # top right detections


    top_right_offset = (0, width //2)

    detections = apply_offsets(detections, top_right_offset)
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        #Compute the real widths and heights

        real_width_m = real_widths.get(det['class'], 0.5)
        real_height_m = real_heights.get(det['class'], 0.5)
        distance = estimate_distance(x1,y1,x2,y2,real_width_m,real_height_m)
        right_detection_info.append({
            'class': det['class'],
            'bbox': det['bbox'],
            'distance': distance
        })

    if DOUBLE_SIDE or TRIPLE_SIDE:

        # Top left detections

        # Crop the top left portion of the image
        top_left_frame = frame[:height // 2, :width // 2]
        #Resize
        resized_top_left_frame = cv2.resize(top_left_frame, (desired_width, desired_height))
        #Predict
        pred, img_shape = run_inference(model, resized_top_left_frame, device)
        #Detect
        detections = process_detections(pred, resized_top_left_frame, img_shape)

        #Offset
        top_left_offset = (0, 0)
        detections = apply_offsets(detections, top_left_offset)

        #Calculate Distance
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            real_width_m = real_widths.get(det['class'], 0.5)
            real_height_m = real_heights.get(det['class'], 0.5)
            distance = estimate_distance(x1,y1,x2,y2,real_width_m,real_height_m)
            #Store info
            left_detection_info.append({
                'class': det['class'],
                'bbox': det['bbox'],
                'distance': distance
            })

        #If I stop the PERSON or CAR class in the left or right image process the middle image
        if 'Person' in [det['class'] for det in left_detection_info] or 'Person' in [det['class'] for det in right_detection_info]:
            TRIPLE_SIDE = True
        elif 'Car' in [det['class'] for det in left_detection_info] or 'Car' in [det['class'] for det in right_detection_info]:
            TRIPLE_SIDE = True
        else:
            TRIPLE_SIDE = False

        if TRIPLE_SIDE:
            #Crop the top middle portion of the image
            top_middle_frame = frame[:height // 2, width // 4:3*width // 4]
            #Resize
            resized_top_middle_frame = cv2.resize(top_middle_frame, (desired_width, desired_height))
            #Predict
            pred, img_shape = run_inference(model, resized_top_middle_frame, device)
            #Detect
            detections = process_detections(pred, resized_top_middle_frame, img_shape)

            #Offset
            top_middle_offset = (0, width // 4)
            detections = apply_offsets(detections, top_middle_offset)

            #Calculate Distance
            for det in detections:
                x1, y1, x2, y2 = det['bbox']

                real_width_m = real_widths.get(det['class'], 0.5)
                real_height_m = real_heights.get(det['class'], 0.5)
                distance = estimate_distance(x1,y1,x2,y2,real_width_m,real_height_m)
                middle_detection_info.append({
                    'class': det['class'],
                    'bbox': det['bbox'],
                    'distance': distance
                })

            #Smooth out the 3 different detections (if right left and middle)
            # Remove any cut off detections and let the middle detection take over it
            right_detection_info = filter_edges(right_detection_info, width//2, 0.15, width)
            left_detection_info = filter_edges(left_detection_info, width//2, 0.15, width)

            #Smooth out edges of the middle detection
            #Filter out edges of middle part of the image
            middle_detection_info = filter_edges(middle_detection_info, (width*3//10), 0.05, width)
            middle_detection_info = filter_edges(middle_detection_info, (width*7//10), 0.05, width)

    detection_info = left_detection_info + right_detection_info + middle_detection_info

    return detection_info


def find_next_available_dir(base_path, base_name):
    """
    Find the next available directory name.
    """
    counter = 0
    while True:
        new_path = os.path.join(base_path, f"{base_name}_{counter}")
        if not os.path.exists(new_path):
            return new_path
        counter += 1