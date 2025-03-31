import cv2
import numpy as np
from line_detection import LineDetector
from steering import SteeringController

class NavigationSystem:
    def __init__(self, width=848, height=480, scale=1):
        self.line_detector = LineDetector(width, height, scale)
        self.steering_controller = SteeringController(width)
        self.width = width
        self.height = height
        self.scale = scale
    
    def process_frame(self, frame, draw=True):
        edges = self.line_detector.get_lines(frame)
        target = self.find_target(edges, frame, draw=draw)
        steering_angle = self.steering_controller.calculate_steering_angle(target)
        speed = self.steering_controller.calculate_speed(steering_angle)
        return steering_angle, speed, frame if draw else None
    
    def find_target(self, edges, frame, draw=True):
        height, width = frame.shape[:2]
        mid_x = width // 2
        target_x = mid_x  # Default to center if no lines detected
        
        if edges is not None:
            points = np.column_stack(np.where(edges > 0))
            if len(points) > 0:
                target_x = int(np.mean(points[:, 1]))
        
        if draw:
            cv2.circle(frame, (target_x, height // 2), 5, (0, 0, 255), -1)
        
        return target_x
