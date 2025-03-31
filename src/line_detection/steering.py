import numpy as np

class SteeringController:
    def __init__(self, width=848):
        self.width = width
    
    def calculate_steering_angle(self, target, mid_point=None):
        if target is None:
            return 0.0
            
        if mid_point is None:
            mid_point = self.width / 2
            
        offset = target - mid_point
        max_angle = 30.0
        steering_angle = (offset / (self.width / 2)) * max_angle
        return max(min(steering_angle, max_angle), -max_angle)
    
    def calculate_speed(self, steering_angle, base_speed=100):
        angle_factor = 1.0 - (abs(steering_angle) / 30.0) * 0.5
        return max(base_speed * angle_factor, base_speed * 0.5)
