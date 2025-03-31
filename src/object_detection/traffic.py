class TrafficManager:
    def __init__(self):
        self.red_light_threshold = 4.5  # Distance threshold to consider traffic light as "red"
        self.speed_limit = 10

    def process_traffic_signals(self, detections):
        saw_red_light = False
        for det in detections:
            if det['class'] == 'Traffic Light Red' and det['distance'] < self.red_light_threshold:
                saw_red_light = True
            elif det['class'].startswith('Speed-limit'):
                speed = int(det['class'].split('-')[2].replace('km', ''))
                self.speed_limit = min(self.speed_limit, speed)
        return {'red_light': saw_red_light, 'speed_limit': self.speed_limit}
