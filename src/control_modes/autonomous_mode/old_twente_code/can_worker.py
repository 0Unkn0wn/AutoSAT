import threading
import os
import struct
from queue import Queue
import can

class CanWorker:
    """
    A worker that writes can-message values to disk.
    """

    def __init__(self, can_queue: Queue, folder: str):
        self.queue = can_queue
        self.thread = threading.Thread(target=self._process, args=(), daemon=True)
        self.folder_name = folder
        self.file_pointer = open(os.path.join(self.folder_name, 'recording.csv'), 'w')
        print('Timestamp|Steering|SteeringSpeed|Throttle|Brake|SpeedSensor|SteeringSensor', file=self.file_pointer)

    def start(self):
        self.thread.start()

    def stop(self):
        self.queue.join()
        self.file_pointer.close()

    def put(self, data):
        self.queue.put(data)

    def _process(self):
        while True:
            timestamp, values = self.queue.get()
            steering = str(struct.unpack("f", bytearray(values["steering"][:4]))[0]) if values["steering"] else ""
            steering_speed = str(struct.unpack(">I", bytearray(values["steering"][4:]))[0]) if values["steering"] else ""
            throttle = str(values["throttle"][0] / 100) if values["throttle"] else ""
            brake = str(values["brake"][0] / 100) if values["brake"] else ""
            speed_sensor = str(struct.unpack(">H", bytearray(values["speed_sensor"][:2]))[0]) if values["speed_sensor"] else ""
            if values["steering_sensor"]:
                steering_sensor = (values["steering_sensor"][1] << 8 | values["steering_sensor"][2])
                steering_sensor -= 65536 if steering_sensor > 32767 else 0
            else:
                steering_sensor = ""
            print(f'{timestamp}|{steering}|{steering_speed}|{throttle}|{brake}|{speed_sensor}|{steering_sensor}', file=self.file_pointer)
            self.queue.task_done()

def initialize_can() -> can.Bus:
    """
    Set up the can bus interface and apply filters for the messages we're interested in.
    """
    bus = can.Bus(interface='virtual', channel='vcan0', bitrate=500000)
    bus.set_filters([
        {'can_id': 0x110, 'can_mask': 0xfff, 'extended': False}, # Brake
        {'can_id': 0x220, 'can_mask': 0xfff, 'extended': False}, # Steering
        {'can_id': 0x330, 'can_mask': 0xfff, 'extended': False}, # Throttle
        {'can_id': 0x440, 'can_mask': 0xfff, 'extended': False}, # Speed sensor
        {'can_id': 0x1e5, 'can_mask': 0xfff, 'extended': False}, # Steering sensor
    ])
    return bus