import threading
from typing import Optional, Dict, List
import can

class CanListener:
    """
    A can listener that listens for specific messages and stores their latest values.
    """
    _id_conversion = {
        0x110: 'brake',
        0x220: 'steering',
        0x330: 'throttle',
        0x440: 'speed_sensor',
        0x1e5: 'steering_sensor'
    }

    def __init__(self, bus: can.Bus):
        self.bus = bus
        self.thread = threading.Thread(target=self._listen, args=(), daemon=True)
        self.running = False
        self.data: Dict[str, List[int]] = {name: None for name in self._id_conversion.values()}

    def start_listening(self):
        self.running = True
        self.thread.start()

    def stop_listening(self):
        self.running = False

    def get_new_values(self):
        return self.data

    def _listen(self):
        while self.running:
            message: Optional[can.Message] = self.bus.recv(0.5)
            if message and message.arbitration_id in self._id_conversion:
                self.data[self._id_conversion[message.arbitration_id]] = message.data