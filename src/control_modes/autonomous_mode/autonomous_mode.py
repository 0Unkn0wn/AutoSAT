# src/control_modes/autonomous_mode/AutonomousMode.py
import cv2
import asyncio
from .line_detection.LineDetection import LineFollowingNavigation
from .object_detection.ObjectDetection import ObjectDetection
from ...car_variables import CarType, HunterControlMode, KartGearBox
from ...control_modes.IControlMode import IControlMode
from ...control_modes.autonomous_mode.object_detection.TrafficDetection import TrafficManager
from ...can_interface.bus_connection import connect_to_can_interface
from ...can_interface.can_controller import CarCanController

class AutonomousMode(IControlMode):
    def __init__(self, car_type: CarType):
        self.car_type = car_type
        self.can_controller = None
        self.nav = LineFollowingNavigation(width=848, height=480)
        self.object_detector = ObjectDetection(weights_path='v5_model.pt', input_source='video')
        self.traffic_manager = TrafficManager()
        self.can_bus = connect_to_can_interface(0)

    async def start(self):
        self.can_controller = CarCanController(self.can_bus, self.car_type)
        await self.can_controller.send_control(0, True, HunterControlMode.command_mode)
        cap = cv2.VideoCapture('../../../assets/trash.mp4')
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                steering_angle, speed, viz_img, end_x = self.nav.process(frame)
                traffic_state, detections = self.object_detector.process(frame)
                saw_red_light = traffic_state['red_light']
                speed_limit = traffic_state['speed_limit']
                print(f"Traffic State: {traffic_state}, Saw Red Light: {saw_red_light}, Speed Limit: {speed_limit}")
                cv2.imshow("Line Following", viz_img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
    async def stop(self) -> None:
        if self.car_type == CarType.hunter:
            await self.can_controller.send_control(0, True, HunterControlMode.idle_mode)
        else:
            await self.can_controller.send_control(100, True, HunterControlMode.idle_mode)
            await self.can_controller.send_movement(0, KartGearBox.neutral, 0)
        self.can_bus.shutdown()
        print("Stopping autonomous mode.")