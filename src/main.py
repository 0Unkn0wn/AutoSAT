import asyncio

import cv2

from src.gamepad import Gamepad
from src.misc import calc_axis_angle, calculate_hunter_throttle, calculate_hunter_steering
from src.gamepad.controller_mapping import ControllerMapping
from src.can_interface.can_controller import CarCanController, CarType, HunterControlMode
from src.can_interface.bus_connection import connect_to_can_interface
from car_variables import KartGearBox


from src.line_detection.LineDetection import LineFollowingNavigation
from src.object_detection.Detection import ObjectDetection
from src.object_detection.TrafficDetection import TrafficManager

def cameraLoop():

    return;

async def main() -> None:
    can_bus = connect_to_can_interface(0)
    can_controller = CarCanController(can_bus, CarType.hunter)

    if Gamepad.available():
        print("Connected to Gamepad")
        gamepad = Gamepad.Xbox360()
        gamepad.startBackgroundUpdates()
    else:
        print('Controller not connected :(')
        return

    await can_controller.send_control(0, True, HunterControlMode.command_mode)
    try:

        nav = LineFollowingNavigation(width=848, height=480)
        object_detector = ObjectDetection(weights_path='v5_model.pt', input_source='video')

        cap = cv2.VideoCapture("trash.mp4")  # Replace with your actual video file
        if not cap.isOpened():
            print("Error: Could not open video file.")
            exit()

        while gamepad.isConnected():
            if gamepad.beenPressed(ControllerMapping.buttonExit):
                print("exiting....")
                break
            steering = calculate_hunter_steering(gamepad.axis(ControllerMapping.L_joystickX), CarType.hunter)
            throttle = calculate_hunter_throttle(gamepad.axis(ControllerMapping.R_joystickY), CarType.hunter)
            if gamepad.axis(ControllerMapping.park) == 1:
                park = False
            else:
                park = True
            await can_controller.send_control(0, park, HunterControlMode.command_mode)
            await can_controller.send_movement(throttle, KartGearBox.neutral, steering)
            message = await can_controller.monitor_bus()

            print(f"{steering} \t {throttle} \t {message}")

            ret, frame = cap.read()
            if ret:
                steering_angle, speed, viz_img, end_x = nav.process(frame)
                traffic_state, detections = object_detector.process(frame)

                saw_red_light = traffic_state['red_light']
                speed_limit = traffic_state['speed_limit']
                print(f"Traffic State: {traffic_state}, Saw Red Light: {saw_red_light}, Speed Limit: {speed_limit}")
                cv2.imshow("Line Following", viz_img)



        # print(f"{steering} \t {throttle}")
    finally:
        gamepad.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
