import logging

from ..IControlMode import IControlMode
from ...can_interface.bus_connection import connect_to_can_interface
from ...can_interface.can_controller import CarCanController
from ...car_variables import CarType, HunterControlMode, KartGearBox
from ...gamepad import Gamepad
from ...gamepad.controller_mapping import ControllerMapping
from ...misc import calculate_steering, calculate_throttle

class ManualMode(IControlMode):

    def __init__(self, car_type: CarType):
        self.can_bus = connect_to_can_interface(0)
        self.car_type = car_type
        self.can_controller = CarCanController(self.can_bus, self.car_type)

        if Gamepad.available():
            print("Connected to Gamepad")
            self.gamepad = Gamepad.XboxONE()
            self.gamepad.startBackgroundUpdates()
        else:
            print('Controller not connected :(')
            return

    async def car_input(self):
        if self.gamepad.beenPressed(ControllerMapping.buttonExit):
            print("exiting....")
            await self.stop()
            return None, None, None
        steering = calculate_steering(self.gamepad.axis(ControllerMapping.L_joystickX), self.car_type)
        throttle = calculate_throttle(self.gamepad.axis(ControllerMapping.R_joystickY), self.car_type)
        park = dead_man_switch(self.gamepad)
        return steering, throttle, park

    async def start(self) -> None:
        if self.car_type == CarType.hunter:
            await self.can_controller.send_control(0, True, HunterControlMode.command_mode)
            try:
                while self.gamepad.isConnected():
                    steering, throttle, park = await self.car_input()
                    if steering is not None and throttle is not None and park is not None:
                        await self.can_controller.send_control(0, park, HunterControlMode.command_mode)
                        await self.can_controller.send_movement(throttle, KartGearBox.neutral, steering)
                        message = await self.can_controller.monitor_bus()
                        print(f"{steering} \t {throttle} \t {message}")
                    else:
                        break
            finally:
                self.gamepad.disconnect()

        elif self.car_type == CarType.kart:
            try:
                while self.gamepad.isConnected():
                    steering, throttle, park = await self.car_input()
                    if steering is not None and throttle is not None and park is not None:
                        await self.can_controller.send_control(controller_break_value(self.gamepad), park, HunterControlMode.idle_mode)
                        await self.can_controller.send_movement(throttle, KartGearBox.forward, steering)
                        message = await self.can_controller.monitor_bus()
                        print(f"{steering} \t {throttle} \t {message}")
                    else:
                        break
            finally:
                self.gamepad.disconnect()


    async def stop(self) -> None:
        if self.car_type == CarType.hunter:
            await self.can_controller.send_control(0, True, HunterControlMode.idle_mode)
        else:
            await self.can_controller.send_control(100, True, HunterControlMode.idle_mode)
            await self.can_controller.send_movement(0, KartGearBox.neutral, 0)
        self.can_bus.shutdown()
        print("Stopping manual mode")

def dead_man_switch(gamepad: Gamepad) -> bool:
    if gamepad.axis(ControllerMapping.park) == 1:
        park = False
        logging.debug("Dead Man Switch pressed")
    else:
        park = True
        logging.debug("Dead Man Switch not pressed")
    return park

def controller_break_value(gamepad: Gamepad) -> int:
    return -(gamepad.axis(ControllerMapping.park)**3)*100