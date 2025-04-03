from enum import StrEnum

class ControllerMapping(StrEnum):
    buttonExit = 'B'
    L_joystickX = 'LAS -X'
    L_joystickY = 'LAS -Y'
    R_joystickX = 'RAS -X'
    R_joystickY = 'RAS -Y'
    throttle = 'LT'
    park = 'RT'