from src.car_variables import CarType

def calculate_throttle(controller_axis_value: float, car_type: CarType) -> float | None:
    if car_type == CarType.kart:
        x = max(0.0, -controller_axis_value)
        return (x ** 3) * 100
    elif car_type == CarType.hunter:
        return -(controller_axis_value ** 3) * 1500

def calculate_steering(controller_axis_value: float, car_type: CarType) -> float | None:
    cubic = -(controller_axis_value ** 3)
    if car_type == CarType.kart:
        return cubic * 1.25
    elif car_type == CarType.hunter:
        return cubic * 576
