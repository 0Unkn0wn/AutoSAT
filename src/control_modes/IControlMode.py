from abc import ABC, abstractmethod

class IControlMode(ABC):

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass