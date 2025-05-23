from typing import Annotated
from dataclasses import dataclass
import numpy as np


@dataclass
class State:
    t: Annotated[np.longdouble, 'Время [sec]']
    latitude: Annotated[np.longdouble, 'Широта [deg]']
    longitude: Annotated[np.longdouble, 'Долгота [deg]']
    velocity_x_ref: Annotated[np.longdouble, 'Скорость по оси x [m/s]']
    velocity_y_ref: Annotated[np.longdouble, 'Скорость по оси y [m/s]']
    velocity_z_ref: Annotated[np.longdouble, 'Скорость по оси z [m/s]']
    heading: Annotated[np.longdouble, 'Курс']
    pitch: Annotated[np.longdouble, 'Тангаж']
    roll: Annotated[np.longdouble, 'Крен']
    C_body_to_ref: Annotated[np.ndarray, 'МНК перехода от связанной к опорной']
    C_inertial_to_body: Annotated[np.ndarray, 'МНК перехода от инерциальной к связанной']
    C_inertial_to_ref: Annotated[np.ndarray, 'МНК перехода от инерциальной к опорной']

    def __init__(
        self, 
        t: int | float | np.longdouble,
        latitude: int | float | np.longdouble,
        longitude: int | float | np.longdouble,
        velocity_x_ref: int | float | np.longdouble,
        velocity_y_ref: int | float | np.longdouble,
        velocity_z_ref: int | float | np.longdouble,
        heading: int | float | np.longdouble,
        pitch: int | float | np.longdouble,
        roll: int | float | np.longdouble,
        C_body_to_ref: np.ndarray,
        C_inertial_to_body: np.ndarray,
        C_inertial_to_ref: np.ndarray | None = None,
    ):
        self.t = np.longdouble(t)
        self.latitude = np.longdouble(latitude)
        self.longitude = np.longdouble(longitude)
        self.velocity_x_ref = np.longdouble(velocity_x_ref)
        self.velocity_y_ref = np.longdouble(velocity_y_ref)
        self.velocity_z_ref = np.longdouble(velocity_z_ref)
        self.heading = np.longdouble(heading)
        self.pitch = np.longdouble(pitch)
        self.roll = np.longdouble(roll)
        self.C_body_to_ref = C_body_to_ref
        self.C_inertial_to_body = C_inertial_to_body
        self.C_inertial_to_ref = np.array(C_inertial_to_ref or np.eye(3), dtype=np.longdouble)

    @property
    def velocity(self):
        return np.array([self.velocity_x_ref, self.velocity_y_ref, self.velocity_z_ref], np.longdouble)

    @velocity.setter
    def velocity(self, value: np.ndarray) -> None:
        assert len(value) == 3, '3 values of velocity projection was expected'
        self.velocity_x_ref = value[0]
        self.velocity_y_ref = value[1]
        self.velocity_z_ref = value[2]
