from typing import Annotated
from dataclasses import dataclass
import numpy as np

from .constants import Const


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

        C_body_to_ref: list | np.ndarray | None = None,
        C_inertial_to_body: list | np.ndarray | None = None,
        C_inertial_to_ref: list | np.ndarray | None = None,
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

        self.C_body_to_ref = np.array(
            C_body_to_ref or Const.E(),
            dtype=np.longdouble
        )
        self.C_inertial_to_body = np.array(
            C_inertial_to_body or Const.E(),
            dtype=np.longdouble
        )
        self.C_inertial_to_ref = np.array(
            C_inertial_to_ref or Const.E(),
            dtype=np.longdouble
        )
