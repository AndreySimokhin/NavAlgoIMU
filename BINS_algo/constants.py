from typing import Annotated
import numpy as np


class Const:
    U_EARTH_ROTATION_RATE: Annotated[np.longdouble, "Скорость вращения земли (rad/sec)"] = np.longdouble(np.deg2rad(15 / 3_600))
    GRAVITY_AXELERATION: Annotated[np.longdouble, "Ускорение свободного падения (m/sec^2)"] = np.longdouble(9.8067)
    RADIUS_EARTH: Annotated[np.longdouble, "Радиус Земли (m)"] = np.longdouble(6_371_000)

    @staticmethod
    def E():
        return np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], np.longdouble)
