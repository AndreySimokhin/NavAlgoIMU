from typing import Annotated
import numpy as np


U_EARTH_ROTATION_RATE: Annotated[np.longdouble, "Скорость вращения земли (rad/sec)"] = np.longdouble(np.deg2rad(15) / 3_600)
GRAVITY_AXELERATION: Annotated[np.longdouble, "Ускорение свободного падения (m/sec^2)"] = np.longdouble(9.8067)
RADIUS_EARTH: Annotated[np.longdouble, "Радиус Земли (m)"] = np.longdouble(6_371_000)

__all__ = (
    'U_EARTH_ROTATION_RATE',
    'GRAVITY_AXELERATION',
    'RADIUS_EARTH',
)
