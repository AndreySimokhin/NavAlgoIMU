from typing import Annotated
from dataclasses import dataclass
import numpy as np


@dataclass
class SmallIncrements:
    t: Annotated[np.longdouble, 'Время [sec]']
    dax: Annotated[np.longdouble, 'Малое приращение ускорения по x [м/с^2]']
    day: Annotated[np.longdouble, 'Малое приращение ускорения по y [м/с^2]']
    daz: Annotated[np.longdouble, 'Малое приращение ускорения по z [м/с^2]']
    dwx: Annotated[np.longdouble, 'Малое приращение угловой скорости по x [рад/с]']
    dwy: Annotated[np.longdouble, 'Малое приращение угловой скорости по y [рад/с]']
    dwz: Annotated[np.longdouble, 'Малое приращение угловой скорости по z [рад/с]']

    def __init__(
        self, 
        t: int | float | np.longdouble,
        dax: int | float | np.longdouble,
        day: int | float | np.longdouble,
        daz: int | float | np.longdouble,
        dwx: int | float | np.longdouble,
        dwy: int | float | np.longdouble,
        dwz: int | float | np.longdouble,
    ):
        self.t = np.longdouble(t)
        self.dax = np.longdouble(dax)
        self.day = np.longdouble(day)
        self.daz = np.longdouble(daz)
        self.dwx = np.longdouble(dwx)
        self.dwy = np.longdouble(dwy)
        self.dwz = np.longdouble(dwz)

    @property
    def da(self) -> np.ndarray:
        return np.array([
            self.dax, 
            self.day,
            self.daz,
        ], np.longdouble)
    
    @property
    def dw(self) -> np.ndarray:
        return np.array([
            self.dwx, 
            self.dwy,
            self.dwz,
        ], np.longdouble)
