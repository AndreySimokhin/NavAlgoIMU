import numpy as np
import pandas as pd

from .small_increments import SmallIncrements
from .state import State
from . import math_functions as MathFunc
from .constants import U_EARTH_ROTATION_RATE, RADIUS_EARTH, GRAVITY_AXELERATION


class IMU_reader:
    initial_state: State
    filepath: str
    dataframe: pd.DataFrame

    def __init__(self, initial_state: State, filepath: str):
        self.initial_state = initial_state
        self.filepath = filepath

    async def iter(self):
        with open(self.filepath, 'r+') as fp:
            self.dataframe = pd.read_csv(fp, sep=' ')
        for _, frame in self.dataframe.iterrows():
            yield SmallIncrements(
                t=frame['t'],
                dax=frame['ax'],
                day=frame['ay'],
                daz=frame['az'],
                dwx=frame['wx'],
                dwy=frame['wy'],
                dwz=frame['wz'],
            )


class IMU_emulator:
    initial_state: State
    a_b: np.ndarray
    w_b: np.ndarray
    frequency: int
    ttl_sec: int

    def __init__(self, initial_state: State, a_b: np.ndarray, w_b: np.ndarray, frequency: int = 800, ttl_sec: int = 90 * 60):
        self.initial_state = initial_state
        self.a_b = a_b
        self.w_b = w_b
        self.frequency = frequency
        self.ttl_sec = ttl_sec

    async def iter(self):
        current_time = 0

        while current_time < self.ttl_sec:
            dt = 1 / self.frequency
            current_time += dt
            yield SmallIncrements(
                t=current_time,
                dax=self.a_b[0] * dt,
                day=self.a_b[1] * dt,
                daz=self.a_b[2] * dt,
                dwx=self.w_b[0] * dt,
                dwy=self.w_b[1] * dt,
                dwz=self.w_b[2] * dt,
            )
