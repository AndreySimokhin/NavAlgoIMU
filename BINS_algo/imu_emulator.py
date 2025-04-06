import numpy as np
import pandas as pd

from .constants import Const
from .small_increments import SmallIncrements
from .state import State
from . import math_functions as MathFunc


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
    frequency: int
    ttl_sec: int
    integrating_prescaler: float

    def __init__(self, initial_state: State, frequency: int = 800, ttl_sec: int = 90 * 60):
        self.initial_state = initial_state
        self.frequency = frequency
        self.ttl_sec = ttl_sec
        self.integrating_prescaler = 1

    def get_signals(self) -> tuple[np.ndarray, np.ndarray]:
        state = self.initial_state
        state.C_body_to_ref = MathFunc.calc_body_to_ref(state.heading, state.pitch, state.roll)
        state.C_inertial_to_body = state.C_body_to_ref.transpose()  # C_inertial_to_body(0) = C_ref_to_body = (C_body_to_ref(0)).transpose() 
        w_o = [
            np.longdouble(0),
            np.longdouble(Const.U_EARTH_ROTATION_RATE * np.cos(state.latitude)),
            np.longdouble(Const.U_EARTH_ROTATION_RATE * np.sin(state.latitude))
        ]
        a_o = [
            np.longdouble(0),
            np.longdouble(0),
            np.longdouble(Const.GRAVITY_AXELERATION)
        ]
        w_b = state.C_body_to_ref.transpose() @ w_o     # w_b = C_ref_to_body @ w_o
        a_b = state.C_body_to_ref.transpose() @ a_o     # a_b = C_ref_to_body @ a_o
        return w_b, a_b

    async def iter(self):
        current_time = 0
        w_b, a_b = self.get_signals()

        while current_time < self.ttl_sec:
            dt = self.integrating_prescaler / self.frequency
            current_time += dt
            yield SmallIncrements(
                t=current_time,
                dax=a_b[0] * dt,
                day=a_b[1] * dt,
                daz=a_b[2] * dt,
                dwx=w_b[0] * dt,
                dwy=w_b[1] * dt,
                dwz=w_b[2] * dt,
            )
