
import csv
import numpy as np

from .constants import Const
from .state import State
from .small_increments import SmallIncrements
from .imu_emulator import IMU_emulator
from . import math_functions as MathFunc


class Navigation_System:
    __slot__ = ('imu')
    imu: IMU_emulator
    rate_decrease: int
    state_vault: list[State]

    def __init__(self, imu: IMU_emulator, rate_decrease: int = 4):
        self.imu = imu
        self.rate_decrease = rate_decrease
        self.state_vault = []
        self.imu.integrating_prescaler = self.imu.frequency / rate_decrease

    async def navigate(self) -> None:
        # Const
        H1 = 1
        H4 = H1 * 4

        # Buffered data
        increment: SmallIncrements | None = None
        increments: list[SmallIncrements] = []
        prevState = self.imu.initial_state
        time = 0

        async for small_increment in self.imu.iter():
            increment = increment or SmallIncrements(0, 0, 0, 0, 0, 0, 0)
            time = increment.t = small_increment.t
            
            # [1] Накопление приращений скорости
            MathFunc.integrateAngularRate(increment, small_increment)

            # [2] Накопление приращений ускорений
            MathFunc.integrateAxeleration(increment, small_increment)

            if time % H1 == 0:
                # [3] Компенсация погрешностей акселерометров
                MathFunc.errorCompensationAxelerometr(increment)
                
                # [4] Компенсация погрешностей гироскопов
                MathFunc.errorCompensationAngularRateSensor(increment)
                
                increments.append(increment)
                increment = None

            if time % H4 == 0:
                state = State(small_increment.t, 0, 0, 0, 0, 0, 0, 0, 0, 0)    # type:ignore

                # [5] Вычисление ускорения методом Рунге-Кута 4-го порядка
                delta_acceleration_body = MathFunc.calculateAxeleration(increments, H1)
                
                # [6] Вычисление ускорения в осях опорной СК
                delta_acceleration_ref = prevState.C_body_to_ref @ delta_acceleration_body

                # [7] Вычисление проекций вектора Эйлера
                euler_vector_matrix = MathFunc.calculateEulerRotationVectorProjection(increments)

                # [8] Расчёт матрицы поворота связанной СК (body) на малый угол
                C_prevbody_to_body = MathFunc.calculateAngleOfBodyRotation(euler_vector_matrix)

                # [9] Вычисление матрицы МНК для перехода из инерциальной СК в связанную
                C_inertial_to_body = C_prevbody_to_body @ prevState.C_inertial_to_body
                state.C_inertial_to_body = C_inertial_to_body

                # [10] Вычисление абсолютной угловой скорости опорной географической СК (ref)
                delta_angular_rate_ref = MathFunc.calculateAngularRateProjection(prevState.velocity_x_ref, prevState.velocity_y_ref, prevState.latitude)

                # [11] Расчёт матрицы поворота опорной СК (ref) на малый угол
                C_prevref_to_ref = MathFunc.calculateAngleOfRefRotation(delta_angular_rate_ref, H4)

                # [12] Вычисление матрицы МНК для перехода из инерциальной СК в опорную
                C_inertial_to_ref = C_prevref_to_ref @ prevState.C_inertial_to_ref
                state.C_inertial_to_ref = C_inertial_to_ref

                # [13] Вычисление матрицы МНК для перехода из связанной СК в опорную
                C_body_to_ref = C_inertial_to_ref @ C_inertial_to_body.transpose()

                # [14] Нормирование матрицы МНК для перехода из связанной СК в опорную
                MathFunc.normalizeMatrix(C_body_to_ref)
                state.C_body_to_ref = C_body_to_ref

                # [15] Линейные скорости в опорной СК
                [
                    state.velocity_x_ref,
                    state.velocity_y_ref,
                    state.velocity_z_ref,
                ] = MathFunc.calculateVelocityInRef(
                    prevState.velocity_x_ref,
                    prevState.velocity_y_ref,
                    prevState.velocity_z_ref,
                    delta_acceleration_ref,
                    delta_angular_rate_ref,
                    H4,
                    prevState.latitude,
                )
                state.velocity_z_ref = np.longdouble(0.0)

                # [16] Вычисление координат
                state.latitude = prevState.latitude + H4 * state.velocity_y_ref / Const.RADIUS_EARTH
                state.longitude = prevState.longitude + H4 * state.velocity_x_ref / (Const.RADIUS_EARTH * np.cos(prevState.latitude))

                # [17] Вычисление углов ориентации
                state.heading = np.arctan2(
                    C_body_to_ref[0, 1],
                    C_body_to_ref[1, 1]
                )
                state.roll = -np.arctan2(
                    C_body_to_ref[2, 0],
                    C_body_to_ref[2, 2]
                )
                state.pitch = -np.arctan2(
                    C_body_to_ref[2, 1],
                    np.sqrt(C_body_to_ref[0, 1] ** 2 + C_body_to_ref[1, 1] ** 2)
                )
                
                # Reset
                increment = None
                increments = []
                prevState = state
                self.state_vault.append(state)

    def save_states(self, filepath: str) -> None:
        with open(filepath, 'w+') as fp:
            with open(filepath, mode='w+', newline='') as fp:
                writer = csv.writer(fp)
                writer.writerow(["t", "latitude", "longitude", "velocity_x_ref", "velocity_y_ref", "velocity_z_ref", "heading", "pitch", "roll"])
                for state in self.state_vault:
                    writer.writerow([state.t, state.latitude, state.longitude, state.velocity_x_ref, state.velocity_y_ref, state.velocity_z_ref, state.heading, state.pitch, state.roll])
