import numpy as np

from .small_increments import SmallIncrements
from .constants import U_EARTH_ROTATION_RATE, RADIUS_EARTH, GRAVITY_AXELERATION


def calc_body_to_ref(heading: np.longdouble, pitch: np.longdouble, roll: np.longdouble) -> np.ndarray:
    return np.array([
        [
            np.cos(heading) * np.cos(roll) + np.sin(heading) * np.sin(pitch) * np.sin(roll),
            np.sin(heading) * np.cos(pitch),
            np.cos(heading) * np.sin(roll) - np.sin(heading) * np.sin(pitch) * np.cos(roll)
        ],
        [
            -np.sin(heading) * np.cos(roll) + np.cos(heading) * np.sin(pitch) * np.sin(roll),
            np.cos(heading) * np.cos(pitch),
            -np.sin(heading) * np.sin(roll) - np.cos(heading) * np.sin(pitch) * np.cos(roll)
        ],
        [
            -np.cos(pitch) * np.sin(roll),
            np.sin(pitch),
            np.cos(pitch) * np.cos(roll)
        ]
    ], dtype=np.longdouble)

def normalizeMatrix(matrix: np.ndarray) -> None:
    '''Нормализация матрицы'''
    A64 = matrix.astype(np.float64)
    U, _, Vt = np.linalg.svd(A64)
    C = U @ Vt
    return C.astype(np.longdouble)

def EarthRotationRateRef(latitude: np.longdouble) -> np.ndarray:
    '''[0] Угловая скорость вращения Земли'''
    return  np.array([
        0,
        U_EARTH_ROTATION_RATE * np.cos(latitude),
        U_EARTH_ROTATION_RATE * np.sin(latitude),
    ], np.longdouble)

def integrateAngularRate(increment: SmallIncrements, small_increment: SmallIncrements) -> None:
    '''[1] Накопление приращений угловой скорости'''
    increment.dw += small_increment.dw

def integrateAxeleration(increment: SmallIncrements, small_increment: SmallIncrements) -> None:
    '''[2] Накопление приращений ускорения'''
    increment.da += small_increment.da

def errorCompensationAxelerometr(value: SmallIncrements) -> None:
    '''[3] Компенсация погрешностей акселерометров'''
    return

def errorCompensationAngularRateSensor(value: SmallIncrements) -> None:
    '''[4] Компенсация погрешностей гироскопов'''
    return

def calculateAxeleration(data: list[SmallIncrements], h1: int | float | np.longdouble) -> np.ndarray:
    '''[5] Вычисление ускорения на интервале с пониженной частотой (RATE_DECREASE * h) уравнение Рунге-Кута 4 порядка'''
    delta_acceleration = np.array([0, 0 , 0], np.longdouble)

    for i in range(0, len(data)):
        da = data[i].da
        dw = np.array([
            [           0, -data[i].dwz,  data[i].dwy],
            [ data[i].dwz,            0, -data[i].dwx],
            [-data[i].dwy,  data[i].dwx,           0],
        ], np.longdouble)

        k1 = da - dw @ delta_acceleration
        k2 = da - dw @ (delta_acceleration + h1 / 2 * k1)
        k3 = da - dw @ (delta_acceleration + h1 / 2 * k2)
        k4 = da - dw @ (delta_acceleration + h1 * k3)
        delta_acceleration += (k1 + 2*k2 + 2*k3 + k4) / 6

    return delta_acceleration

def calculateEulerRotationVectorProjection(data: list[SmallIncrements]) -> np.ndarray:
    '''[7] Вычисление проекций вектора конечного поворота Эйлера θ с пониженной частотой'''
    tetta = np.array([
        sum([incr.dwx for incr in data]),
        sum([incr.dwy for incr in data]),
        sum([incr.dwz for incr in data]),
    ], np.longdouble) + 2/3 * np.array([
        (data[0].dwy + data[1].dwy) * (data[2].dwz + data[3].dwz) - (data[0].dwz + data[1].dwz) * (data[2].dwy + data[3].dwy),
        (data[0].dwz + data[1].dwz) * (data[2].dwx + data[3].dwx) - (data[0].dwx + data[1].dwx) * (data[2].dwz + data[3].dwz),
        (data[0].dwx + data[1].dwx) * (data[2].dwy + data[3].dwy) - (data[0].dwy + data[1].dwy) * (data[2].dwx + data[3].dwx),
    ])
    
    return tetta

def calculateAngleOfBodyRotation(euler_vector_projection: np.ndarray) -> np.ndarray:
    '''[8] Расчёт матрицы поворота связанной СК (body) на малый угол'''
    euler_vector_module = np.sqrt(euler_vector_projection[0] ** 2 + euler_vector_projection[1] ** 2 + euler_vector_projection[2] ** 2)
    euler_vector_matrix = np.array([
        [0, -euler_vector_projection[2], euler_vector_projection[1]],
        [euler_vector_projection[2], 0, -euler_vector_projection[0]],
        [-euler_vector_projection[1], euler_vector_projection[0], 0],
    ], np.longdouble)
    C_prevbody_to_body = np.eye(3) - (np.sin(euler_vector_module) / euler_vector_module) * euler_vector_matrix + ((1 - np.cos(euler_vector_module)) / (euler_vector_module ** 2)) * (euler_vector_matrix @ euler_vector_matrix)
    return C_prevbody_to_body

def calculateAngularRateProjection(velocity_x_ref:np.longdouble, velocity_y_ref: np.longdouble, latitude: np.longdouble) -> np.ndarray:
    '''[10] Вычисление абсолютной угловой скорости опорной географической СК (ref)'''
    earthRotationRateRef = EarthRotationRateRef(latitude)
    delta_angular_rate_ref = np.array([
        -velocity_y_ref / RADIUS_EARTH,
        earthRotationRateRef[1] + velocity_x_ref / RADIUS_EARTH,
        earthRotationRateRef[2] + velocity_x_ref / RADIUS_EARTH * np.tan(latitude),
    ], np.longdouble)
    return delta_angular_rate_ref

def calculateAngleOfRefRotation(w_ref: np.ndarray, H4: int | float | np.longdouble) -> np.ndarray:
    '''[11] Расчёт матрицы поворота опорной СК (ref) на малый угол'''
    w_ref_matrix = np.array([
        [0, -w_ref[2], w_ref[1]],
        [w_ref[2], 0, -w_ref[0]],
        [-w_ref[1], w_ref[0], 0],
    ], np.longdouble)
    C_prevref_to_ref = np.eye(3) - H4 * w_ref_matrix + (H4 ** 2) / 2 * (w_ref_matrix @ w_ref_matrix)
    return C_prevref_to_ref

def calculateVelocityInRef(
        velocity_x_ref: np.longdouble,
        velocity_y_ref: np.longdouble,
        velocity_z_ref: np.longdouble,
        delta_acceleration_ref,
        delta_angular_rate_ref,
        H4: int | float | np.longdouble,
        latitude: np.longdouble,
    ) -> np.ndarray:
    '''Расчёт линейной скорости в опорной СК (ref)'''
    earth_rotation_rate = EarthRotationRateRef(latitude)

    next_velocity_x_ref = velocity_x_ref + delta_acceleration_ref[0] + H4 * ((earth_rotation_rate[2] + delta_angular_rate_ref[2]) * velocity_y_ref - (earth_rotation_rate[1] + delta_angular_rate_ref[1]) * velocity_z_ref)
    next_velocity_y_ref = velocity_y_ref + delta_acceleration_ref[1] + H4 * (-(earth_rotation_rate[2] + delta_angular_rate_ref[2]) * velocity_x_ref + delta_angular_rate_ref[0] * velocity_z_ref)
    next_velocity_z_ref = velocity_z_ref + delta_acceleration_ref[2] + H4 * ((earth_rotation_rate[1] + delta_angular_rate_ref[1]) * velocity_x_ref - delta_angular_rate_ref[0] * velocity_y_ref - GRAVITY_AXELERATION)
    return np.array([next_velocity_x_ref, next_velocity_y_ref, next_velocity_z_ref], dtype=np.longdouble)
