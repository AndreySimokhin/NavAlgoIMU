import asyncio
import numpy as np

from BINS_algo import IMU_emulator, Navigation_System, State


if __name__ == '__main__':
    imu = IMU_emulator(
        initial_state=State(
            t=0,
            latitude=np.deg2rad(56),
            longitude=0,
            heading=np.deg2rad(45),
            pitch=np.deg2rad(0),
            roll=np.deg2rad(5),
            velocity_x_ref=0,
            velocity_y_ref=0,
            velocity_z_ref=0,
        ),
        frequency=800,  # 800 Hz
        ttl_sec=5400,  # 90 min
    )

    nav = Navigation_System(
        imu=imu,
    )
    asyncio.run(nav.navigate())
    nav.save_states('./src/result.txt')

