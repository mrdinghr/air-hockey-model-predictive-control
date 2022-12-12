import os
import time
import numpy as np
import mujoco
from mushroom_rl.utils.mujoco.viewer import MujocoGlfwViewer
from script import control


if __name__ == '__main__':
    table_dir = os.path.join("table_test.xml")
    mj_model = mujoco.MjModel.from_xml_path(table_dir)
    mj_model.opt.timestep = 0.001
    mj_data = mujoco.MjData(mj_model)
    mj_viewer = MujocoGlfwViewer(mj_model, 0.001, start_paused=True)
    controller = control.Controller()
    # Read Recording
    record_data = np.load("hundred_data_after_clean.npy", allow_pickle=True)
    for data_idx in range(len(record_data)):
        record_data[data_idx][:, 0] -= 1.948/2
        record_data[data_idx][:, -1] -= record_data[data_idx][0, -1]
        init_diff = record_data[data_idx][5] - record_data[data_idx][0]
        from mushroom_rl.utils.angles import shortest_angular_distance
        init_diff[2] = shortest_angular_distance(record_data[data_idx][0, 2], record_data[data_idx][5, 2])
        init_vel = init_diff[:3] / init_diff[3]

        # Set up Puck (Red) Initial State
        mj_data.qpos[0:3] = record_data[data_idx][0, :3]
        mj_data.qvel[:] = 0.
        mj_data.qvel[0:3] = init_vel

        # # Set up Puck Record (Blue)
        # mj_data.qpos[3:5] = record_data[data_idx][0, :2]
        # mj_data.qpos[5] = 0.
        # mujoco.mju_axisAngle2Quat(mj_data.qpos[6:10], (0., 0., 1.), record_data[data_idx][0, 2])
        # mj_data.qvel[3:] = 0.

        # set up Mallet
        mj_data.qpos[-3] = -0.3
        mj_data.qpos[-2] = 0

        t = 0.
        i = 0

        mujoco.mj_step(mj_model, mj_data, 1)
        mj_viewer.render(mj_data)
        while t < 2:
            controller.reset(mj_data.qpos[-3:-1], mj_data.qvel[-3:-1])
            u = controller.action(mj_data.qpos[0:2], mj_data.qvel[0:2])
            # mujoco.mj_contactForce(mj_model, mj_data, mj_model.geom('mallet').id, u)
            mj_data.qvel[-3] = u[0]
            mj_data.qvel[-2] = u[1]
            mujoco.mj_step(mj_model, mj_data, 1)
            mj_viewer.render(mj_data)
            if mj_model.geom('puck').id in mj_data.contact.geom1 or mj_model.geom('puck').id in mj_data.contact.geom1:
                print("contact")
            t += 0.001
            time.sleep(1/1000)

