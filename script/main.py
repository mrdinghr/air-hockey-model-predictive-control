import mujoco
import os
from control import Controller
from mushroom_rl.utils.mujoco.viewer import MujocoGlfwViewer
import numpy as np
import time


def out_of_boundary(puck_pos):
    if abs(puck_pos[0]) > 0.974 or abs(puck_pos[1]) > 0.519:
        return True
    return False


def check_hit(contact, puck_id, mallet_id):
    if (puck_id in contact.geom1 and mallet_id in contact.geom2) or (
            mallet_id in contact.geom1 and puck_id in contact.geom2):
        return True
    return False


def main():
    table_dir = os.path.join("../air_hockey_table/table_test.xml")
    mj_model = mujoco.MjModel.from_xml_path(table_dir)
    mj_model.opt.timestep = 0.001
    mj_data = mujoco.MjData(mj_model)
    mj_viewer = MujocoGlfwViewer(mj_model, 0.001, start_paused=True)
    controller = Controller()
    trajectory_num = 10
    for _ in range(trajectory_num):
        has_coll = False
        # Set up Puck (Red) Initial State
        mj_data.qpos[0:3] = np.random.rand(3) * 0.5
        mj_data.qvel[:] = 0.
        # mj_data.qvel[0:3] = np.random.rand(3) * 10

        # set up Mallet
        mj_data.qpos[-3] = -0.3
        mj_data.qpos[-2] = 0
        mj_data.qvel[-3:] = np.array([3, 3, 0])
        t = 0.
        mujoco.mj_step(mj_model, mj_data, 1)
        mj_viewer.render(mj_data)
        while t < 2:
            controller.reset(mj_data.qpos[-3:-1], mj_data.qvel[-3:-1])
            if has_coll:
                mj_data.qvel[3:] = 0.
            else:
                u = controller.pd_control(mj_data.qpos[0:2], mj_data.qvel[0:2])
                u = np.append(np.zeros(3), u)
                u = np.append(u, 0)
                mj_data.qfrc_applied = u
            mujoco.mj_step(mj_model, mj_data, 1)
            mj_viewer.render(mj_data)
            if mj_model.geom('puck').id in mj_data.contact.geom1:
                print("contact")
            if check_hit(mj_data.contact, mj_data.geom('puck').id, mj_data.geom('mallet').id):
                has_coll = True
            t += 0.001
            time.sleep(1 / 1000)
            if out_of_boundary(mj_data.qpos[0:3]):
                print('score')
                break


if __name__ == '__main__':
    main()
