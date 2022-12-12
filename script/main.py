import mujoco
import os
from control import Controller
from mushroom_rl.utils.mujoco.viewer import MujocoGlfwViewer
import numpy as np
import time


def main():
    table_dir = os.path.join("../air_hockey_table/table_test.xml")
    mj_model = mujoco.MjModel.from_xml_path(table_dir)
    mj_model.opt.timestep = 0.001
    mj_data = mujoco.MjData(mj_model)
    mj_viewer = MujocoGlfwViewer(mj_model, 0.001, start_paused=True)
    controller = Controller()
    trajectory_num = 10
    for _ in range(trajectory_num):

        # Set up Puck (Red) Initial State
        mj_data.qpos[0:3] = np.random.rand(3)
        mj_data.qvel[:] = 0.
        mj_data.qvel[0:3] = np.random.rand(3)

        # set up Mallet
        mj_data.qpos[-3] = -0.3
        mj_data.qpos[-2] = 0

        t = 0.
        mujoco.mj_step(mj_model, mj_data, 1)
        mj_viewer.render(mj_data)
        while t < 2:
            controller.reset(mj_data.qpos[-3:-1], mj_data.qvel[-3:-1])
            u = controller.action(mj_data.qpos[0:2], mj_data.qvel[0:2])
            mj_data.qvel[-3] = u[0]
            mj_data.qvel[-2] = u[1]
            mujoco.mj_step(mj_model, mj_data, 1)
            mj_viewer.render(mj_data)
            if mj_model.geom('puck').id in mj_data.contact.geom1 or mj_model.geom('puck').id in mj_data.contact.geom1:
                print("contact")
            t += 0.001
            time.sleep(1 / 1000)



if __name__ == '__main__':
    main()




