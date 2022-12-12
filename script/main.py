import mujoco
import os

table_dir = os.path.join("table_strike.xml")
mj_model = mujoco.MjModel.from_xml_path(table_dir)




