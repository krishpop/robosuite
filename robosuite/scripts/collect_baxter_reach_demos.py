"""End-effector control for bimanual Baxter robot.

This script shows how to use inverse kinematics solver from Bullet
to command the end-effectors of two arms of the Baxter robot.
"""
import argparse
import datetime
import os
import shutil
from glob import glob

import h5py
import numpy as np

import robosuite


def gather_demonstrations_as_hdf5(directory, out_dir):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file, and another directory that contains the
    raw model.xml files.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - name of corresponding model xml in `models` directory
            states (dataset) - flattened mujoco states
            joint_velocities (dataset) - joint velocities applied during demonstration
            gripper_actuations (dataset) - gripper controls applied during demonstration
            right_dpos (dataset) - end effector delta position command for
                single arm robot or right arm
            right_dquat (dataset) - end effector delta rotation command for
                single arm robot or right arm
            left_dpos (dataset) - end effector delta position command for
                left arm (bimanual robot only)
            left_dquat (dataset) - end effector delta rotation command for
                left arm (bimanual robot only)

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file and model xmls.
            The model xmls will be stored in a subdirectory called `models`.
    """

    # store model xmls in this directory
    model_dir = os.path.join(out_dir, "models")
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)

    hdf5_path = os.path.join(out_dir, "demo.hdf5")
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = 'BaxterReach'  # will get populated at some point

    for ep_directory in os.listdir(directory):

        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        joint_velocities = []
        gripper_actuations = []
        right_dpos = []
        right_dquat = []
        left_dpos = []
        left_dquat = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                joint_velocities.append(ai["joint_velocities"])
                gripper_actuations.append(ai["gripper_actuation"])
                right_dpos.append(ai.get("right_dpos", []))
                right_dquat.append(ai.get("right_dquat", []))
                left_dpos.append(ai.get("left_dpos", []))
                left_dquat.append(ai.get("left_dquat", []))

        if len(states) == 0:
            continue

        # Delete the first actions and the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action.
        del states[-1]
        del joint_velocities[0]
        del gripper_actuations[0]
        del right_dpos[0]
        del right_dquat[0]
        del left_dpos[0]
        del left_dquat[0]

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model file name as an attribute
        ep_data_grp.attrs["model_file"] = "model_{}.xml".format(num_eps)

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("joint_velocities", data=np.array(joint_velocities))
        ep_data_grp.create_dataset(
            "gripper_actuations", data=np.array(gripper_actuations)
        )
        ep_data_grp.create_dataset("right_dpos", data=np.array(right_dpos))
        ep_data_grp.create_dataset("right_dquat", data=np.array(right_dquat))
        ep_data_grp.create_dataset("left_dpos", data=np.array(left_dpos))
        ep_data_grp.create_dataset("left_dquat", data=np.array(left_dquat))
        ep_data_grp.create_dataset("goal", data=dic["goal"])
        # copy over and rename model xml
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        shutil.copy(xml_path, model_dir)
        os.rename(
            os.path.join(model_dir, "model.xml"),
            os.path.join(model_dir, "model_{}.xml".format(num_eps)),
        )

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = robosuite.__version__
    grp.attrs["env"] = env_name

    f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(robosuite.models.assets_root, "demonstrations"),
    )
    args = parser.parse_args()

    tmp_directory = "/tmp/1559068066_9817421"

    new_dir = os.path.join(args.directory, "BaxterReach")

    gather_demonstrations_as_hdf5(tmp_directory, new_dir)
