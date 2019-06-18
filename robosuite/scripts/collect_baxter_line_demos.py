"""End-effector control for bimanual Baxter robot.

This script shows how to use inverse kinematics solver from Bullet
to command the end-effectors of two arms of the Baxter robot.
"""
import argparse
import datetime
import os
import shutil
import time
from glob import glob

import h5py
import numpy as np

import robosuite
from robosuite.wrappers import IKWrapper, DataCollectionWrapper


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
    env_name = 'BaxterLine'  # will get populated at some point

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
        # ep_data_grp.attrs["goal"] = dic["goal"]
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


""" EE Positions at env reset (center of table)
l ee pos:
[0.85159155,
 0.03451803,
 0.82883006]
r ee pos:
[0.84031527,
 -0.03995245,  
 .83450149]
"""


def run_collection(env, goal_y, init_jpos=None, init_ind_pos=None, perturb_init=False):
    obs = env.reset()
    # env.viewer.set_camera(camera_id=2)
    # env.render()

    # First set indicator (goal) position for collection
    bullet_data_path = os.path.join(robosuite.models.assets_root, "bullet_data")
    goal_pos = env.goal
    env.set_goal(goal_pos + np.array([0, goal_y, 0.03]))
    # env.set_goal(np.array([0.85, -.35, 0.83]))
    goal_pos = env.goal

    # Init joint position if specified, and
    if init_jpos is not None:
        env.set_robot_joint_positions(init_jpos)
        if perturb_init:
            env.ignore_inputs = True
            z_init = np.array([0, 0, np.random.choice([-.02, .03])])
            z_init += init_ind_pos
            l_goal_pos, r_goal_pos = z_init + np.array([0, 0.03, 0]), z_init + np.array([0, -0.03, 0])
            l_offset, r_offset = env._l_eef_xpos - l_goal_pos, env._r_eef_xpos - r_goal_pos
            while (np.abs(r_offset)[1] > 0.005 or np.abs(l_offset)[1] > 0.005):
                dpos_right = r_goal_pos - env._r_eef_xpos
                dpos_left = l_goal_pos - env._l_eef_xpos
                l_offset, r_offset = env._l_eef_xpos - l_goal_pos, env._r_eef_xpos - r_goal_pos
                dquat = np.array([0, 0, 0, 1])
                grasp = 0.
                action = np.concatenate([4e-3 * dpos_right, dquat, 4e-3 * dpos_left, dquat,
                                         [grasp, grasp]])
                env.step(action)
                # env.render()
            env.ignore_inputs = False

    n_steps = 3 # int(8 - (0.35 - abs(goal_y)) // 0.05)
    print('Starting: ', 'l_ee', env._l_eef_xpos, 'r_ee', env._r_eef_xpos, 'g', goal_pos)

    ret = move_to_goal(env, goal_pos, n_steps)
    env.close()
    return ret


def run_random(env):
    obs = env.reset()
    for i in range(200):
        vel = np.zeros(16)
        vel[4:6], vel[11:13] = (np.random.sample(size=2)-1)*.3, (np.random.sample(size=2)-1)*.3
        env.step(vel)
        print('Step {}:'.format(i))
        print('='*30)
        print(vel)
        print(env.sim.get_state().flatten())
        env.render()
    return


def move_to_goal(env, goal_pos, n_steps):
    l_goal_pos, r_goal_pos = goal_pos + np.array([0, 0.03, 0]), goal_pos + np.array([0, -0.03, 0])
    # lee y-offset: 0.03, ree y-offset: -0.03
    last_dist = 0

    is_first = True
    task_completion_hold_count = -1
    step_count = 0
    for i in range(n_steps - 1):
        left_traj = np.linspace(env._l_eef_xpos, l_goal_pos, n_steps - i)
        right_traj = np.linspace(env._r_eef_xpos, r_goal_pos, n_steps - i)
        l_ac, r_ac = left_traj[1], right_traj[1]
        for t in range(500):
            dpos_right = r_ac - env._r_eef_xpos
            dpos_left = l_ac - env._l_eef_xpos
            if np.abs(dpos_left).sum() < .05 and np.abs(dpos_right).sum() < .05 and i != n_steps - 2:
                break
            A = 1e-3 * t

            dquat = np.array([0, 0, 0, 1])
            grasp = 0.
            action = np.concatenate([A * dpos_right, dquat, A * dpos_left, dquat, [grasp, grasp]])

            dist = env.get_dist()
            if last_dist - dist > 0.01:  # stop applying actions, something went wrong
                done = True
            else:
                obs, reward, done, info = env.step(action)
                step_count += 1

                if is_first:
                    is_first = False

                    # We grab the initial model xml and state and reload from those so that
                    # we can support deterministic playback of actions from our demonstrations.
                    # This is necessary due to rounding issues with the model xml and with
                    # env.sim.forward(). We also have to do this after the first action is 
                    # applied because the data collector wrapper only starts recording
                    # after the first action has been played.
                    initial_mjstate = env.sim.get_state().flatten()
                    xml_str = env.model.get_xml()
                    env.reset_from_xml_string(xml_str)
                    env.sim.reset()
                    env.sim.set_state_from_flattened(initial_mjstate)
                    env.sim.forward()
                    # env.viewer.set_camera(camera_id=2)

            if args.render:
                env.render()

            l_offset, r_offset = env._l_eef_xpos - l_goal_pos, env._r_eef_xpos - r_goal_pos

            # state machine to check for having a success for 10 consecutive timesteps
            if env._check_success():
                if task_completion_hold_count > 0:
                    task_completion_hold_count -= 1  # latched state, decrement count
                else:
                    task_completion_hold_count = 10  # reset count on first success timestep
            else:
                task_completion_hold_count = -1  # null the counter if there's no success

            if (env._check_success() and task_completion_hold_count==0):
                print('Steps:', step_count)
                print('Joint positions:', robot_jpos_getter(env).round(3))
                print('EE positions:', env._l_eef_xpos, env._r_eef_xpos)
                print('done:', done, 'success:', env._check_success())
                return (robot_jpos_getter(env).round(3), env._l_eef_xpos, env._r_eef_xpos,
                        goal_pos, env._check_success())


def collect_joints(env):
    left_y_states = np.linspace(-0.35, -0.05, 7)
    right_y_states = np.linspace(0.05, 0.35, 7)
    d = np.load('line_start_pos.npz')
    jpos_l, lee_l, ree_l, goals = list(d['jpos']), list(d['l_ee']), list(d['r_ee']), list(d['goals'])

    if len(jpos_l) < 7:
        for y_del in left_y_states[len(jpos_l):]:
            jpos, lee, ree, g, succ = run_collection(env, y_del)
            if succ:
                jpos_l.append(jpos)
                lee_l.append(lee)
                ree_l.append(ree)
                goals.append(g)
                np.savez('line_start_pos.npz', jpos=np.array(jpos_l), l_ee=np.array(lee_l), r_ee=np.array(ree_l),
                         goals=np.array(goals))
            else:
                print('Failed to reach', g)

    for y_del in right_y_states[len(jpos_l) - 7:]:
        jpos, lee, ree, g, succ = run_collection(env, y_del)
        if succ:
            jpos_l.append(jpos)
            lee_l.append(lee)
            ree_l.append(ree)
            goals.append(g)
            np.savez('line_start_pos.npz', jpos=np.array(jpos_l), l_ee=np.array(lee_l), r_ee=np.array(ree_l),
                     goals=np.array(goals))
        else:
            print('Failed to reach', g)


def robot_jpos_getter(env) :
    return np.array(env._joint_positions)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default=os.path.join(robosuite.models.assets_root, "demonstrations"),
    )
    parser.add_argument(
        "--render",
        type=bool,
        default=False,
    )

    args = parser.parse_args()

    # initialize a Baxter environment
    env = robosuite.make(
        "BaxterLine",
        ignore_done=True,
        has_renderer=args.render,
        gripper_visualization=True,
        use_camera_obs=False,
    )
    # env.viewer.set_camera(camera_id=2)
    env = IKWrapper(env, 1)
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, tmp_directory)

    j_states = np.load('line_start_pos.npz')

    for i, g in enumerate(j_states['goals']):
        if i >= 3:
            run_collection(env, j_states['goals'][i - 3][1], j_states['jpos'][i], g, True)
            run_collection(env, j_states['goals'][i - 3][1], j_states['jpos'][i], g, False)
        if i <= len(j_states['goals']) - 4:
            run_collection(env, j_states['goals'][i + 3][1], j_states['jpos'][i], g, True)
            run_collection(env, j_states['goals'][i + 3][1], j_states['jpos'][i], g, False)
    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(args.directory, "BaxterLine")
    os.makedirs(new_dir)

    gather_demonstrations_as_hdf5(tmp_directory, new_dir)
