import os
import h5py
import argparse
import numpy as np

from os.path import join as pjoin
from PIL import Image

import robosuite
import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import postprocess_model_xml
from robosuite.wrappers import IKWrapper
# from batchRL.envs.ik_wrapper import IKWrapper


class SARS_Extractor:
    """
    Class for extracting (s, a, r, s') tuples from the raw MuJoCo demonstrations.
    This can be used in a pipeline for Imitation Learning, Batch RL, or Off-policy RL.
    """

    def __init__(self, 
        demo_path, 
        use_eef_actions=False, 
        use_reward_shaping=False,
        use_images=False,
        robot_only=False,
        object_only=False):
        """
        Args:
            demo_path (string): The path to the folder containing the demonstrations.
                There should be a `demo.hdf5` file and a folder named `models` with 
                all of the stored model xml files from the demonstrations.

            use_eef_actions (bool): If true, do extraction in end effector action space.

            use_reward_shaping (bool): If true, use shaped rewards in transitions.

            use_images (bool): If true, include image observations.

            robot_only (bool): If true, restrict observations to robot features.

            object_only (bool): If true, restrict observations to object features.
        """
        self.demo_path = demo_path
        self.use_eef_actions = use_eef_actions
        self.eef_downsample = 10 # factor by which to downsample the points for eef actions
        self.use_reward_shaping = use_reward_shaping
        self.use_images = use_images
        self.robot_only = robot_only
        self.object_only = object_only
        self.hdf5_path = os.path.join(self.demo_path, "demo.hdf5")
        self.f = h5py.File(self.hdf5_path, "r")  
        self.env_name = self.f["data"].attrs["env"]

        print("Making extractor for env name {}".format(self.env_name))
        self.env = robosuite.make(
            self.env_name,
            has_renderer=False,
            has_offscreen_renderer=self.use_images,
            ignore_done=True,
            use_object_obs=(not self.use_images),
            use_camera_obs=self.use_images,
            camera_height=84,
            camera_width=84,
            camera_name="agentview",
            gripper_visualization=False,
            control_freq=100,
        )

        # list of all demonstrations episodes
        self.demos = list(self.f["data"].keys())

        # write extracted (s, a, s') to an output file
        base_name = "states"
        if self.use_images:
            base_name += "_images"
        elif self.robot_only:
            base_name += "_robot"
        elif self.object_only:
            base_name += "_object"
        if self.use_reward_shaping:
            base_name += "_dense"
        if self.use_eef_actions:
            base_name += "_eef_downsample_{}".format(self.eef_downsample)
        self.states_path = os.path.join(self.demo_path, "{}.hdf5".format(base_name))

    def extract_all_states(self):
        """
        Extracts states from all demonstrations in the hdf5 and writes
        them to the output hdf5.
        """

        self.f_sars = h5py.File(self.states_path, "w")   
        self.f_sars_grp = self.f_sars.create_group("data")
        num_samples_arr = []

        for demo_id in range(len(self.demos)):
            # if demo_id > 0:
            #     break
            num_samples = self.extract_states_for_episode_id(demo_id)
            if num_samples > 0:
                num_samples_arr.append(num_samples)

        # write dataset attributes (metadata)
        self.f_sars_grp.attrs["date"] = self.f["data"].attrs["date"]
        self.f_sars_grp.attrs["time"] = self.f["data"].attrs["time"]
        self.f_sars_grp.attrs["repository_version"] = self.f["data"].attrs["repository_version"]
        self.f_sars_grp.attrs["env"] = self.f["data"].attrs["env"]
        self.f_sars_grp.attrs["total"] = np.sum(num_samples_arr)

        print("Total number of samples: {}".format(np.sum(num_samples_arr)))
        print("Average number of samples {}".format(np.mean(num_samples_arr)))

        self.f_sars.close()

    def extract_states_for_episode_id(self, demo_id):
        """
        Extracts states and actions from a particular demonstration episode
        and writes it to the output hdf5 file.

        Args:
            demo_id (int): An episode index to extract states and actions for.
        """
        
        ep = self.demos[demo_id]
        ep_data_grp = self.f_sars_grp.create_group("demo_{}".format(demo_id))

        # read the model xml, using the metadata stored in the attribute for this episode
        model_file = self.f["data/{}".format(ep)].attrs["model_file"]
        model_path = os.path.join(self.demo_path, "models", model_file)
        with open(model_path, "r") as model_f:
            model_xml = model_f.read()

        # also store the location of model xml in the new hdf5 file
        ep_data_grp.attrs["model_file"] = model_file

        self.env.reset()
        xml = postprocess_model_xml(model_xml)
        self.env.reset_from_xml_string(xml)
        self.env.sim.reset()

        # load the flattened mujoco states and the actions in the environment
        states = self.f["data/{}/states".format(ep)][()]
        jvels = self.f["data/{}/joint_velocities".format(ep)][()]
        gripper_acts = self.f["data/{}/gripper_actuations".format(ep)][()]
        goal = np.array(self.f["data/{}".format(ep)].attrs["goal"])

        # assert gripper_acts.shape[1] == 1, 'gripper_acts shape: {}'.format(gripper_acts.shape)

        prev_obs = None
        prev_ac = None
        prev_rew = None
        prev_done = None
        prev_state = None
        ep_obs = []
        ep_acts = []
        ep_rews = []
        ep_next_obs = []
        ep_dones = []
        ep_states = []
        if self.use_images:
            prev_image = None
            ep_images = []
            ep_next_images = []

        # prev_eef_pos = None
        # prev_eef_rot = None

        # force the sequence of internal mujoco states one by one
        for t in range(len(states)):
            # self.env.sim.reset()
            self.env.sim.set_state_from_flattened(states[t])
            self.env.sim.forward()

            # make teleop visualization site colors transparent
            self.env.sim.model.site_rgba[self.env.eef_site_id] = np.array([0., 0., 0., 0.])
            self.env.sim.model.site_rgba[self.env.eef_cylinder_id] = np.array([0., 0., 0., 0.])

            obs_dict = self.env._get_observation()
            if self.use_images or self.robot_only:
                obs = np.array(obs_dict["robot-state"])
            elif self.object_only:
                obs = np.array(obs_dict["object-state"])
            else:
                obs = np.concatenate([obs_dict["robot-state"], obs_dict["object-state"]])
            action = np.concatenate([jvels[t], gripper_acts[t]])
            if self.use_images:
                image = np.array(obs_dict["image"])[::-1]

            # NOTE: ours tasks use reward r(s'), reward AFTER transition, so this is 
            # the reward for the previous timestep
            prev_rew = self.env.reward(None) 
            prev_done = 0
            if self.env._check_success():
                prev_done = 1

            # skips the first iteration
            if prev_obs is not None: 
                # record (s, a) from last iteration and s' from this one
                ep_obs.append(prev_obs) 
                ep_acts.append(prev_ac)
                ep_rews.append(prev_rew)
                ep_next_obs.append(obs)
                ep_dones.append(prev_done)
                ep_states.append(prev_state)
                if self.use_images:
                    ep_images.append(prev_image)
                    ep_next_images.append(image)
                    # im = Image.fromarray(prev_image)
                    # path = pjoin("tmp", "img%06d.png"%t)
                    # im.save(path)

            prev_obs = np.array(obs)
            prev_ac = np.array(action)
            prev_state = np.array(states[t])
            if self.use_images:
                prev_image = np.array(image)

        # if len(states) == len(jvels):
        assert(len(states) == len(jvels))
        
        # play the last action to get one more additional data point.
        # this might be critical if only the last state got a reward in
        # the sparse reward setting. 
        # self.env.sim.reset()
        self.env.sim.set_state_from_flattened(states[-1])
        self.env.sim.forward()

        # make teleop visualization site colors transparent
        self.env.sim.model.site_rgba[self.env.eef_site_id] = np.array([0., 0., 0., 0.])
        self.env.sim.model.site_rgba[self.env.eef_cylinder_id] = np.array([0., 0., 0., 0.])

        obs_dict = self.env._get_observation()
        if self.use_images or self.robot_only:
            obs = np.array(obs_dict["robot-state"])
        elif self.object_only:
            obs = np.array(obs_dict["object-state"])
        else:
            obs = np.concatenate([obs_dict["robot-state"], obs_dict["object-state"]])
        action = np.concatenate([jvels[-1], gripper_acts[-1]])
        if self.use_images:
            image = np.array(obs_dict["image"])[::-1]
        self.env.step(action)

        reward = self.env.reward(None)
        done = 0
        if self.env._check_success():
            done = 1

        # ensure consistency from loop above
        assert(np.array_equal(prev_obs, obs)) 
        if not self.use_eef_actions:
            assert(np.array_equal(prev_ac, action))
        # assert(done == 1)

        obs_dict = self.env._get_observation()
        if self.use_images or self.robot_only:
            next_obs = np.array(obs_dict["robot-state"])
        elif self.object_only:
            next_obs = np.array(obs_dict["object-state"])
        else:
            next_obs = np.concatenate([obs_dict["robot-state"], obs_dict["object-state"]])
        if self.use_images:
            next_image = np.array(obs_dict["image"])[::-1]

        ep_obs.append(obs)
        ep_acts.append(action)
        ep_rews.append(reward)
        ep_next_obs.append(next_obs)
        ep_dones.append(done)
        ep_states.append(states[-1])
        if self.use_images:
            ep_images.append(image)
            ep_next_images.append(next_image)

        if self.use_eef_actions:

            # re-run through transitions to rewrite actions as end effector actions, and
            # downsample the frames
            ep_eef_obs = []
            ep_eef_acts = []
            ep_eef_rews = []
            ep_eef_next_obs = []
            ep_eef_dones = []
            ep_eef_states = []
            if self.use_images:
                ep_eef_images = []
                ep_eef_next_images = []

            self.env.sim.reset()
            self.env.sim.set_state_from_flattened(ep_states[0])
            self.env.sim.forward()

            # make teleop visualization site colors transparent
            self.env.sim.model.site_rgba[self.env.eef_site_id] = np.array([0., 0., 0., 0.])
            self.env.sim.model.site_rgba[self.env.eef_cylinder_id] = np.array([0., 0., 0., 0.])

            obs_dict = self.env._get_observation()
            prev_eef_pos = np.array(obs_dict["eef_pos"])
            prev_eef_rot = T.quat2mat(obs_dict["eef_quat"])
            prev_obs = np.array(ep_obs[0])
            prev_done = ep_dones[0]
            if self.use_images:
                prev_image = np.array(obs_dict["image"])[::-1]

            loop_cnt = self.eef_downsample
            num_transitions = len(ep_obs)
            while True:
                #self.env.sim.reset()
                self.env.sim.set_state_from_flattened(ep_states[loop_cnt])
                self.env.sim.forward()

                # make teleop visualization site colors transparent
                self.env.sim.model.site_rgba[self.env.eef_site_id] = np.array([0., 0., 0., 0.])
                self.env.sim.model.site_rgba[self.env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
                
                obs_dict = self.env._get_observation()
                cur_eef_pos = np.array(obs_dict["eef_pos"])
                cur_eef_rot = T.quat2mat(obs_dict["eef_quat"])
                dpos = cur_eef_pos - prev_eef_pos
                drotation = prev_eef_rot.T.dot(cur_eef_rot)
                dquat = T.mat2quat(drotation)

                # current timestep lets us compute delta pose from last downsampled timestep to get action for last timestep
                # but take gripper action from last downsapled timestep
                prev_eef_action = np.concatenate([dpos, dquat, ep_acts[loop_cnt - self.eef_downsample][-1:]])
                
                # take reward from the immediate prior timestep, since rewards at s depend on s'
                prev_rew = ep_rews[loop_cnt - 1]
                cur_obs = np.array(ep_obs[loop_cnt])
                if self.use_images:
                    cur_image = np.array(ep_images[loop_cnt])

                ep_eef_obs.append(prev_obs)
                ep_eef_acts.append(prev_eef_action)
                ep_eef_rews.append(prev_rew)
                ep_eef_next_obs.append(cur_obs)
                ep_eef_dones.append(prev_done)
                if self.use_images:
                    ep_eef_images.append(prev_image)
                    ep_eef_next_images.append(cur_image)

                prev_obs = np.array(cur_obs)
                prev_done = ep_dones[loop_cnt]
                prev_eef_pos = np.array(cur_eef_pos)
                prev_eef_rot = np.array(cur_eef_rot)
                if self.use_images:
                    prev_image = np.array(cur_image)

                loop_cnt += self.eef_downsample
                if loop_cnt > num_transitions - 1:
                    if loop_cnt - self.eef_downsample != num_transitions - 1:
                        # back track to the last point to make sure we add it
                        loop_cnt = num_transitions - 1
                    else:
                        break

            # overwrite prior variables
            ep_obs = ep_eef_obs
            ep_acts = ep_eef_acts
            ep_rews = ep_eef_rews
            ep_next_obs = ep_eef_next_obs
            ep_dones = ep_eef_dones
            if self.use_images:
                ep_images = ep_eef_images
                ep_next_images = ep_eef_next_images

        # write datasets for states and actions
        ep_data_grp.create_dataset("obs", data=np.array(ep_obs))
        ep_data_grp.create_dataset("actions", data=np.array(ep_acts))
        ep_data_grp.create_dataset("rewards", data=np.array(ep_rews))
        ep_data_grp.create_dataset("next_obs", data=np.array(ep_next_obs))
        ep_data_grp.create_dataset("dones", data=np.array(ep_dones))
        ep_data_grp.create_dataset("states", data=np.array(ep_states))
        ep_data_grp.attrs["goal"] = goal
        if self.use_images:
            ep_data_grp.create_dataset("images", data=np.array(ep_images))
            ep_data_grp.create_dataset("next_images", data=np.array(ep_next_images))

        # write some metadata
        ep_data_grp.attrs["num_samples"] = len(ep_obs) # number of transitions in this episode
        print("ep {}: wrote {} transitions".format(demo_id, ep_data_grp.attrs["num_samples"]))
        return ep_data_grp.attrs["num_samples"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
    )
    # flag for end effector actions
    parser.add_argument(
        "--eef", 
        action='store_true',
    )
    # flag for reward shaping
    parser.add_argument(
        "--dense", 
        action='store_true',
    )
    # flag for image observations
    parser.add_argument(
        "--images", 
        action='store_true',
    )
    # flag for restricting to robot observations
    parser.add_argument(
        "--robot", 
        action='store_true',
    )
    # flag for restricting to object observations
    parser.add_argument(
        "--object", 
        action='store_true',
    )
    args = parser.parse_args()

    extractor = SARS_Extractor(demo_path=args.folder, use_eef_actions=args.eef, use_reward_shaping=args.dense, use_images=args.images, robot_only=args.robot, object_only=args.object)
    extractor.extract_all_states()
