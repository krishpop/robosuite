from collections import OrderedDict
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.baxter import BaxterEnv

from robosuite.models.arenas import TableArena
from robosuite.models.tasks import TableTopTask, UniformRandomSampler


class BaxterLine(BaxterEnv):
    def __init__(
        self,
        table_full_size=(0.8, 0.8, 0.8),
        table_friction=(1., 5e-3, 1e-4),
        use_object_obs=False,
        reward_shaping=False,
        **kwargs
    ):
        """
        Args:

            table_full_size (3-tuple): x, y, and z dimensions of the table.

            table_friction (3-tuple): the three mujoco friction parameters for
                the table.

            use_object_obs (bool): if True, include object (pot) information in
                the observation.

            reward_shaping (bool): if True, use dense rewards.

        Inherits the Baxter environment; refer to other parameters described there.
        """

        self.mujoco_objects = OrderedDict()

        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction

        self.init_qpos = np.array([0.755, 0.126, 0.074, 0.123, 0.644, 1.415, -1.241, -0.711,
                                   0.159, -0.11, 0.167, -0.755, 1.347, -0.156])
        """ for arms above table
            np.array([0.535, -0.093, 0.038, 0.166, 0.643, 1.960, -1.297,
                                   -0.518, -0.026, -0.076, 0.175, -0.748, 1.641, -0.158])
            for arms at center of table
            np.array([0.755, 0.126, 0.074, 0.123, 0.644, 1.415, -1.241, -0.711,
                                   0.159, -0.11, 0.167, -0.755, 1.347, -0.156])
            for arms outstretched above table
            np.array([0.00, -0.55, 0.00, 1.28, 0.00, 0.26, 0.00,
                                   0.00, -0.55, 0.00, 1.28, 0.00, 0.26, 0.00])
        """

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        self.object_initializer = None
        self.goal = None

        super().__init__(
            use_indicator_object=True, gripper_left="LeftTwoFingerGripper", gripper_right="TwoFingerGripper", **kwargs
        )

    def _load_model(self):
        """
        Loads the arena and pot object.
        """
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0, 0, 0])

        # load model for table top workspace
        self.mujoco_arena = TableArena(
            table_full_size=self.table_full_size, table_friction=self.table_friction
        )
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # The sawyer robot has a pedestal, we want to align it with the table
        self.mujoco_arena.set_origin([0.45 + self.table_full_size[0] / 2, 0, 0])

        # task includes arena, robot, and objects of interest
        self.model = TableTopTask(
            self.mujoco_arena,
            self.mujoco_robot,
            self.mujoco_objects,
            self.object_initializer,
        )

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flattened array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()
        self.table_top_id = self.sim.model.site_name2id("table_top")

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()
        self.sim.data.qpos[self._ref_joint_pos_indexes] = self.init_qpos
        goal = np.zeros(3)  # np.random.uniform(-0.3, 0.3, size=3)
        goal[2] = 0.03  # z value is always table offset + 0.03
        goal += self.model.table_top_offset
        self.goal = goal

    @property
    def goal(self):
        return self.sim.data.qpos[self._ref_indicator_pos_low: self._ref_indicator_pos_low + 3]

    @goal.setter
    def goal(self, new_goal):
        self.move_indicator(new_goal)

    def get_dist(self):
        goal = self.goal
        return np.linalg.norm(self._l_eef_xpos - goal) + np.linalg.norm(self._r_eef_xpos - goal)

    def reward(self, action):
        """
        Reward function for the task.
        """
        dist = self.get_dist()
        return -dist

    @property
    def _world_quat(self):
        """World quaternion."""
        return T.convert_quat(np.array([1, 0, 0, 0]), to="xyzw")

    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:
            robot-state: contains robot-centric information.
            object-state: requires @self.use_object_obs to be True.
                contains object-centric information.
            image: requires @self.use_camera_obs to be True.
                contains a rendered frame from the simulation.
            depth: requires @self.use_camera_obs and @self.camera_depth to be True.
                contains a rendered depth map from the simulation
        """
        di = super()._get_observation()
        di['object-state'] = self.goal.copy()
        # camera observations
        if self.use_camera_obs:
            camera_obs = self.sim.render(
                camera_name=self.camera_name,
                width=self.camera_width,
                height=self.camera_height,
                depth=self.camera_depth,
            )
            if self.camera_depth:
                di["image"], di["depth"] = camera_obs
            else:
                di["image"] = camera_obs

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        contact_geoms = (
            self.gripper_right.contact_geoms() + self.gripper_left.contact_geoms()
        )
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if (
                self.sim.model.geom_id2name(contact.geom1) in contact_geoms
                or self.sim.model.geom_id2name(contact.geom2) in contact_geoms
            ):
                collision = True
                break
        return collision

    def _check_success(self):
        """
        Returns True if task is successfully completed
        """
        l_dist = np.abs(self._l_eef_xpos - self.goal)
        r_dist = np.abs(self._r_eef_xpos - self.goal)
        return l_dist[1] < 0.005 and r_dist[1] < 0.005 and self.get_dist() < 0.1
        # if dist < 1:
        #    print('dist:', dist)
        # if dist < 0.1:
        #     return True
        # return False
