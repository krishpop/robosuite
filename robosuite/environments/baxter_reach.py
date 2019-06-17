from collections import OrderedDict
import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.baxter import BaxterEnv

from robosuite.models.arenas import TableArena
from robosuite.models.tasks import TableTopTask, UniformRandomSampler


class BaxterReach(BaxterEnv):
    def __init__(
            self,
            table_full_size=(0.8, 0.8, 0.8),
            table_friction=(1., 5e-3, 1e-4),
            use_object_obs=True,
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

        # whether to use ground-truth object states
        self.use_object_obs = False

        self.object_initializer = None

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
        goal = np.random.uniform(-0.3, 0.3, size=3)
        goal[2] = 0
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
        dist = self.get_dist()
        # if dist < 1:
        #    print('dist:', dist)
        if dist < 0.2:
            return True
        return False
