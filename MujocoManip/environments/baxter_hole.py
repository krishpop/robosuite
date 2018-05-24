import numpy as np
from collections import OrderedDict
from MujocoManip.miscellaneous import RandomizationError
from MujocoManip.environments.baxter import BaxterEnv
from MujocoManip.models import *
from MujocoManip.models.model_util import xml_path_completion, array_to_string, joint


class BaxterHoleEnv(BaxterEnv):

    def __init__(self, 
                 gripper_type='TwoFingerGripper',
                 use_eef_ctrl=False,
                 use_camera_obs=True,
                 use_object_obs=True,
                 camera_name='frontview',
                 reward_shaping=True,
                 gripper_visualization=False,
                 **kwargs):
        """
            @gripper_type, string that specifies the gripper type
            @use_eef_ctrl, position controller or default joint controllder
            @table_size, full dimension of the table
            @table_friction, friction parameters of the table
            @use_camera_obs, using camera observations
            @use_object_obs, using object physics states
            @camera_name, name of camera to be rendered
            @camera_height, height of camera observation
            @camera_width, width of camera observation
            @camera_depth, rendering depth
            @reward_shaping, using a shaping reward
        """
        # initialize objects of interest
        cube = RandomBoxObject(size_min=[0.02, 0.02, 0.02],
                               size_max=[0.025, 0.025, 0.025])
        #pot = DefaultPotObject()
        self.hole = DefaultHoleObject()

        self.cylinder = CylinderObject(size=(0.01, 0.13))
        #pot = cube
        self.mujoco_objects = OrderedDict()

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # whether to show visual aid about where is the gripper
        self.gripper_visualization = gripper_visualization

        # reward configuration
        self.reward_shaping = reward_shaping

        super().__init__(gripper_left=None,#'LeftTwoFingerGripper',
                         gripper_right=None,#'TwoFingerGripper',
                         use_eef_ctrl=use_eef_ctrl,
                         use_camera_obs=use_camera_obs,
                         camera_name=camera_name,
                         gripper_visualization=gripper_visualization,
                         **kwargs)

    def _load_model(self):
        super()._load_model()
        self.mujoco_robot.set_base_xpos([0,0,0])

        """self.mujoco_arena = TableArena(full_size=self.table_size,
                                       friction=self.table_friction)

        self.mujoco_arena.set_origin([0.45 + self.table_size[0] / 2,0,0])
        """

        self.model = MujocoWorldBase()
        self.model.merge(EmptyArena())
        self.model.merge(self.mujoco_robot)

        self.hole_obj = self.hole.get_collision(name='hole', site=True)
        self.hole_obj.set('quat','0 0 0.707 0.707')
        self.hole_obj.set('pos','0.11 0 0.18')
        self.model.merge_asset(self.hole)
        self.model.worldbody.find(".//body[@name='left_hand']").append(self.hole_obj)

        self.cyl_obj = self.cylinder.get_collision(name='cylinder', site=True)
        self.cyl_obj.set('pos','0 0 0.15')

        self.model.merge_asset(self.cylinder)
        self.model.worldbody.find(".//body[@name='right_hand']").append(self.cyl_obj)
        self.model.worldbody.find(".//geom[@name='cylinder']").set('rgba','0 0 1 1')

    def _get_reference(self):
        super()._get_reference()
        self.hole_body_id = self.sim.model.body_name2id('hole')
        self.cyl_body_id = self.sim.model.body_name2id('cylinder')

    def _reset_internal(self):
        super()._reset_internal()

    def _compute_orientation(self):
        cyl_mat = self.sim.data.body_xmat[self.cyl_body_id]
        cyl_mat.shape = (3,3)
        cyl_pos = self.sim.data.body_xpos[self.cyl_body_id]

        hole_pos = self.sim.data.body_xpos[self.hole_body_id]
        hole_mat = self.sim.data.body_xmat[self.hole_body_id]
        hole_mat.shape = (3,3)

        v = cyl_mat @ np.array([0,0,1])
        v = v / np.linalg.norm(v)
        center = hole_pos + hole_mat @ np.array([0.1,0,0])

        t = (center - cyl_pos)@v / (np.linalg.norm(v)**2)
        d = np.linalg.norm(np.cross(v, cyl_pos-center))/np.linalg.norm(v)
        
        hole_normal = hole_mat @ np.array([0,0,1])
        return (t, d, abs(np.dot(hole_normal, v)/np.linalg.norm(hole_normal) \
                                                /np.linalg.norm(v)))

    def reward(self, action):
        reward = 0

        t, d, cos = self._compute_orientation()

        # Right location and angle
        if d < 0.06 and t >= -0.12 and t <= 0.14 and cos > 0.95:
            reward = 1

        # use a shaping reward
        if self.reward_shaping:
            # reaching reward
            hole_pos = self.sim.data.body_xpos[self.hole_body_id]
            gripper_site_pos = self.sim.data.body_xpos[self.cyl_body_id]
            dist = np.linalg.norm(gripper_site_pos - hole_pos)
            reaching_reward = 1 - np.tanh(3*dist)
            reward += reaching_reward

        return reward

    def _get_observation(self):
        di = super()._get_observation()
        # camera observations
        if self.use_camera_obs:
            camera_obs = self.sim.render(camera_name=self.camera_name,
                                         width=self.camera_width,
                                         height=self.camera_height,
                                         depth=self.camera_depth)
            if self.camera_depth:
                di['image'], di['depth'] = camera_obs
            else:
                di['image'] = camera_obs

        # low-level object information
        if self.use_object_obs:
            # position and rotation of cylinder and hole
            hole_pos = self.sim.data.body_xpos[self.hole_body_id]
            hole_quat = self.sim.data.body_xquat[self.hole_body_id]
            di['hole_pos'] = hole_pos
            di['hole_quat'] = hole_quat

            cyl_pos = self.sim.data.body_xpos[self.cyl_body_id]
            cyl_quat = self.sim.data.body_xquat[self.cyl_body_id]
            di['cyl_to_hole'] = cyl_pos - hole_pos
            di['cyl_quat'] = cyl_quat

            # Relative orientation parameters
            t, d, cos = self._compute_orientation()
            di['angle'] = cos
            di['t'] = t
            di['d'] = d

        return di

    def _check_contact(self):
        """
        Returns True if gripper is in contact with an object.
        """
        collision = False
        contact_geoms = self.gripper_right.contact_geoms() + self.gripper_left.contact_geoms()
        for contact in self.sim.data.contact[:self.sim.data.ncon]:
            if self.sim.model.geom_id2name(contact.geom1) in contact_geoms or \
               self.sim.model.geom_id2name(contact.geom2) in contact_geoms:
                collision = True
                break
        return collision

    def _check_terminated(self):
        """
        Returns True if task is successfully completed
        """
        return False
