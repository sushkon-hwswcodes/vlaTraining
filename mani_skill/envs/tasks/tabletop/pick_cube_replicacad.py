from typing import Any, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.replicacad.scene_builder import ReplicaCADSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("PickCubeReplicaCAD-v1", max_episode_steps=200)
class PickCubeReplicaCADEnv(BaseEnv):
    """
    Pick cube task set inside a ReplicaCAD apartment scene.

    A Fetch robot must grasp a red cube placed on the floor and lift it
    to a randomised goal position. The scene geometry and lighting come
    from the ReplicaCAD dataset; only the cube and goal marker are added
    on top.

    Randomisations per episode:
    - cube xy position on the floor in a 0.4 x 0.4 m patch in front of
      the robot, random yaw
    - goal position in the same xy patch, z in [cube_half_size + 0.1,
      cube_half_size + 0.6]

    Success conditions:
    - cube within goal_thresh (0.025 m) of the goal position
    - robot is static (qvel < 0.2)
    """

    SUPPORTED_ROBOTS = ["fetch"]
    agent: Fetch

    cube_half_size = 0.02
    goal_thresh = 0.025
    # Robot spawns at [-1, 0, 0.02]; cube spawns ~1.5 m in front of it
    cube_spawn_center = (0.5, 0.0)
    cube_spawn_half_size = 0.2
    max_goal_height = 0.5

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        robot_init_qpos_noise=0.02,
        build_config_idxs=None,
        reconfiguration_freq=None,
        num_envs=1,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.scene_builder = ReplicaCADSceneBuilder(
            self, robot_init_qpos_noise=robot_init_qpos_noise
        )
        self._build_config_idxs = build_config_idxs
        if reconfiguration_freq is None:
            reconfiguration_freq = 0 if num_envs > 1 else 1
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            spacing=50,
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**21,
                max_rigid_contact_count=2**23,
            ),
        )

    @property
    def _default_sensor_configs(self):
        # Camera mounted on the robot torso, looking down at the floor in front
        pose = sapien_utils.look_at(eye=[1.0, 0, 1.4], target=[0.5, 0, 0.0])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        room_pose = sapien_utils.look_at([2.5, -2.5, 3], [0.0, 0.0, 0.0])
        room_cfg = CameraConfig("render_camera", room_pose, 512, 512, 1, 0.01, 100)
        robot_pose = sapien_utils.look_at([2, 0, 1], [0, 0, -1])
        robot_cfg = CameraConfig(
            "robot_render_camera",
            robot_pose,
            512,
            512,
            1.5,
            0.01,
            100,
            mount=self.agent.torso_lift_link,
        )
        return [room_cfg, robot_cfg]

    def _load_lighting(self, options: dict):
        # ReplicaCADSceneBuilder sets up its own lighting
        if self.scene_builder.builds_lighting:
            return
        return super()._load_lighting(options)

    def _load_agent(self, options: dict):
        super()._load_agent(options, self.scene_builder.robot_initial_pose)

    def _load_scene(self, options: dict):
        idxs = self._build_config_idxs
        if idxs is None:
            idxs = self.scene_builder.sample_build_config_idxs()
        elif isinstance(idxs, int):
            idxs = [idxs]
        self.scene_builder.build(idxs)

        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[self.cube_spawn_center[0], 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.scene_builder.initialize(env_idx)

            # Cube: random xy on floor in front of robot, random yaw
            xyz = torch.zeros((b, 3))
            xy = torch.rand((b, 2)) * self.cube_spawn_half_size * 2 - self.cube_spawn_half_size
            xyz[:, 0] = xy[:, 0] + self.cube_spawn_center[0]
            xyz[:, 1] = xy[:, 1] + self.cube_spawn_center[1]
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            # Goal: same xy range, z lifted above floor
            goal_xyz = torch.zeros((b, 3))
            goal_xy = torch.rand((b, 2)) * self.cube_spawn_half_size * 2 - self.cube_spawn_half_size
            goal_xyz[:, 0] = goal_xy[:, 0] + self.cube_spawn_center[0]
            goal_xyz[:, 1] = goal_xy[:, 1] + self.cube_spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b,)) * self.max_goal_height + self.cube_half_size + 0.1
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: dict):
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        # Fetch has 2 finger joints at the end — exclude them from static check
        qvel = self.agent.robot.get_qvel()[..., :-2]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
