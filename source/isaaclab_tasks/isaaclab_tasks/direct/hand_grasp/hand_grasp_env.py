# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.utils.math import quat_apply
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.sim as sim_utils


if TYPE_CHECKING:
    from isaaclab_tasks.direct.franka_hand.franka_hand_env_cfg import FrankaHandEnvCfg
    from isaaclab_tasks.direct.franka_hand.franka_env_cfg import FrankaEnvCfg


class HandGraspEnv(DirectRLEnv):
    cfg: FrankaEnvCfg | FrankaHandEnvCfg

    def __init__(self, cfg: FrankaEnvCfg | FrankaHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints#27
        # self.num_hand_dofs = 20

        # buffers for position targets
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()
        # self.actuated_arm_indices = list()
        # for joint_name in cfg.arm_joint_names:
        #     self.actuated_arm_indices.append(self.hand.joint_names.index(joint_name))
        # self.actuated_arm_indices.sort()

        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)
        # arm end body
        self.arm_end_body = self.hand.body_names.index(self.cfg.armend_body_name[0])
        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # track goal resets
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # used to compare object position
        self.in_hand_pos = self.object.data.default_root_state[:, 0:3].clone()
        # self.in_hand_pos[:, 2] -= 0.04
        # default goal positions
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([0.4, 0.0, 0.5], device=self.device)
        # initialize goal marker
        # self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # ik controller
        self.diff_ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",        # 控制位姿（而非 velocity）
            use_relative_mode=True,    # 使用绝对坐标控制
            ik_method="dls"             # damped least squares 方法
        )
        self.ik_controller = DifferentialIKController(
            cfg=self.diff_ik_cfg,
            num_envs=self.num_envs,
            device=self.device
        )
        self.pos_min = torch.tensor([0.3, -0.4, 0.3], device=self.device)
        self.pos_max = torch.tensor([0.7, 0.4, 0.7], device=self.device)

        # 识别末端和参与 IK 的关节
        self.ik_entity_cfg = SceneEntityCfg(
            name="robot",
            joint_names=self.cfg.arm_joint_names,  # IK 控制的关节，例如 ["panda_joint1", ..., "panda_joint7"]
            body_names=self.cfg.armend_body_name   # IK 控制的末端，例如 ["panda_hand"]
        )
        self.ik_entity_cfg.resolve(self.scene)

        self.ik_joint_ids = self.ik_entity_cfg.joint_ids
        self.arm_joint_ids = self.ik_joint_ids
        self.finger_joint_ids = self.actuated_dof_indices
        self.ee_body_id = self.ik_entity_cfg.body_ids[0]

        # Jacobian 索引（如果是固定底座，PhysX 不包含 root link）
        if self.hand.is_fixed_base:
            self.ee_jacobi_idx = self.ee_body_id - 1
        else:
            self.ee_jacobi_idx = self.ee_body_id
        
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/EEFrame",
            markers={
                "ee_frame": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.05, 0.05, 0.05),
                ),
            },
        )
        self.ee_marker = VisualizationMarkers(marker_cfg)



    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # hand
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.actions[:,7:],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        # arm
        # 假设 action shape = [num_envs, 7]
        # delta_pos = scale(self.actions[:, 0:3], -0.05, 0.05)  # Δx, Δy, Δz (每步最大5cm)
        # --- 解码动作 ---
        pos_actions = self.actions[:, 0:3]
        rot_actions = self.actions[:, 3:6]


        # --- 缩放动作 ---
        action_scale = torch.tensor([0.05, 0.05, 0.05, 0.2, 0.2, 0.2], device=self.device)
        delta_pose = action_scale * torch.cat([pos_actions, rot_actions], dim=-1)  # [num_envs, 6]

        # --- 更新当前末端位姿（世界坐标下） ---
        ee_pose_w = self.hand.data.body_state_w[:, self.ee_body_id, 0:7]
        ee_pos = ee_pose_w[:, 0:3]
        ee_quat = ee_pose_w[:, 3:7]

        # --- 设置控制器目标（需要提供 ee_pos 和 ee_quat） ---
        

        # --- 获取 Jacobian 和当前关节角 ---
        jacobian = self.hand.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.ik_joint_ids]
        joint_pos_arm = self.hand.data.joint_pos[:, self.ik_joint_ids]
        root_pose_w = self.hand.data.root_state_w[:, 0:7]

        # --- 转换为 base frame 下的 ee pose（IK 控制要求） ---
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pos, ee_quat
        )
        self.ik_controller.set_command(delta_pose, ee_pos=ee_pos_b, ee_quat=ee_quat_b)
        # --- 执行 IK 计算目标关节角 ---
        joint_pos_des = self.ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos_arm)

        # --- 应用目标 ---
        self.hand.set_joint_position_target(joint_pos_des, joint_ids=self.ik_joint_ids)


        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )
        self.ee_marker.visualize(ee_pos, ee_quat,
            marker_indices=torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        )


    def _get_observations(self) -> dict:
        if self.cfg.asymmetric_obs:
            self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[
                :, self.finger_bodies
            ]

        if self.cfg.obs_type == "openai":
            obs = self.compute_reduced_observations()
        elif self.cfg.obs_type == "full":
            obs = self.compute_full_observations()
        else:
            print("Unknown observations type!")

        if self.cfg.asymmetric_obs:
            states = self.compute_full_state()

        observations = {"policy": obs}
        if self.cfg.asymmetric_obs:
            observations = {"policy": obs, "critic": states}

        # 当前 z 向量（单位长度）
        # z_axis = torch.tensor([0.0, 0.0, 0.1], device=self.device).expand(self.num_envs, 3)  # 线长0.1
        # ee_z_vec = quat_apply(self.ee_quat, z_axis)

        # # z轴末端点
        # ee_z_end = self.ee_pos + ee_z_vec

        # # 更新 marker：末端z轴
        # self.ee_z_axis_marker.visualize(self.ee_pos + self.scene.env_origins, self.ee_quat)
        # 可视化目标线（末端 → 物体）
        # self.ee_to_obj_marker.visualize(self.ee_pos, self.object_pos + self.scene.env_origins)
        return observations

    def _get_rewards(self) -> torch.Tensor:
        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_rot,
            self.ee_pos,
            self.ee_quat,
            self.fingertip_pos,
            self.in_hand_pos,
            self.goal_rot,
            self.goal_pos,
            self.cfg.align_reward_scale,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.actions,
            self.cfg.action_penalty_scale,
            self.cfg.success_tolerance,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.av_factor,
        )

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        # reset goals if the goal has been reached
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)
            # self._reset_idx(goal_env_ids)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # reset when cube has fallen
        goal_dist = torch.norm(self.ee_pos - self.in_hand_pos, p=2, dim=-1)
        out_of_reach = goal_dist >= self.cfg.fall_dist

        if self.cfg.max_consecutive_success > 0:
            # Reset progress (episode length buf) on goal envs if max_consecutive_success > 0
            move_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
            self.episode_length_buf = torch.where(
                move_dist >= self.cfg.success_tolerance,
                torch.zeros_like(self.episode_length_buf),
                self.episode_length_buf,
            )
            max_success_reached = self.successes >= self.cfg.max_consecutive_success

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.max_consecutive_success > 0:
            time_out = time_out | max_success_reached
        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # reset goals
        # self._reset_target_pose(env_ids)

        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        # global object positions
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

        arm_default_pos = torch.tensor(
            [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8], device=self.device  # panda_arm 常见初始位姿
        ).unsqueeze(0).repeat(len(env_ids), 1)

        arm_pos_noise = sample_uniform(-0.05, 0.05, (len(env_ids), len(self.arm_joint_ids)), device=self.device)
        dof_pos_arm = arm_default_pos + arm_pos_noise * self.cfg.reset_dof_pos_noise


        # reset hand
        # delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        # delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        finger_delta_max = self.hand_dof_upper_limits[env_ids][:, self.finger_joint_ids] - self.hand.data.default_joint_pos[env_ids][:, self.finger_joint_ids]
        finger_delta_min = self.hand_dof_lower_limits[env_ids][:, self.finger_joint_ids] - self.hand.data.default_joint_pos[env_ids][:, self.finger_joint_ids]
        finger_noise = sample_uniform(-1.0, 1.0, (len(env_ids), len(self.finger_joint_ids)), device=self.device)
        finger_rand_delta = finger_delta_min + (finger_delta_max - finger_delta_min) * 0.5 * finger_noise
        dof_pos_finger = self.hand.data.default_joint_pos[env_ids][:, self.finger_joint_ids] + self.cfg.reset_dof_pos_noise * finger_rand_delta
        dof_pos = torch.zeros((len(env_ids), self.num_hand_dofs), device=self.device)
        dof_pos[:, self.arm_joint_ids] = dof_pos_arm
        dof_pos[:, self.finger_joint_ids] = dof_pos_finger

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self.successes[env_ids] = 0
        self._compute_intermediate_values()

    def _reset_target_pose(self, env_ids):
        # reset goal rotation
        pos_x = sample_uniform(0.4, 0.6, (len(env_ids), 1), device=self.device)
        pos_y = sample_uniform(-0.25, 0.25, (len(env_ids), 1), device=self.device)
        pos_z = sample_uniform(0.25, 0.5, (len(env_ids), 1), device=self.device)

        new_goal_pos = torch.cat([pos_x, pos_y, pos_z], dim=-1)  # shape: (N, 3)

        # 2. 加上各自的 env origin
        self.goal_pos[env_ids] = new_goal_pos
        # self.goal_pos = self.goal_pos[env_ids] + self.scene.env_origins[env_ids]
        # rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        # new_rot = randomize_rotation(
        #     rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        # )

        # # update goal pose and markers
        # self.in_hand_pos[env_ids] = self.object_pos[env_ids] + self.scene.env_origins[env_ids]
        # self.goal_rot[env_ids] = new_rot
        # goal_pos = self.goal_pos + self.scene.env_origins
        # self.goal_markers.visualize(goal_pos, self.goal_rot)

        self.reset_goal_buf[env_ids] = 0

    def _compute_intermediate_values(self):
        # data for hand
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        # data for object
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w
            # ✅ 末端执行器 pose
        ee_pose = self.hand.data.body_state_w[:, self.ee_body_id, 0:7]
        self.ee_pos = ee_pose[:, 0:3] - self.scene.env_origins
        self.ee_quat = ee_pose[:, 3:7]
        finger_pose = self.hand.data.body_state_w[:, self.finger_bodies[0], 0:7]
        self.finger_pos = finger_pose[:, 0:3] - self.scene.env_origins
        

    def compute_reduced_observations(self):
        # Per https://arxiv.org/pdf/1808.00177.pdf Table 2
        #   Fingertip positions
        #   Object Position, but not orientation
        #   Relative target orientation
        obs = torch.cat(
            (
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.object_pos,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                self.actions,
            ),
            dim=-1,
        )

        return obs

    def compute_full_observations(self):
        ee_pose_w = self.hand.data.body_state_w[:, self.ee_body_id, 0:7]
        ee_pos = ee_pose_w[:, 0:3]
        ee_quat = ee_pose_w[:, 3:7]
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                ee_pos,
                ee_quat,
                # object
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                self.object_pos - ee_pos,                     # 物体-末端
                # goal
                self.goal_pos,
                self.goal_rot,
                # quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                self.object_pos - self.goal_pos,
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return obs

    def compute_full_state(self):
        states = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # object
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                # goal
                self.in_hand_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                self.cfg.force_torque_obs_scale
                * self.fingertip_force_sensors.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return states


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention


@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    max_episode_length: float,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    finger_pos: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    goal_pos: torch.Tensor,
    align_reward_scale: float,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions: torch.Tensor,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,
):


    goal_dist = torch.norm(ee_pos - object_pos, dim=-1) 
    # goal_dist1 = torch.norm(finger_pos - object_pos, dim=-1)  # 物体-指尖距离
    reach_rew = goal_dist * -10.0 # 可调权重
 
    action_penalty = torch.sum(actions**2, dim=-1)
    move_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    move_reward = move_dist
    lift_dist  = torch.norm(goal_pos - object_pos, p=2, dim=-1)
    lift_reward = 2.0 * (1.0 - torch.tanh(lift_dist / 0.1))
    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    # reward = dist_rew + rot_rew + reach_rew +z_align_reward + action_penalty * action_penalty_scale
    reward = reach_rew + move_reward + lift_reward + action_penalty * action_penalty_scale
    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(lift_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes
