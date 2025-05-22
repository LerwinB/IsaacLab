# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.usd

# from Isaac Sim 4.2 onwards, pxr.Semantics is deprecated
try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate, quat_apply
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms, combine_frame_transforms
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz
from . import factory_control as fc
import isaacsim.core.utils.torch as torch_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.sim as sim_utils
from ..franka_hand.feature_extractor import FeatureExtractor, FeatureExtractorCfg
from isaaclab_tasks.direct.franka_hand.franka_panda_env_cfg import FrankaPandaEnvCfg
import imageio

if TYPE_CHECKING:
    from isaaclab_tasks.direct.franka_hand.franka_hand_env_cfg import FrankaHandEnvCfg
    from isaaclab_tasks.direct.franka_hand.franka_panda_env_cfg import FrankaPandaEnvCfg

@configclass
class FrankaPandaVisionEnvCfg(FrankaPandaEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1225, env_spacing=2.0, replicate_physics=True)

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(1.0, 0, 1.5), rot=(0.573576, 0.0, 0.819152, 0.0), convention="world"),
        data_types=["rgb", "depth", "semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=27.0, focus_distance=1000.0, horizontal_aperture=20.955, clipping_range=(0.1, 5.0)
        ),
        width=120,
        height=120,
    )
    feature_extractor = FeatureExtractorCfg()

    # env
    observation_space = 109  # state observation + vision CNN embedding
    state_space = 114  # asymettric states + vision CNN embedding


@configclass
class FrankaPandaVisionEnvPlayCfg(FrankaPandaVisionEnvCfg):
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=2.0, replicate_physics=True)
    # inference for CNN
    feature_extractor = FeatureExtractorCfg(train=False, load_checkpoint=True)



class PandaGraspVisionEnv(DirectRLEnv):
    cfg: FrankaPandaVisionEnvCfg

    def __init__(self, cfg: FrankaPandaVisionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.feature_extractor = FeatureExtractor(self.cfg.feature_extractor, self.device)
        self.num_hand_dofs = self.hand.num_joints#27
        self.gt_keypoints = torch.ones(self.num_envs, 8, 3, dtype=torch.float32, device=self.device)
        self.goal_keypoints = torch.ones(self.num_envs, 8, 3, dtype=torch.float32, device=self.device)

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
        self.ee_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.ee_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.ee_pos_target = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.ee_reach_target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self.initial_object_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

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
        self.target_marker = VisualizationMarkers(marker_cfg)
        self.goal_markers = VisualizationMarkers(marker_cfg)

        self.offset_local = torch.tensor([0.0, 0.0, 0.1034], device=self.device)  # 例如 +0.1m

        self.episode_count_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.prev_gripper_closed = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)



    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        stage = omni.usd.get_context().get_stage()
        # add semantics for in-hand cube
        prim = stage.GetPrimAtPath("/World/envs/env_0/object")
        sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr()
        sem.CreateSemanticDataAttr()
        sem.GetSemanticTypeAttr().Set("class")
        sem.GetSemanticDataAttr().Set("cube")
        # # add ground plane
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=300.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.prev_object_pos = self.object_pos.clone()
        self.prev_ee_pos = self.ee_pos_target.clone()
        self.prev_gripper_pos = self.gripper_joint_pos.clone()

        # self.actions = torch.zeros_like(actions)

    def _apply_action(self) -> None:
        # --- 解码动作 ---
        pos_actions = self.actions[:, 0:3]
        rot_actions = self.actions[:, 3:6]
        gripper_act = self.actions[:, -1]

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

        # --- 控制夹爪 ---
        
            
        aligned = torch.norm(self.ee_pos_target - self.object_pos,dim=-1) < 0.01
        gripper_act_now = self.actions[:, -1]  # policy输出
        closing_threshold = 0.5
        opening_threshold = -0.5

        # self.prev_gripper_closed 必须初始化过（在env初始化时，全0或全1）
        # 比如 self.prev_gripper_closed = torch.zeros((num_envs,), dtype=torch.bool, device=device)
        prev_gripper_closed = self.prev_gripper_closed

        # 滞后判断：根据之前是开/关，使用不同的 threshold
        gripper_closed_now = torch.where(
            prev_gripper_closed,
            gripper_act_now < opening_threshold,  # was closed，保持关，除非特别开
            gripper_act_now > closing_threshold   # was open，需要超过threshold才关
        )
        self.prev_gripper_closed = gripper_closed_now  # 更新状态


        open_pos = 0.04
        close_pos = 0.0
        use_hardcode = (self.episode_count_buf > 5) & (self.episode_count_buf < 8)

        # 先计算两种gripper_target
        gripper_target_hardcode = torch.where(
            aligned.unsqueeze(-1),
            torch.full_like(gripper_closed_now.unsqueeze(-1), close_pos, dtype=torch.float),
            torch.full_like(gripper_closed_now.unsqueeze(-1), open_pos, dtype=torch.float)
        )

        gripper_target_policy = torch.where(
            gripper_closed_now.unsqueeze(-1),
            torch.full_like(gripper_closed_now.unsqueeze(-1), close_pos, dtype=torch.float),
            torch.full_like(gripper_closed_now.unsqueeze(-1), open_pos, dtype=torch.float)
        )

        # 选择不同来源的 gripper_target
        gripper_target = torch.where(
            use_hardcode.unsqueeze(-1),  # broadcast到 [num_envs, 1]
            gripper_target_hardcode,
            gripper_target_policy
        )

        # 同样控制 action（注意，只影响 gripper维度）
        # imitation阶段 action取 aligned，探索阶段 action取 policy输出
        self.actions[:, -1] = torch.where(
            use_hardcode,
            aligned.float() * 2.0 - 1.0,       # imitation时直接根据 aligned
            self.actions[:, -1]  # 探索时根据 policy输出 (也可以直接 gripper_close)
        )
        gripper_target_all = gripper_target.repeat(1, 2)
        self.hand.set_joint_position_target(gripper_target_all, joint_ids=self.actuated_dof_indices)
        # 指尖位置
        # offset_world = quat_apply(ee_quat, self.offset_local.expand(ee_quat.shape[0], 3)) 
        # ee_pos_target = ee_pos+ offset_world
        ee_target, ee_quat_target = combine_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            self.ee_pos_target, self.ee_quat
        )
        # self.ee_marker.visualize(ee_target, ee_quat_target,
        #     marker_indices=torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        # )
        # object_pos,object_quat = combine_frame_transforms(
        #     root_pose_w[:, 0:3], root_pose_w[:, 3:7],
        #     self.object_pos, self.ee_reach_target_quat)
        # self.target_marker.visualize(object_pos, object_quat,
        #     marker_indices=torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        # )
        # self.goal_markers.visualize(self.goal_pos + self.scene.env_origins, self.goal_rot,
        #     marker_indices=torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        # )

    def _compute_states(self):
        """Asymmetric states for the critic."""
        sim_states = self.compute_full_state()
        state = torch.cat((sim_states, self.embeddings), dim=-1)
        return state
    
    def _get_observations(self) -> dict:
        state_obs = self._compute_proprio_observations()
        # vision observations from CMM
        image_obs = self._compute_image_observations()
        obs = torch.cat((state_obs, image_obs), dim=-1)
        # asymmetric critic states
        self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[:, self.finger_bodies]
        state = self._compute_states()

        observations = {"policy": obs, "critic": state}

        # rgb=self._tiled_camera.data.output["rgb"][0].cpu().numpy()
        # imageio.imwrite("camera_view.png", (rgb * 255).astype("uint8"))
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
            reward_terms,
        ) = compute_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_rot,
            self.ee_reach_target_quat,
            self.gripper_joint_pos,
            self.prev_ee_pos,
            self.prev_object_pos,
            self.prev_gripper_pos,
            self.ee_pos,
            self.ee_quat,
            self.ee_pos_target,
            self.fingertip_pos,
            self.initial_object_pos,
            self.goal_rot,
            self.goal_pos,
            self.cfg.align_reward_scale,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.episode_length_buf,
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
        self.extras["log"]["mean_episode_count"] = self.episode_count_buf.float().mean()

        for key, value in reward_terms.items():
            self.extras["log"][f"reward/{key}"] = value.mean()
        # reset goals if the goal has been reached
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)
            # self._reset_idx(goal_env_ids)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # reset when cube has fallen
        goal_dist = torch.norm(self.object_pos - self.goal_pos, p=2, dim=-1)
        out_of_reach = goal_dist >= self.cfg.fall_dist

        if self.cfg.max_consecutive_success > 0:
            # Reset progress (episode length buf) on goal envs if max_consecutive_success > 0
            move_dist = torch.norm(self.object_pos - self.ee_pos, p=2, dim=-1)
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
        self.episode_count_buf[env_ids] += 1
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
        # 添加一个角速度扰动：每个维度在 [-0.5, 0.5] 范围内
        ang_vel_noise = 0.5 * (2.0 * torch.rand((len(env_ids), 3), device=self.device) - 1.0)

        # 设置 linear velocity 为 0，angular velocity 为随机值
        object_default_state[:, 10:13] = ang_vel_noise  # angular velocity

        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)
        # 获取当前物体重置后的初始位置（world 坐标）
        # self.initial_object_pos = self.object.data.root_pos_w.clone() - self.scene.env_origins  # [num_envs, 3]
        self.initial_object_pos[env_ids] = object_default_state[:, 0:3] - self.scene.env_origins[env_ids]

        pos_x = sample_uniform(0.4, 0.6, (len(env_ids), 1), device=self.device)
        pos_y = sample_uniform(-0.25, 0.25, (len(env_ids), 1), device=self.device)
        pos_z = sample_uniform(0.25, 0.5, (len(env_ids), 1), device=self.device)

        new_goal_pos = torch.cat([pos_x, pos_y, pos_z], dim=-1)  # shape: (N, 3)

        # 2. 加上各自的 env origin
        self.goal_pos[env_ids] = new_goal_pos


        arm_default_pos = torch.tensor(
            [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8], device=self.device  # panda_arm 常见初始位姿
        ).unsqueeze(0).repeat(len(env_ids), 1)

        arm_pos_noise = sample_uniform(-0.05, 0.05, (len(env_ids), len(self.arm_joint_ids)), device=self.device)
        dof_pos_arm = arm_default_pos + arm_pos_noise * self.cfg.reset_dof_pos_noise


        # reset hand
        # delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        # delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        dof_pos = torch.zeros((len(env_ids), self.num_hand_dofs), device=self.device)
        dof_pos[:, self.arm_joint_ids] = dof_pos_arm
        # dof_pos[:, self.finger_joint_ids] = dof_pos_finger
        gripper_open_pos = torch.tensor([0.035, 0.035], device=self.device).unsqueeze(0).repeat(len(env_ids), 1)
        dof_pos[:, self.finger_joint_ids] = gripper_open_pos

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
        offset_world = quat_apply(self.ee_quat, self.offset_local.expand(self.ee_quat.shape[0], 3)) 
        self.ee_pos_target = self.ee_pos+ offset_world
        finger_pose = self.hand.data.body_state_w[:, self.finger_bodies[0], 0:7]
        self.finger_pos = finger_pose[:, 0:3] - self.scene.env_origins
        self.gripper_joint_pos = self.hand.data.joint_pos[:, self.finger_joint_ids]
        self.ee_reach_target_quat=self.align_gripper_y_to_nearest_cube_face(self.ee_quat, self.object_rot)

    def _compute_image_observations(self):
        # generate ground truth keypoints for in-hand cube
        compute_keypoints(pose=torch.cat((self.object_pos, self.object_rot), dim=1), out=self.gt_keypoints)

        object_pose = torch.cat([self.object_pos, self.gt_keypoints.view(-1, 24)], dim=-1)

        # train CNN to regress on keypoint positions
        pose_loss, embeddings = self.feature_extractor.step(
            self._tiled_camera.data.output["rgb"],
            self._tiled_camera.data.output["depth"],
            self._tiled_camera.data.output["semantic_segmentation"][..., :3],
            object_pose,
        )

        self.embeddings = embeddings.clone().detach()
        # compute keypoints for goal cube
        compute_keypoints(
            pose=torch.cat((torch.zeros_like(self.goal_pos), self.goal_rot), dim=-1), out=self.goal_keypoints
        )

        obs = torch.cat(
            (
                self.embeddings,
                self.goal_keypoints.view(-1, 24),
            ),
            dim=-1,
        )

        # log pose loss from CNN training
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["pose_loss"] = pose_loss

        return obs

    def _compute_proprio_observations(self):
        """Proprioception observations from physics."""
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # goal
                self.in_hand_pos,
                self.goal_rot,
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
        # ee_pose_w = self.hand.data.body_state_w[:, self.ee_body_id, 0:7]
        # ee_pos = ee_pose_w[:, 0:3]
        # ee_quat = ee_pose_w[:, 3:7]
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                self.ee_pos,
                self.ee_quat,
                # object
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                self.object_pos - self.ee_pos_target,                     # 物体-末端
                self.ee_reach_target_quat,
                # goal
                self.goal_pos,
                self.goal_rot,
                quat_mul(quat_conjugate(self.ee_reach_target_quat), self.ee_quat),
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
                self.goal_pos,
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
    
    def align_gripper_y_to_nearest_cube_face(self, ee_quat: torch.Tensor, object_quat: torch.Tensor) -> torch.Tensor:
        """
        给定当前末端姿态 ee_quat 和物体姿态 object_quat，
        自动选择立方体 6 个面中与末端 y 轴最接近的一个方向对齐。

        Args:
            ee_quat: [N, 4] 当前末端姿态（四元数）
            object_quat: [N, 4] 目标物体姿态

        Returns:
            ee_quat_target: [N, 4] 对齐后的姿态（夹爪 y 轴朝向立方体最近一个面）
        """
        device = ee_quat.device
        batch_size = ee_quat.shape[0]

        # Step 1: 立方体局部坐标下的 6 个面法向
        face_normals_local = torch.tensor([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        ], dtype=torch.float32, device=device)  # [6, 3]
        face_normals_local_batch = face_normals_local.unsqueeze(0).expand(batch_size, -1, -1)
        # Step 2: 将法向量转换为世界坐标（使用 object_quat）
        object_quat_batch = object_quat.unsqueeze(1).expand(-1, 6, -1)
        face_normals_world = quat_apply(object_quat_batch, face_normals_local_batch)  # [N, 6, 3]

        # Step 3: 当前末端姿态下的 y 轴方向（夹爪开合方向）
        y_axis = torch.tensor([0.0, 1.0, 0.0], device=device).repeat(batch_size, 1)  # [N, 3]
        ee_y = quat_apply(ee_quat, y_axis).unsqueeze(1)  # [N, 1, 3]

        # Step 4: 找到最接近的 face normal
        dot = torch.sum(ee_y * face_normals_world, dim=-1)  # [N, 6]
        best_idx = torch.argmax(torch.abs(dot), dim=-1)     # [N]

        # Step 5: 获取最佳目标方向
        target_y = face_normals_world[torch.arange(batch_size), best_idx]  # [N, 3]

        # Step 6: 构造最小旋转，将当前 y 轴对齐到 target_y
        q_align = self.quat_from_to(ee_y.squeeze(1), target_y)  # [N, 4]
        ee_quat_target = quat_mul(q_align, ee_quat)        # [N, 4]

        return ee_quat_target

    def quat_from_to(self, v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        返回将向量 v1 旋转到 v2 的四元数，使用最小旋转。
        
        Args:
            v1: [N, 3] 源向量（应单位化）
            v2: [N, 3] 目标向量（应单位化）
        
        Returns:
            quat: [N, 4] 四元数 (w, x, y, z)
        """
        v1 = torch.nn.functional.normalize(v1, dim=-1)
        v2 = torch.nn.functional.normalize(v2, dim=-1)
        
        cross = torch.cross(v1, v2, dim=-1)  # [N, 3]
        dot = torch.sum(v1 * v2, dim=-1, keepdim=True)  # [N, 1]

        w = dot + 1.0
        near_zero = w < eps  # [N, 1]

        # 处理 v1 ≈ -v2 的情况（夹角 180°，无法直接用 cross）
        alt_axis = torch.tensor([1.0, 0.0, 0.0], device=v1.device).expand_as(v1)
        alt_cross = torch.cross(v1, alt_axis, dim=-1)
        alt_cross = torch.where(
            torch.norm(alt_cross, dim=-1, keepdim=True) < eps,
            torch.cross(v1, torch.tensor([0.0, 1.0, 0.0], device=v1.device).expand_as(v1), dim=-1),
            alt_cross,
        )
        quat_main = torch.cat([w, cross], dim=-1)          # [N, 4]
        quat_fallback = torch.cat([torch.zeros_like(w), alt_cross], dim=-1)  # [N, 4]
        quat = torch.where(near_zero.expand_as(quat_main), quat_fallback, quat_main)
        return torch.nn.functional.normalize(quat, dim=-1)


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
    ee_reach_target_quat: torch.Tensor,
    gripper_joint_pos: torch.Tensor,
    prev_ee_pos: torch.Tensor,
    prev_object_pos: torch.Tensor,
    prev_gripper_pos: torch.Tensor,
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,  
    ee_pos_target: torch.Tensor,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    finger_pos: torch.Tensor,
    inital_object_pos: torch.Tensor,
    target_rot: torch.Tensor,
    goal_pos: torch.Tensor,
    align_reward_scale: float,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    progress_step: torch.Tensor,
    actions: torch.Tensor,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,
):
    # offset_local = torch.tensor([0.0, 0.0, 0.1034], device=ee_quat.device).repeat(ee_quat.shape[0], 1)
    # offset_world = quat_apply(ee_quat, offset_local)
    # gripper_pos = ee_pos + offset_world
    goal_dist = torch.norm(ee_pos_target - object_pos, dim=-1) 
    # goal_dist1 = torch.norm(finger_pos - object_pos, dim=-1)  # 物体-指尖距离
    reach_rew = goal_dist * -10.0 # 可调权重
    # reach_rew = 5.0 * torch.exp(-goal_dist / 0.05)
    # reach_rew = 5.0 * torch.exp(-((goal_dist / 0.05) ** 2))
    reach_rew1 = 10.0 / (1.0 + 50.0 * goal_dist)

    q_diff = quat_mul(quat_conjugate(ee_reach_target_quat), ee_quat)
    angle_diff = 2 * torch.acos(torch.clamp(q_diff[:, 0], -1.0, 1.0))
    # rot_reward = -10.0 * torch.tanh(angle_diff)
    sigma = 0.3  # 弧度，约 17°
    rot_reward = 2.0 * torch.exp(-angle_diff**2 / (2 * sigma**2))   

    alignment_thresh = 0.02  # 可调，比如 2cm

    aligned_pos = torch.norm(ee_pos_target - object_pos, dim=-1) < alignment_thresh  # [N] bool
    aligned_rot = angle_diff < 0.3
    aligned = aligned_pos & aligned_rot

    action_penalty = torch.sum(actions**2, dim=-1)
    move_dist = torch.norm(object_pos - inital_object_pos, p=2, dim=-1)
    # move_reward = move_dist
    move_reward = 5.0 * torch.tanh(move_dist / 0.01) #距离 ~0.1m 时趋于 5.0
    lift_dist  = torch.norm(goal_pos - object_pos, p=2, dim=-1)
    lift_reward = 50.0 * (1.0 - torch.tanh(lift_dist / 0.1))
    
    
    offset_now = object_pos - ee_pos_target
    offset_prev = prev_object_pos - prev_ee_pos
    offset_diff = torch.norm(offset_now - offset_prev, dim=-1)
    grabbed_by_offset = offset_diff < 0.001
    aperture_now = torch.sum(gripper_joint_pos, dim=-1)
    aperture_prev = torch.sum(prev_gripper_pos, dim=-1)
    aperture_change = torch.abs(aperture_now - aperture_prev)
    grabbed_by_aperture = aperture_change < 0.001

    within_grab_range = (aperture_now > 0.02) & (aperture_now < 0.06) &(move_dist > 0.01)
    # grabbed = aligned & grabbed_by_offset & grabbed_by_aperture & within_grab_range
    grabbed = aligned & within_grab_range & grabbed_by_aperture & grabbed_by_offset
    # grab_reward = grabbed.float() * 5

    close_enough = goal_dist < 0.001
    gripper_open = ~((aperture_now > 0.02) & (aperture_now < 0.06))
    # no_movement = torch.norm(object_pos - inital_object_pos, dim=-1) < 0.01
    near_and_trying = aligned & (aperture_change > 0.01)
    near_but_not_grabbing = aligned & (~grabbed)
    penalty_scale = 0.01 * progress_step
    ngrab_reward = penalty_scale * near_but_not_grabbing.float() * -1.0
    # near_bnot_aligned = ~aligned & (goal_dist < 0.05)
    # align_reward = penalty_scale * aligned.float() * 1.0
    gripper_act_change = torch.abs(aperture_now - aperture_prev)
    gripper_action_rew = 1.0 * gripper_act_change
    grab_bnot_aligned = ~aligned & (aperture_now < 0.06)
    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    # reward = dist_rew + rot_rew + reach_rew +z_align_reward + action_penalty * action_penalty_scale
    reward = reach_rew +reach_rew1+ rot_reward+move_reward+ lift_reward +ngrab_reward+action_penalty * action_penalty_scale
    # no_movement = torch.norm(object_pos - prev_object_pos, dim=-1) < 1e-4
    reward = torch.where(grab_bnot_aligned, reward - 5.0, reward)
    reward = torch.where(aligned_pos, reward + 20.0, reward)
    reward = torch.where(aligned_rot, reward + 5.0, reward)
    # reward = torch.where(no_movement, reward - 2.0, reward)
    # reward = torch.where(near_but_not_grabbing, reward - 10.0, reward)
    # reward = torch.where((~aligned) & (grabbed_by_aperture), reward - 2.0, reward)
    # grabbed bonus
    reward = torch.where(grabbed, reward+30.0, reward)
    # Find out which envs hit the goal and update successes count
    reward = torch.where(torch.abs(lift_dist) <= 0.05, reward + 50.0, reward)
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

    return reward, goal_resets, successes, cons_successes, {
    "reward": reward,
    "reach_rew": reach_rew,
    "rot_reward": rot_reward,
    "reach_rew1": reach_rew1,
    "move_reward": move_reward,
    "lift_reward": lift_reward,
    "grabbed_bonus": grabbed.float() * 30.0,
    "aligned_bonus": aligned_pos.float() * 20.0,
    "ngrab_reward": ngrab_reward, 
    # "near_but_not_grabbing_bonus": near_but_not_grabbing.float() * -5.0,
    "grab_bnot_aligned_bonus": grab_bnot_aligned.float() * -5.0,
    "action_penalty": action_penalty * action_penalty_scale,
}

@torch.jit.script
def compute_keypoints(
    pose: torch.Tensor,
    num_keypoints: int = 8,
    size: tuple[float, float, float] = (2 * 0.03, 2 * 0.03, 2 * 0.03),
    out: torch.Tensor | None = None,
):
    """Computes positions of 8 corner keypoints of a cube.

    Args:
        pose: Position and orientation of the center of the cube. Shape is (N, 7)
        num_keypoints: Number of keypoints to compute. Default = 8
        size: Length of X, Y, Z dimensions of cube. Default = [0.06, 0.06, 0.06]
        out: Buffer to store keypoints. If None, a new buffer will be created.
    """
    num_envs = pose.shape[0]
    if out is None:
        out = torch.ones(num_envs, num_keypoints, 3, dtype=torch.float32, device=pose.device)
    else:
        out[:] = 1.0
    for i in range(num_keypoints):
        # which dimensions to negate
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner_loc = ([(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],)
        corner = torch.tensor(corner_loc, dtype=torch.float32, device=pose.device) * out[:, i, :]
        # express corner position in the world frame
        out[:, i, :] = pose[:, :3] + quat_apply(pose[:, 3:7], corner)

    return out