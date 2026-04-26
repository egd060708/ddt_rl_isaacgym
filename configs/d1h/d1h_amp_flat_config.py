"""
d1h 双足 AMP 配置：支持 height40_pg_tau 数据集（包含 project_gravity 和 joint_tau）。

请将动作数据 JSON 放在 resources/d1h/datasets/height40_pg_tau/*.txt，
每帧长度 = 52（pos3+rot4+pg3+jpos8+tar_toe6+lin3+ang3+jvel8+jtau8+tar_vel6）。
AMP 观测维度为 30（使用 d1h_pg_without_wheel_pos 风格：masked_joint_pos(6)+pg(3)+foot_pos(6)+lin(3)+ang(3)+jvel(8)+z(1)）。
数据集读取全帧维度(52)与 AMP 使用的专家特征维度(30)不同，可通过 amp_observation_dim 显式指定。
使用 amp_feed_forward_style = "d1h_pg_without_wheel_pos" 并设置 amp_observation_dim = 30。
"""
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from configs.d1h.d1h_flat_config import D1HFlat, D1HFlatCfg, D1HFlatCfgPPO
from algorithm.datasets.motion_loader import AMPLoader, motion_layout_from_legged_cfg
import glob
from algorithm.wamp_discriminator import WAMPDiscriminator


# 用户放置 d1h 动作数据后自动加载；若目录为空需先创建并放入 .txt
# 支持 height35_pg (带 project_gravity) 数据集
MOTION_FILES_D1H_AMP = glob.glob("resources/d1h/datasets/height35/*.txt")
MOTION_FILES_D1H_WAMP = glob.glob("resources/d1h/datasets/height35/*.txt")

class D1HAMPFlat(D1HFlat):
    """与 D1AMPFlat 类似，适配 8 DOF 与双足 AMP 观测维度。"""

    def _init_buffers(self):
        super()._init_buffers()
        self.foot_joint_mask = torch.ones(8, dtype=torch.bool, device=self.device)
        self.foot_joint_mask[self.foot_joint_indices] = False

        self.amp_min_std_limit = self.dof_pos_limits.clone()
        self.amp_min_std_limit[self.foot_joint_indices, 1] = (
            self.torque_limits[self.foot_joint_indices] / 10.0
        )
        self.amp_min_std_limit[self.foot_joint_indices, 0] = (
            -self.torque_limits[self.foot_joint_indices] / 10.0
        )

    def __init__(self, cfg: D1HFlatCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False

        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if self.cfg.env.reference_state_initialization:
            _ml = motion_layout_from_legged_cfg(self.cfg)
            self.amp_loader = AMPLoader(
                motion_files=self.cfg.env.amp_motion_files,
                device=self.device,
                time_between_frames=self.dt,
                motion_layout=_ml if _ml else None,
            )
            
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def _get_feet_local_pos_vel(self):
        N = self.num_envs
        F = len(self.feet_indices)

        foot_pos_rel = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        foot_vel_rel = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)

        coriolis = torch.cross(
            self.base_ang_vel.unsqueeze(1),
            foot_pos_rel,
            dim=2,
        )

        foot_pos_flat = foot_pos_rel.view(-1, 3)
        foot_vel_flat = (foot_vel_rel - coriolis).view(-1, 3)
        base_quat_flat = self.base_quat.repeat_interleave(F, dim=0)

        foot_pos_body = quat_rotate_inverse(base_quat_flat, foot_pos_flat)
        foot_vel_body = quat_rotate_inverse(base_quat_flat, foot_vel_flat)

        return (
            foot_pos_body.view(N, -1),
            foot_vel_body.view(N, -1),
        )

    def get_amp_observations(self):
        """支持 d1h_pg_without_wheel_pos 风格：将 projected_gravity 加入 AMP 观测，与 height40_pg_tau 数据集匹配。
        维度固定为 30：masked_joint_pos(6) + proj_gravity(3) + foot_pos(6) + lin_vel(3) + ang_vel(3) + joint_vel(8) + root_z(1)。
        """
        style = self.cfg.env.amp_motion_layout.amp_feed_forward_style
        if style == "d1h_pg_without_wheel_pos":
            # 观测组成匹配 motion_loader._expert_features_from_full_frames 中的提取逻辑
            joint_pos = self.dof_pos[:, self.foot_joint_mask]  # 6 dims (排除 foot joints [3,7])
            foot_pos, _foot_vel = self._get_feet_local_pos_vel()  # 6 dims (2 feet * 3)
            base_lin_vel = self.base_lin_vel  # 3
            base_ang_vel = self.base_ang_vel  # 3
            proj_gravity = self.projected_gravity  # 3
            joint_vel = self.dof_vel  # 8
            z_pos = self.root_states[:, 2:3]  # 1
            amp_obs = torch.cat(
                (joint_pos, proj_gravity, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos),
                dim=-1,
            )
            return amp_obs
        elif style == "d1h_pg_without_wheel_foot_pos":
            joint_pos = self.dof_pos[:, self.foot_joint_mask]
            foot_pos, _foot_vel = self._get_feet_local_pos_vel()
            base_lin_vel = self.base_lin_vel
            base_ang_vel = self.base_ang_vel
            proj_gravity = self.projected_gravity
            joint_vel = self.dof_vel
            z_pos = self.root_states[:, 2:3]
            amp_obs = torch.cat(
                (joint_pos, proj_gravity, base_lin_vel, base_ang_vel, joint_vel, z_pos),
                dim=-1,
            )
            return amp_obs
        elif style == "d1h_without_wheel_pos":
            joint_pos = self.dof_pos[:, self.foot_joint_mask]
            foot_pos, _foot_vel = self._get_feet_local_pos_vel()
            base_lin_vel = self.base_lin_vel
            base_ang_vel = self.base_ang_vel
            joint_vel = self.dof_vel
            z_pos = self.root_states[:, 2:3]
            return torch.cat(
                (joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos),
                dim=-1,
            )
        elif style == "d1h_without_wheel_angVel":
            joint_pos = self.dof_pos[:, self.foot_joint_mask]
            foot_pos, _foot_vel = self._get_feet_local_pos_vel()
            base_lin_vel = self.base_lin_vel
            base_ang_vel = self.base_ang_vel
            joint_vel = self.dof_vel[:, self.foot_joint_mask]
            z_pos = self.root_states[:, 2:3]
            return torch.cat(
                (joint_pos, foot_pos, base_lin_vel, joint_vel, z_pos),
                dim=-1,
            )
        elif style == "d1h_without_wheelpos_angVel":
            joint_pos = self.dof_pos[:, self.foot_joint_mask]
            foot_pos, _foot_vel = self._get_feet_local_pos_vel()
            base_lin_vel = self.base_lin_vel
            base_ang_vel = self.base_ang_vel
            joint_vel = self.dof_vel
            z_pos = self.root_states[:, 2:3]
            return torch.cat(
                (joint_pos, foot_pos, base_lin_vel, joint_vel, z_pos),
                dim=-1,
            )
        elif style == "d1h_pg_tau_without_wheel_pos":
            joint_pos = self.dof_pos[:, self.foot_joint_mask]
            foot_pos, _foot_vel = self._get_feet_local_pos_vel()
            base_lin_vel = self.base_lin_vel
            base_ang_vel = self.base_ang_vel
            proj_gravity = self.projected_gravity
            joint_vel = self.dof_vel
            joint_tau = self.torques
            z_pos = self.root_states[:, 2:3]
            amp_obs = torch.cat(
                (joint_pos, proj_gravity, foot_pos, base_lin_vel, base_ang_vel, joint_vel, joint_tau, z_pos),
                dim=-1,
            )
            return amp_obs
        else:
            # 默认
            joint_pos = self.dof_pos
            foot_pos, _foot_vel = self._get_feet_local_pos_vel()
            base_lin_vel = self.base_lin_vel
            base_ang_vel = self.base_ang_vel
            joint_vel = self.dof_vel
            z_pos = self.root_states[:, 2:3]
            return torch.cat(
                (joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos),
                dim=-1,
            )

    def _reset_dofs_amp(self, env_ids, frames):
        self.dof_pos[env_ids] = self.amp_loader.get_joint_pose_batch(frames)
        self.dof_vel[env_ids] = self.amp_loader.get_joint_vel_batch(frames)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states_amp(self, env_ids, frames):
        root_pos = self.amp_loader.get_root_pos_batch(frames)
        root_pos[:, :2] = root_pos[:, :2] + self.env_origins[env_ids, :2]
        self.root_states[env_ids, :3] = root_pos
        root_orn = self.amp_loader.get_root_rot_batch(frames)
        self.root_states[env_ids, 3:7] = root_orn
        self.root_states[env_ids, 7:10] = quat_rotate(
            root_orn, self.amp_loader.get_linear_vel_batch(frames)
        )
        self.root_states[env_ids, 10:13] = quat_rotate(
            root_orn, self.amp_loader.get_angular_vel_batch(frames)
        )
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 7:10]
        )
        self.base_ang_vel[:] = quat_rotate_inverse(
            self.base_quat, self.root_states[:, 10:13]
        )
        self.projected_gravity[:] = quat_rotate_inverse(
            self.base_quat, self.gravity_vec
        )

        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]

        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact

        self._post_physics_step_callback()

        self.check_termination()
        self.compute_reward()
        self.compute_cost()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_amp_states = self.get_amp_observations()[env_ids]
        self.reset_idx(env_ids)
        self.compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, terminal_amp_states

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        if self.cfg.env.baseline_mode:
            obs, _, _, _, _, _ = self.step(
                torch.zeros(
                    self.num_envs, self.num_actions, device=self.device, requires_grad=False
                )
            )
        else:
            obs, _, _, _, _, _, _, _ = self.step(
                torch.zeros(
                    self.num_envs, self.num_actions, device=self.device, requires_grad=False
                )
            )
        return obs

    def step(self, actions):
        self.action_history_buf = torch.cat(
            [self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1
        )
        actions = actions.to(self.device)

        self.global_counter += 1
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(self.torques)
            )
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.dof_pos[:, self.foot_joint_indices] = 0

        reset_env_ids, terminal_amp_states = self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )

        if self.cfg.env.baseline_mode:
            return (
                self.obs_buf,
                self.privileged_obs_buf,
                self.rew_buf,
                self.cost_buf,
                self.reset_buf,
                self.extras,
            )
        else:
            return (
                self.obs_buf,
                self.privileged_obs_buf,
                self.rew_buf,
                self.cost_buf,
                self.reset_buf,
                self.extras,
                reset_env_ids,
                terminal_amp_states,
            )

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        if self.cfg.commands.curriculum and (
            self.common_step_counter % self.max_episode_length == 0
        ):
            self._update_command_curriculum(env_ids)

        if self.cfg.env.reference_state_initialization:
            ref_init_prob = float(
                getattr(self.cfg.env, "reference_state_initialization_prob", 1.0)
            )
            ref_init_prob = max(0.0, min(1.0, ref_init_prob))

            if ref_init_prob >= 1.0:
                amp_env_ids = env_ids
                default_env_ids = env_ids[:0]
            elif ref_init_prob <= 0.0:
                amp_env_ids = env_ids[:0]
                default_env_ids = env_ids
            else:
                amp_mask = (
                    torch.rand(len(env_ids), device=self.device) < ref_init_prob
                )
                amp_env_ids = env_ids[amp_mask]
                default_env_ids = env_ids[~amp_mask]

            if len(amp_env_ids) > 0:
                frames = self.amp_loader.get_full_frame_batch(len(amp_env_ids))
                self._reset_dofs_amp(amp_env_ids, frames)
                self._reset_root_states_amp(amp_env_ids, frames)

            if len(default_env_ids) > 0:
                self._reset_dofs(default_env_ids)
                self._reset_root_states(default_env_ids)
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.last_torques[env_ids] = 0.0
        self.last_root_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.0
        self.contact_buf[env_ids, :, :] = 0.0
        self.action_history_buf[env_ids, :, :] = 0.0

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        for key in self.cost_episode_sums.keys():
            self.extras["episode"]["cost_" + key] = (
                torch.mean(self.cost_episode_sums[key][env_ids])
                / self.max_episode_length_s
            )
            self.cost_episode_sums[key][env_ids] = 0.0
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(
                self.terrain_levels.float()
            )
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][
                1
            ]
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.lag_buffer[env_ids, :, :] = 0


class D1HAMPFlatCfg(D1HFlatCfg):
    class env(D1HFlatCfg.env):
        reference_state_initialization = True
        reference_state_initialization_prob = 1.0
        amp_motion_files = MOTION_FILES_D1H_AMP
        baseline_mode = False

        class amp_motion_layout:
            pos_size = 3
            rot_size = 4
            # project_gravity_size = 3
            joint_pos_size = 8
            joint_vel_size = 8
            # joint_tau_size = 8
            tar_toe_pos_local_size = 6
            tar_toe_vel_local_size = 6
            linear_vel_size = 3
            angular_vel_size = 3
            amp_feed_forward_style = "d1h_without_wheel_pos"
            # 数据集读取维度(52)与 AMP 特征使用维度(30)不同，通过 amp_observation_dim 显式指定匹配 get_amp_observations 返回的维度
            # 当前风格下专家特征维度为 30，可避免 normalize_torch 中的形状不匹配 (30 vs 38)
            amp_observation_dim = 27
            
    class asset( D1HFlatCfg.asset ):
        file = '{ROOT_DIR}/resources/d1h/urdf/robot.urdf'
        foot_name = "foot"
        name = "d1h"
        penalize_contacts_on = ["thigh", "calf", "base"]
        penalize_contact_head_on = ["base"]
        # terminate_after_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False

    class commands(D1HFlatCfg.commands):
        curriculum = False
        heading_command = False


class D1HAMPFlatCfg_Play(D1HAMPFlatCfg):
    class env(D1HAMPFlatCfg.env):
        num_envs = 10

    class terrain(D1HAMPFlatCfg.terrain):
        mesh_type = "plane"
        curriculum = False

    class noise(D1HAMPFlatCfg.noise):
        add_noise = False

    class domain_rand(D1HAMPFlatCfg.domain_rand):
        push_robots = False
        randomize_friction = False
        randomize_base_com = False
        randomize_base_mass = False
        randomize_motor = False
        randomize_lag_timesteps = False
        randomize_restitution = False
        disturbance = False
        randomize_kpkd = False

    class commands( D1HAMPFlatCfg.commands ):
        heading_command = False  # if true: compute ang vel command from heading error
        resampling_time = 2.
        class ranges:
            lin_vel_x = [-1., 1.]  # min max [m/s]
            lin_vel_y = [-1., 1.]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]


class D1HAMPFlatCfgPPO(D1HFlatCfgPPO):
    class algorithm(D1HFlatCfgPPO.algorithm):
        amp_replay_buffer_size = 3000000

    class runner(D1HFlatCfgPPO.runner):
        run_name = ""
        experiment_name = "d1h_amp_flat"
        policy_class_name = "ActorCriticBarlowTwins"
        runner_class_name = "AMPOnConstraintPolicyRunner"
        algorithm_class_name = "AMPNP3O"
        max_iterations = 40000
        num_steps_per_env = 24
        resume = False
        resume_path = ""

        amp_reward_coef = 0.5
        amp_motion_files = MOTION_FILES_D1H_AMP
        amp_num_preload_transitions = 6000000
        amp_task_reward_lerp = 0.5
        amp_reward_scale = 0.25
        amp_discr_hidden_dims = [1024, 512]

        # 8 个关节，与 d1h 每条腿 4 关节对应的两组系数重复一次
        min_normalized_std = [0.05, 0.02, 0.05, 0.1] * 2


class D1HWAMPFlat(D1HAMPFlat):
    """Wasserstein Adversarial Imitation (WAMP) version for HumanMimic paper.
    Uses WGAN-GP discriminator and WAMPNP3O.
    """
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # Ensure amp_loader is initialized (WAMP uses the same motion loader)
        # self._ensure_amp_loader()


class D1HWAMPFlatCfg(D1HAMPFlatCfg):
    class env(D1HAMPFlatCfg.env):
        reference_state_initialization = True
        reference_state_initialization_prob = 1.0
        amp_motion_files = MOTION_FILES_D1H_WAMP

        class amp_motion_layout:
            pos_size = 3
            rot_size = 4
            # project_gravity_size = 3
            joint_pos_size = 8
            joint_vel_size = 8
            # joint_tau_size = 8
            tar_toe_pos_local_size = 6
            tar_toe_vel_local_size = 6
            linear_vel_size = 3
            angular_vel_size = 3
            amp_feed_forward_style = "d1h_without_wheel_pos"
            # 与 AMP 配置一致，显式设置维度 30 以匹配 env.get_amp_observations()
            amp_observation_dim = 27


class D1HWAMPFlatCfg_Play(D1HWAMPFlatCfg):
    class env(D1HWAMPFlatCfg.env):
        num_envs = 10

    class terrain(D1HWAMPFlatCfg.terrain):
        mesh_type = "plane"
        curriculum = False

    class noise(D1HWAMPFlatCfg.noise):
        add_noise = False

    class domain_rand(D1HWAMPFlatCfg.domain_rand):
        push_robots = False
        randomize_friction = False
        randomize_base_com = False
        randomize_base_mass = False
        randomize_motor = False
        randomize_lag_timesteps = False
        randomize_restitution = False
        disturbance = False
        randomize_kpkd = False

    class commands(D1HWAMPFlatCfg.commands):
        heading_command = False
        resampling_time = 2.0
        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0, 0]
            heading = [-3.14, 3.14]


class D1HWAMPFlatCfgPPO(D1HAMPFlatCfgPPO):
    class algorithm(D1HFlatCfgPPO.algorithm):
        amp_replay_buffer_size = 3000000

    class runner(D1HAMPFlatCfgPPO.runner):
        run_name = ""
        experiment_name = "d1h_wamp_flat"
        policy_class_name = "ActorCriticBarlowTwins"
        runner_class_name = "WAMPOnConstraintPolicyRunner"
        algorithm_class_name = "WAMPNP3O"
        max_iterations = 40000
        num_steps_per_env = 24
        resume = False
        resume_path = ""

        amp_reward_coef = 0.25
        amp_motion_files = MOTION_FILES_D1H_WAMP
        amp_num_preload_transitions = 6000000
        amp_task_reward_lerp = 0.5
        amp_reward_scale = 0.25
        wasserstein_lambda = 10.0
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.05, 0.02, 0.05, 0.1] * 2


