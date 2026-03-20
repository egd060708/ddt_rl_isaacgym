from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# config
from configs.d1.d1_flat_config import D1Flat, D1FlatCfg, D1FlatCfgPPO
from algorithm.datasets.motion_loader import AMPLoader
import glob
MOTION_FILES = glob.glob('resources/d1/datasets/all/*')

class D1AMPFlat(D1Flat):
    def _init_buffers(self):
        super()._init_buffers()
        self.foot_joint_mask = torch.ones(16, dtype=torch.bool, device=self.device)
        self.foot_joint_mask[self.foot_joint_indices] = False

        self.amp_min_std_limit = self.dof_pos_limits.clone()
        self.amp_min_std_limit[self.foot_joint_indices, 1] = self.torque_limits[self.foot_joint_indices] / 10.
        self.amp_min_std_limit[self.foot_joint_indices, 0] = -self.torque_limits[self.foot_joint_indices] / 10.

    def __init__(self, cfg: D1FlatCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
    
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if self.cfg.env.reference_state_initialization:
            self.amp_loader = AMPLoader(motion_files=self.cfg.env.amp_motion_files, device=self.device, time_between_frames=self.dt)

    def _get_feet_local_pos_vel(self):
        N = self.num_envs
        F = len(self.feet_indices)

        foot_pos_rel = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        foot_vel_rel = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)

        coriolis = torch.cross(
            self.base_ang_vel.unsqueeze(1),
            foot_pos_rel,
            dim=2
        )

        # flatten
        foot_pos_flat = foot_pos_rel.view(-1, 3)
        foot_vel_flat = (foot_vel_rel - coriolis).view(-1, 3)
        base_quat_flat = self.base_quat.repeat_interleave(F, dim=0)

        foot_pos_body = quat_rotate_inverse(base_quat_flat, foot_pos_flat)
        foot_vel_body = quat_rotate_inverse(base_quat_flat, foot_vel_flat)

        return (
            foot_pos_body.view(N, -1),
            foot_vel_body.view(N, -1)
        )

    def get_amp_observations(self):
        joint_pos = self.dof_pos[:,self.foot_joint_mask]
        foot_pos, foot_vel = self._get_feet_local_pos_vel()
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.dof_vel
        # joint_vel = self.dof_vel[:,self.foot_joint_mask]
        z_pos = self.root_states[:, 2:3]
        return torch.cat((joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos), dim=-1)

    def _reset_dofs_amp(self, env_ids, frames):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
            frames: AMP frames to initialize motion with
        """
        self.dof_pos[env_ids] = AMPLoader.get_joint_pose_batch(frames)
        self.dof_vel[env_ids] = AMPLoader.get_joint_vel_batch(frames)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_amp(self, env_ids, frames):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        root_pos = AMPLoader.get_root_pos_batch(frames)
        root_pos[:, :2] = root_pos[:, :2] + self.env_origins[env_ids, :2]
        self.root_states[env_ids, :3] = root_pos
        root_orn = AMPLoader.get_root_rot_batch(frames)

        self.root_states[env_ids, 3:7] = root_orn
        self.root_states[env_ids, 7:10] = quat_rotate(root_orn, AMPLoader.get_linear_vel_batch(frames))
        self.root_states[env_ids, 10:13] = quat_rotate(root_orn, AMPLoader.get_angular_vel_batch(frames))

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]

        #self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
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
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, _, _, _, _, _, _, _= self.step(
            torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs
    
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        # actions = self.reindex(actions)
        actions = actions.to(self.device)

        self.global_counter += 1   
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.dof_pos[:, self.foot_joint_indices]  = 0  # zero position of wheels 
        reset_env_ids, terminal_amp_states = self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.cost_buf, self.reset_buf, self.extras, reset_env_ids, terminal_amp_states
    
    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self._update_command_curriculum(env_ids)

        # reset robot states
        if self.cfg.env.reference_state_initialization:
            frames = self.amp_loader.get_full_frame_batch(len(env_ids))
            self._reset_dofs_amp(env_ids, frames)
            self._reset_root_states_amp(env_ids, frames)
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        for key in self.cost_episode_sums.keys():
            self.extras["episode"]['cost_'+ key] = torch.mean(self.cost_episode_sums[key][env_ids]) / self.max_episode_length_s
            self.cost_episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # for i in range(len(self.lag_buffer)):
        #     self.lag_buffer[i][env_ids, :] = 0
        self.lag_buffer[env_ids,:,:] = 0

class D1AMPFlatCfg(D1FlatCfg):
    class env(D1FlatCfg.env):
        reference_state_initialization = False
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES
        
    class commands( D1FlatCfg.commands ):
        curriculum = False 
        max_curriculum = 3.0
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        global_reference = False
        zero_min_cmd = False

        class ranges:
            lin_vel_x = [-1.5, 1.5]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards( D1FlatCfg.rewards ):
        class scales( D1FlatCfg.rewards.scales ):
            torques = 0.0
            powers = 0.0#-2e-5
            termination = 0.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            lin_vel_z = -2.0
            orientation = -1.0
            orientation_y = -10.0
            ang_vel_xy = -0.05
            # ang_vel_y = -1.0 # avoid flipping
            dof_pos_limits = -10.0
            dof_vel = 0.0
            dof_acc = -2.5e-7
            base_height = -1.0
            feet_air_time = 0.
            collision = -1.0
            feet_stumble = 0.0
            action_rate = -0.01
            # action_smoothness= -0.01
            # foot_mirror = -0.05
            # hip_pos = 0.5
            upward = 0.5
            # feet_all_contact = -0.5
            # feet_contact_forces = -0.1
            # joint_power=-2e-5
            # powers_dist =-1.0e-5
        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.45
        max_contact_force = 500.  # forces above this value are penalized
    
    class costs(D1FlatCfg.costs):
        num_costs = 5
        class scales:
            pos_limit = 1.0
            torque_limit = 1.0
            dof_vel_limits = 1.0
            hip_pos = 2.0
            default_joint= 0.2

        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0
            hip_pos = 0.0
            default_joint = 0.0

class D1AMPFlatCfg_Play(D1AMPFlatCfg):
    class env(D1AMPFlatCfg.env):
        num_envs = 10
    class terrain(D1AMPFlatCfg.terrain):
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
        num_rows = 5
        num_cols = 5
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # terrain_proportions = [0, 0, 0, 0, 0, 0, 0]
        curriculum = False
        # selected = True  # select a unique terrain type and pass all arguments
        # terrain_kwargs = {
        #     "type": "pit_terrain",  
        #     "depth": 0.5,                     
        #     "platform_size": 4.0               
        # } # Dict of arguments for selected terrain
    class noise( D1AMPFlatCfg.noise ):
        add_noise = False
    class control ( D1AMPFlatCfg.control ):
        use_filter = True
    class domain_rand( D1AMPFlatCfg.domain_rand ):
        push_robots = False
        randomize_friction = False
        randomize_base_com = False
        randomize_base_mass = False
        randomize_motor = False
        randomize_lag_timesteps = False
        randomize_friction = False
        randomize_restitution = False
        disturbance = False
        randomize_kpkd = False
    class commands( D1AMPFlatCfg.commands ):
        heading_command = False  # if true: compute ang vel command from heading error
        resampling_time = 2.
        class ranges:
            lin_vel_x = [0.0, 0.0]  # min max [m/s]
            lin_vel_y = [0.1, 0.1]  # min max [m/s]
            ang_vel_yaw = [-0, 0]  # min max [rad/s]
            heading = [-0.0, 0.0]

class D1AMPFlatCfgPPO(D1FlatCfgPPO):
    class algorithm( D1FlatCfgPPO.algorithm ):
        amp_replay_buffer_size = 1000000

    class runner( D1FlatCfgPPO.runner ):
        run_name = ''
        experiment_name = 'd1_amp_flat'
        policy_class_name = 'ActorCriticBarlowTwins'
        # policy_class_name = 'ActorCriticTransBarlowTwins'
        runner_class_name = 'AMPOnConstraintPolicyRunner'
        algorithm_class_name = 'AMPNP3O'
        max_iterations = 6000
        num_steps_per_env = 24
        resume = False
        resume_path = ''

        amp_reward_coef = 0.005
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.5
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.05, 0.02, 0.05, 0.1] * 4