from isaacgym.torch_utils import *
import torch
# env related
from configs.base.legged_robot import LeggedRobot
from isaacgym import gymtorch

# def random_quat(U):
#     u1 = U[:,0].unsqueeze(1)
#     u2 = U[:,1].unsqueeze(1)
#     u3 = U[:,2].unsqueeze(1)
#     q1 = torch.sqrt(1-u1)*torch.sin(2*torch.pi*u2)
#     q2 = torch.sqrt(1-u1)*torch.cos(2*torch.pi*u2)
#     q3 = torch.sqrt(u1)*torch.sin(2*torch.pi*u3)
#     q4 = torch.sqrt(u1)*torch.cos(2*torch.pi*u3)
#     Q = torch.cat([q1,q2,q3,q4],dim=-1)
#     return Q

class Tita(LeggedRobot):
    def _init_buffers(self):
        super()._init_buffers()
        self.hip_joint_indices = [0, 4]
        self.foot_joint_indices = [3, 7]

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
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
 
        return self.obs_buf,self.privileged_obs_buf,self.rew_buf,self.cost_buf,self.reset_buf, self.extras

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        if self.cfg.control.use_filter:
            actions = self._low_pass_action_filter(actions)

        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, self.hip_joint_indices] *= self.cfg.control.hip_scale_reduction
        # actions_scaled[:, self.foot_joint_indices] *= self.cfg.control.foot_scale_reduction

        # if self.cfg.domain_rand.randomize_lag_timesteps:
        #     self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
        #     joint_pos_target = self.lag_buffer[0] + self.default_dof_pos
        # else:
        #     joint_pos_target = actions_scaled + self.default_dof_pos

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = torch.cat([self.lag_buffer[:,1:,:].clone(),actions_scaled.unsqueeze(1).clone()],dim=1)
            joint_pos_target = self.lag_buffer[self.num_envs_indexes,self.randomized_lag,:] + self.default_dof_pos
        else:
            joint_pos_target = actions_scaled + self.default_dof_pos

        # joint_pos_target = torch.clamp(joint_pos_target,self.dof_pos-1,self.dof_pos+1)

        control_type = self.cfg.control.control_type
        if control_type == "P":
            if not self.cfg.domain_rand.randomize_kpkd:  # TODO add strength to gain directly
                torques = self.p_gains*(joint_pos_target - self.dof_pos) - self.d_gains*self.dof_vel
                torques[:,self.foot_joint_indices] = self.p_gains[self.foot_joint_indices] * actions_scaled[:,self.foot_joint_indices] - self.d_gains[self.foot_joint_indices] * self.dof_vel[:,self.foot_joint_indices]                
            else:
                torques = self.kp_factor * self.p_gains*(joint_pos_target - self.dof_pos) - self.kd_factor * self.d_gains*self.dof_vel
                torques[:,self.foot_joint_indices] = self.kp_factor[:,self.foot_joint_indices]  * self.p_gains[self.foot_joint_indices] * actions_scaled[:,self.foot_joint_indices]- self.kd_factor[:,self.foot_joint_indices] *self.d_gains[self.foot_joint_indices] * self.dof_vel[:,self.foot_joint_indices]
        else: 
            raise NameError(f"Unknown controller type: {control_type}")
        # torques = torques * self.motor_strength
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # def _reset_root_states(self, env_ids):
    #     """ Resets ROOT states position and velocities of selected environmments
    #         Sets base position based on the curriculum
    #         Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
    #     Args:
    #         env_ids (List[int]): Environemnt ids
    #     """
    #     # base position
    #     if self.custom_origins:
    #         self.root_states[env_ids] = self.base_init_state
    #         self.root_states[env_ids, :3] += self.env_origins[env_ids]
    #         self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
    #     else:
    #         self.root_states[env_ids] = self.base_init_state
    #         self.root_states[env_ids, :3] += self.env_origins[env_ids]
    #     # base velocities
    #     # self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
    #     # # random ori
    #     # self.root_states[env_ids, 3:7] = random_quat(torch_rand_float(0, 1, (len(env_ids), 4), device=self.device))
    #     # random height
    #     # self.root_states[env_ids, 2:3] += torch_rand_float(0, 0.2, (len(env_ids), 1), device=self.device)
    #     env_ids_int32 = env_ids.to(dtype=torch.int32)
    #     self.gym.set_actor_root_state_tensor_indexed(self.sim,
    #                                                  gymtorch.unwrap_tensor(self.root_states),
    #                                                  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    

    #------------ reward functions----------------
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        # reward_pos = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        reward_vel = 0.005*torch.sum(torch.abs(self.dof_vel[:, [3,7]]), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.05)
        return reward_vel
    

    def _reward_com_feet(self):
        com = self.root_states[:, 0:2]
        feet_com = (self.feet_pos[:, 0, 0:2] + self.feet_pos[:, 1, 0:2])/2
        err = torch.sum(torch.square(com - feet_com), dim=1)
        return torch.exp(-err/0.25)
    
    def _reward_foot_mirror(self):
        # penalty when feet contact not mirror, RL foot mirror RR foot, FL foot mirror FR foot
        mirror = torch.tensor([-1, 1, 1], device=self.device)
        # reward = torch.exp(-torch.sum(torch.square(self.dof_pos[:,[0,1,2]] - self.dof_pos[:,[4,5,6]] * mirror),dim=-1)/0.05) 
        reward = torch.sum(torch.square(self.dof_pos[:,[0,1,2]] - self.dof_pos[:,[4,5,6]] * mirror),dim=-1) 

        return reward 

    def _reward_hip_pos(self):
        # max_rad = 0.025
        # hip_err = torch.abs(self.dof_pos[:, 0] - self.dof_pos[:, 4])
        # hip_err = torch.where(hip_err < max_rad, torch.zeros_like(hip_err), hip_err)
        # # print(hip_err - max_rad)
        # reward = torch.clip(hip_err - max_rad, 0, 1)
        # return reward
        # # penalty hip joint position not equal to zero
        reward = torch.sum(torch.square(self.dof_pos[:, [0, 4]] - torch.zeros_like(self.dof_pos[:, [0, 4]])), dim=1)
        return reward
    
    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1. * contacts, dim=1) == 1
        return 1. * single_contact
    
    def _reward_feet_distance(self):
        feet_state = self.rigid_body_states[:, self.feet_indices, :]
        feet_distance = torch.abs(torch.norm(feet_state[:, 0, :2] - feet_state[:, 1, :2], dim=-1))
        # reward = torch.abs(feet_distance - self.cfg.rewards.min_feet_distance)
        reward = torch.clip(self.cfg.rewards.min_feet_distance - feet_distance, 0, 1) + \
                 torch.clip(feet_distance - self.cfg.rewards.max_feet_distance, 0, 1)
        return reward

    def _reward_survival(self):
        # return (~self.reset_buf).float() * self.dt
        return (self.episode_length_buf * self.dt) > 10
    
    def _reward_leg_symmetry(self):
        foot_positions_base = self.feet_pos - \
                            (self.root_states[:, 0:3]).unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        for i in range(len(self.feet_indices)):
            foot_positions_base[:, i, :] = quat_rotate_inverse(self.base_quat, foot_positions_base[:, i, :] )
        leg_symmetry_err = (abs(foot_positions_base[:,0,1])-abs(foot_positions_base[:,1,1]))
        return torch.exp(-(leg_symmetry_err ** 2)/ 0.001)
    def _reward_wheel_adjustment(self):
        # 鼓励使用轮子的滑动克服前后的倾斜，奖励轮速和倾斜方向一致的情况，并要求轮速方向也一致
        incline_x = self.projected_gravity[:, 0]
        # mean velocity
        wheel_x_mean = (self.feet_vel[:, 0, 0] + self.feet_vel[:, 1, 0]) / 2
        # 两边轮速不一致的情况，不给奖励
        wheel_x_invalid = (self.feet_vel[:, 0, 0] * self.feet_vel[:, 1, 0]) < 0
        wheel_x_mean[wheel_x_invalid] = 0.0
        wheel_x_mean = wheel_x_mean.reshape(-1)
        reward = incline_x * wheel_x_mean > 0
        return reward
