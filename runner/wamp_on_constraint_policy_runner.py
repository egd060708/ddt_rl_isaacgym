import os
import time
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from global_config import ROOT_DIR
from modules import ActorCriticBarlowTwins
from algorithm.wamp_np3o import WAMPNP3O
from algorithm.wamp_discriminator import WAMPDiscriminator
from algorithm.datasets.motion_loader import AMPLoader, motion_layout_from_legged_cfg
from utils import Normalizer, get_load_path
from envs.vec_env import VecEnv


class WAMPOnConstraintPolicyRunner:
    """
    Independent implementation of WAMP (Wasserstein Adversarial Imitation) Runner.
    References AMPOnConstraintPolicyRunner but does NOT inherit from it.
    Uses WGAN-GP discriminator and WAMPNP3O for HumanMimic-style training.
    """

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device='cpu'):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # Create policy
        actor_critic_class = eval(self.cfg["policy_class_name"])
        self.actor_critic: ActorCriticBarlowTwins = actor_critic_class(
            self.env.cfg.env.n_proprio,
            self.env.cfg.env.n_scan,
            self.env.num_obs,
            self.env.cfg.env.n_priv_latent,
            self.env.cfg.env.history_len,
            self.env.num_actions,
            **self.policy_cfg
        ).to(self.device)

        if self.cfg.get('resume', False):
            log_root = os.path.join(ROOT_DIR, 'logs', self.cfg['experiment_name'], self.cfg.get('resume_path', ''))
            resume_path = get_load_path(log_root, load_run=self.cfg.get('load_run'), checkpoint=self.cfg.get('checkpoint'))
            print("Resume model from: ", resume_path)
            model_dict = torch.load(resume_path, map_location=self.device)
            self.actor_critic.load_state_dict(model_dict['model_state_dict'])

        print("Policy architecture: ", self.actor_critic)

        # Create AMP data loader for expert motions
        _ml = motion_layout_from_legged_cfg(self.env.cfg)
        self.amp_data = AMPLoader(
            device=self.device,
            time_between_frames=self.env.dt,
            preload_transitions=True,
            num_preload_transitions=self.cfg.get('amp_num_preload_transitions', 3000000),
            motion_files=self.cfg.get("amp_motion_files", []),
            motion_layout=_ml if _ml else None,
        )

        self.amp_normalizer = Normalizer(self.amp_data.observation_dim)

        # Create WGAN-GP Discriminator (Wasserstein version)
        self.discriminator = WAMPDiscriminator(
            input_dim=self.amp_data.observation_dim * 2,
            amp_reward_coef=self.cfg.get("amp_reward_coef", 1.0),
            hidden_layer_sizes=self.cfg.get("amp_discr_hidden_dims", [1024, 512, 256]),
            device=self.device,
            soft_bound_scale=self.cfg.get("wamp_soft_bound_scale", 0.3),
            lambda_gp=self.cfg.get("wasserstein_lambda", 10.0),
            task_reward_lerp=self.cfg.get("amp_task_reward_lerp", 0.3),
            amp_reward_scale=self.cfg.get("amp_reward_scale", 0.25),
        ).to(self.device)

        # Create WAMPNP3O algorithm
        self.alg_cfg['k_value'] = getattr(
            self.env, 'cost_k_values', torch.tensor([0.1], device=self.device)
        )

        min_std = None
        if hasattr(self.env, 'amp_min_std_limit') and self.env.amp_min_std_limit is not None:
            min_std = (
                torch.tensor(self.cfg.get("min_normalized_std", [0.05, 0.02, 0.05, 0.1]), device=self.device) *
                torch.abs(self.env.amp_min_std_limit[:, 1] - self.env.amp_min_std_limit[:, 0])
            )

        self.alg = WAMPNP3O(
            actor_critic=self.actor_critic,
            discriminator=self.discriminator,
            amp_data=self.amp_data,
            amp_normalizer=self.amp_normalizer,
            device=self.device,
            wasserstein_lambda=self.cfg.get("wasserstein_lambda", 10.0),
            min_std=min_std,
            **self.alg_cfg
        )

        self.num_steps_per_env = self.cfg.get("num_steps_per_env", 24)
        self.save_interval = self.cfg.get("save_interval", 50)

        # Initialize storage
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [getattr(self.env, 'num_privileged_obs', 0)],
            [self.env.num_actions],
            [getattr(self.env.cfg.costs, 'num_costs', 3)],
            getattr(self.env, 'cost_d_values_tensor', None)
        )

        self.env.reset()

        print(
            f"WAMP Runner initialized with {len(self.amp_data.trajectory_names)} "
            "expert motions using Wasserstein Adversarial Imitation."
        )

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length))

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations() if hasattr(self.env, 'get_privileged_observations') else None
        amp_obs = self.env.get_amp_observations() if hasattr(self.env, 'get_amp_observations') else obs
        critic_obs = privileged_obs if privileged_obs is not None else obs

        obs = obs.to(self.device)
        critic_obs = critic_obs.to(self.device)
        amp_obs = amp_obs.to(self.device)

        self.alg.actor_critic.train()
        self.discriminator.train()

        ep_infos = []
        task_rewbuffer = deque(maxlen=100)
        amp_rewbuffer = deque(maxlen=100)
        total_rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_task_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_amp_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        for it in range(self.current_learning_iteration, tot_iter):
            iter_start = time.time()
            start = time.time()

            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, {}, amp_obs)
                    obs, privileged_obs, rewards, costs, dones, infos, reset_env_ids, terminal_amp_states = self.env.step(actions)
                    next_amp_obs = self.env.get_amp_observations() if hasattr(self.env, 'get_amp_observations') else obs

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, next_amp_obs = obs.to(self.device), critic_obs.to(self.device), next_amp_obs.to(self.device)
                    rewards, costs, dones = rewards.to(self.device), costs.to(self.device), dones.to(self.device)

                    next_amp_obs_with_term = next_amp_obs.clone()
                    if len(reset_env_ids) > 0:
                        next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    rewards, amp_reward, task_reward, _ = self.discriminator.predict_amp_reward(
                        amp_obs, next_amp_obs_with_term, rewards, normalizer=self.alg.amp_normalizer
                    )
                    amp_obs = next_amp_obs.clone()

                    self.alg.process_env_step(rewards, costs, dones, infos, next_amp_obs_with_term)

                    if self.log_dir is not None and 'episode' in infos:
                        ep_infos.append(infos['episode'])

                    cur_task_reward_sum += task_reward
                    cur_amp_reward_sum += amp_reward
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False).flatten()
                    if len(new_ids) > 0:
                        task_rewbuffer.extend(cur_task_reward_sum[new_ids].cpu().numpy().tolist())
                        amp_rewbuffer.extend(cur_amp_reward_sum[new_ids].cpu().numpy().tolist())
                        total_rewbuffer.extend(cur_reward_sum[new_ids].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids].cpu().numpy().tolist())
                        cur_task_reward_sum[new_ids] = 0
                        cur_amp_reward_sum[new_ids] = 0
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

            stop = time.time()
            collection_time = stop - start

            start = stop
            self.alg.compute_returns(critic_obs)
            self.alg.compute_cost_returns(critic_obs)
            k_value = self.alg.update_k_value(it)
            train_info = self.alg.update()

            learn_time = time.time() - start

            log_time = 0.0
            if self.log_dir is not None:
                log_start = time.time()
                self._log(locals(), it)
                log_time = time.time() - log_start
                ep_infos.clear()

            save_time = 0.0
            if it % self.cfg.get("save_interval", 50) == 0:
                save_start = time.time()
                self.save(os.path.join(self.log_dir, f'model_{it}.pt'))
                save_time = time.time() - save_start

            total_iter_time = time.time() - iter_start
            if log_time > 0.5 or save_time > 0.5 or total_iter_time > collection_time + learn_time + 0.5:
                print(
                    f"[WAMP Iter {it} timing] "
                    f"rollout {collection_time:.2f}s | "
                    f"learn {learn_time:.2f}s | "
                    f"log {log_time:.2f}s | "
                    f"save {save_time:.2f}s | "
                    f"total {total_iter_time:.2f}s"
                )

            self.current_learning_iteration += 1

        self.save(os.path.join(self.log_dir, 'model_final.pt'))
        return self.actor_critic

    def _log(self, locals_dict, it):
        train_info = locals_dict.get('train_info', {})
        collection_time = locals_dict.get('collection_time', 0.0)
        learn_time = locals_dict.get('learn_time', 0.0)
        iteration_time = collection_time + learn_time
        self.tot_time += iteration_time
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs

        if self.writer is not None:
            ep_infos = locals_dict.get('ep_infos', [])
            if ep_infos:
                for key in ep_infos[0]:
                    infotensor = torch.tensor([], device=self.device)
                    for ep_info in ep_infos:
                        if key not in ep_info:
                            continue
                        value = ep_info[key]
                        if not torch.is_tensor(value):
                            value = torch.tensor(value, device=self.device)
                        infotensor = torch.cat(
                            (infotensor, value.to(self.device).reshape(-1))
                        )
                    if infotensor.numel() > 0:
                        self.writer.add_scalar(
                            'Episode/' + key, torch.mean(infotensor), it
                        )

            self.writer.add_scalar('Loss/value_function', train_info['value_loss'], it)
            self.writer.add_scalar('Loss/cost_value_function', train_info['cost_value_loss'], it)
            self.writer.add_scalar('Loss/surrogate', train_info['surrogate_loss'], it)
            self.writer.add_scalar('Loss/mean_viol_loss', train_info['viol_loss'], it)
            self.writer.add_scalar('Loss/WAMP_wasserstein', train_info['w_loss'], it)
            self.writer.add_scalar('Loss/WAMP_gp', train_info['gp_loss'], it)
            self.writer.add_scalar('WAMP/policy_score', train_info['policy_score'], it)
            self.writer.add_scalar('WAMP/expert_score', train_info['expert_score'], it)
            self.writer.add_scalar('Data/obs_max', train_info['obs_max'], it)
            self.writer.add_scalar('Data/obs_min', train_info['obs_min'], it)

            if len(locals_dict['amp_rewbuffer']) > 0:
                self.writer.add_scalar('Train/mean_amp_reward', np.mean(locals_dict['amp_rewbuffer']), it)
                self.writer.add_scalar('Train/mean_task_reward', np.mean(locals_dict['task_rewbuffer']), it)
                self.writer.add_scalar('Train/mean_total_reward', np.mean(locals_dict['total_rewbuffer']), it)
                self.writer.add_scalar('Train/mean_episode_length', np.mean(locals_dict['lenbuffer']), it)

        print(
            f"[WAMP Iter {it}] "
            f"value {train_info['value_loss']:.4f} | "
            f"surrogate {train_info['surrogate_loss']:.4f} | "
            f"w_loss {train_info['w_loss']:.4f} | "
            f"gp {train_info['gp_loss']:.4f} | "
            f"policy_score {train_info['policy_score']:.4f} | "
            f"expert_score {train_info['expert_score']:.4f} | "
            f"time {iteration_time:.2f}s"
        )

    def get_inference_policy(self, device=None):
        if device is None:
            device = self.device
        self.actor_critic.to(device)
        self.actor_critic.eval()
        return self.actor_critic.act_inference

    def save(self, path, infos=None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'discriminator_state_dict': self.alg.discriminator.state_dict(),
            'amp_normalizer': self.alg.amp_normalizer,
            'iter': self.current_learning_iteration,
            'infos': infos,
        }
        torch.save(state_dict, path)

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print(f"Loading WAMP model from {path}...")
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if 'discriminator_state_dict' in loaded_dict:
            self.alg.discriminator.load_state_dict(loaded_dict['discriminator_state_dict'])
        if 'amp_normalizer' in loaded_dict:
            self.alg.amp_normalizer = loaded_dict['amp_normalizer']
        if load_optimizer and 'optimizer_state_dict' in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict.get('iter', 0)
        print("*" * 80)
        return loaded_dict.get('infos')
