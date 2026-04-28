import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd

from modules.actor_critic import ActorCriticBarlowTwins
from runner.rollout_storage import RolloutStorageWithCost
from utils import unpad_trajectories
from runner.replay_buffer import ReplayBuffer
from runner.rollout_storage import RolloutStorage
from .wamp_discriminator import WAMPDiscriminator


class WAMPNP3O:
    """
    Wasserstein Adversarial Imitation + NP3O (WAMP-NP3O)
    Implements the Wasserstein Adversarial Imitation from HumanMimic paper.
    Uses WGAN-GP style critic loss instead of LSGAN/MSE.
    """
    actor_critic: ActorCriticBarlowTwins

    def __init__(
        self,
        actor_critic,
        discriminator,
        amp_data,
        amp_normalizer,
        k_value,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        cost_value_loss_coef=1.0,
        cost_viol_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        amp_replay_buffer_size=100000,
        min_std=None,
        dagger_update_freq=20,
        priv_reg_coef_schedual=[0, 0, 0],
        wasserstein_lambda=10.0,
        **kwargs,
    ):
        self.device = device
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.min_std = min_std
        self.wasserstein_lambda = wasserstein_lambda

        # Discriminator (Wasserstein version)
        self.discriminator: WAMPDiscriminator = discriminator
        self.discriminator.to(self.device)
        self.amp_transition = RolloutStorage.Transition()
        self.amp_storage = ReplayBuffer(
            discriminator.input_dim // 2, amp_replay_buffer_size, device
        )
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        # Policy
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None

        # Optimizer - shared between policy and critic (WGAN style)
        params = [
            {"params": self.actor_critic.parameters(), "name": "actor_critic"},
            {
                "params": self.discriminator.trunk.parameters(),
                "weight_decay": 1e-4,
                "name": "wamp_trunk",
            },
            {
                "params": self.discriminator.output_layer.parameters(),
                "weight_decay": 1e-2,
                "name": "wamp_head",
            },
        ]
        self.optimizer = optim.Adam(params, lr=learning_rate)

        # Imitation flag
        if hasattr(self.actor_critic, "imitation_learning_loss") and getattr(
            self.actor_critic, "imi_flag", False
        ):
            self.imi_flag = True
            print("WAMP: running with imitation loss")
        else:
            self.imi_flag = False
            print("WAMP: running without imitation loss")

        self.imi_weight = 1
        self.transition = RolloutStorageWithCost.Transition()

        # NP3O / PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.cost_value_loss_coef = cost_value_loss_coef
        self.cost_viol_loss_coef = cost_viol_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.k_value = k_value

        self.priv_reg_coef_schedual = priv_reg_coef_schedual
        self.priv_reg_coef = priv_reg_coef_schedual[0]

        print("WAMPNP3O initialized with Wasserstein Adversarial Imitation")

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, cost_shape, cost_d_values):
        self.storage = RolloutStorageWithCost(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, cost_shape, cost_d_values, self.device)

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def set_imi_flag(self, flag):
        self.imi_flag = flag
        if self.imi_flag:
            print("WAMP: running with imitation")
        else:
            print("WAMP: running without imitation")

    def set_imi_weight(self, value):
        self.imi_weight = value
        
    def compute_returns(self, last_critic_obs):
        aug_last_critic_obs = last_critic_obs.detach()
        last_values= self.actor_critic.evaluate(aug_last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def compute_cost_returns(self, obs):
        last_cost_values = self.actor_critic.evaluate_cost(obs).detach()
        self.storage.compute_cost_returns(last_cost_values,self.gamma,self.lam)

    def compute_surrogate_loss(self, actions_log_prob_batch, old_actions_log_prob_batch, advantages_batch):
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
        return surrogate_loss

    def compute_cost_surrogate_loss(self, actions_log_prob_batch, old_actions_log_prob_batch, cost_advantages_batch):
        ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
        surrogate = cost_advantages_batch * ratio.view(-1, 1)
        surrogate_clipped = cost_advantages_batch * torch.clamp(ratio.view(-1, 1), 1.0 - self.clip_param, 1.0 + self.clip_param)
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean(0)
        return surrogate_loss

    def compute_value_loss(self, target_values_batch, value_batch, returns_batch):
        if self.use_clipped_value_loss:
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param, self.clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns_batch - value_batch).pow(2).mean()
        return value_loss

    def compute_viol(self, actions_log_prob_batch, old_actions_log_prob_batch, cost_advantages_batch, cost_volation_batch):
        cost_surrogate_loss = self.compute_cost_surrogate_loss(
            actions_log_prob_batch=actions_log_prob_batch,
            old_actions_log_prob_batch=old_actions_log_prob_batch,
            cost_advantages_batch=cost_advantages_batch)
        cost_volation_loss = cost_volation_batch.mean()
        cost_loss = cost_surrogate_loss + cost_volation_loss
        cost_loss = torch.sum(self.k_value * F.relu(cost_loss))
        return cost_loss

    def update_k_value(self, i):
        self.k_value = torch.min(torch.ones_like(self.k_value), self.k_value * (1.0004 ** i))
        return self.k_value

    def act(self, obs, critic_obs, info, amp_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # compute the actions and values
        aug_obs, aug_critic_obs = obs.detach(), critic_obs.detach()
        self.transition.actions = self.actor_critic.act(aug_obs).detach()
        self.transition.values = self.actor_critic.evaluate(aug_critic_obs).detach()
        self.transition.cost_values = self.actor_critic.evaluate_cost(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.amp_transition.observations = amp_obs
        return self.transition.actions

    def process_env_step(self, rewards, costs, dones, infos, amp_obs):
        self.transition.rewards = rewards.clone()
        self.transition.costs = costs.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
            self.transition.costs += self.gamma * (self.transition.costs * infos['time_outs'].unsqueeze(1).to(self.device))
        self.amp_storage.insert(
            self.amp_transition.observations, amp_obs)
        
        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.amp_transition.clear()
        self.actor_critic.reset(dones)

    def update(self):
        """Wasserstein Adversarial Imitation + NP3O update.
        Fully aligned with amp_np3o.py, only discriminator loss is replaced with WGAN-GP.
        """
        mean_value_loss = 0
        mean_cost_value_loss = 0
        mean_viol_loss = 0
        mean_surrogate_loss = 0
        mean_imitation_loss = 0
        mean_w_loss = 0
        mean_gp_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        obs_batch_max = -math.inf
        obs_batch_min = math.inf

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        amp_policy_generator = self.amp_storage.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
            self.num_mini_batches)
        amp_expert_generator = self.amp_data.feed_forward_generator(
            self.num_learning_epochs * self.num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env //
            self.num_mini_batches)

        for sample, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, target_cost_values_batch, cost_advantages_batch, cost_returns_batch, cost_violation_batch = sample

            aug_obs_batch = obs_batch.detach()

            self.actor_critic.act(aug_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            aug_critic_obs_batch = critic_obs_batch.detach()
            value_batch = self.actor_critic.evaluate(aug_critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            cost_value_batch = self.actor_critic.evaluate_cost(aug_critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL adaptive learning rate
            if self.desired_kl is not None and self.schedule == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            surrogate_loss = self.compute_surrogate_loss(actions_log_prob_batch=actions_log_prob_batch,
                                                         old_actions_log_prob_batch=old_actions_log_prob_batch,
                                                         advantages_batch=advantages_batch)

            viol_loss = self.compute_viol(actions_log_prob_batch=actions_log_prob_batch,
                                          old_actions_log_prob_batch=old_actions_log_prob_batch,
                                          cost_advantages_batch=cost_advantages_batch,
                                          cost_volation_batch=cost_violation_batch)

            value_loss = self.compute_value_loss(target_values_batch=target_values_batch,
                                                 value_batch=value_batch,
                                                 returns_batch=returns_batch)

            cost_value_loss = self.compute_value_loss(target_values_batch=target_cost_values_batch,
                                                      value_batch=cost_value_batch,
                                                      returns_batch=cost_returns_batch)

            # === Wasserstein Discriminator Loss (replaces MSE in amp_np3o) ===
            policy_state, policy_next_state = sample_amp_policy
            expert_state, expert_next_state = sample_amp_expert
            policy_state_raw = policy_state
            expert_state_raw = expert_state

            if self.amp_normalizer is not None:
                with torch.no_grad():
                    policy_state = self.amp_normalizer.normalize_torch(policy_state, self.device)
                    policy_next_state = self.amp_normalizer.normalize_torch(policy_next_state, self.device)
                    expert_state = self.amp_normalizer.normalize_torch(expert_state, self.device)
                    expert_next_state = self.amp_normalizer.normalize_torch(expert_next_state, self.device)

            real_input = torch.cat([expert_state, expert_next_state], dim=-1)
            fake_input = torch.cat([policy_state, policy_next_state], dim=-1)

            real_scores = self.discriminator(real_input)
            fake_scores = self.discriminator(fake_input)

            w_loss = self.discriminator.compute_wasserstein_loss(real_scores, fake_scores)
            gp_loss = self.discriminator.compute_gradient_penalty(real_input, fake_input, self.wasserstein_lambda)

            amp_loss = w_loss + gp_loss   # amp_loss now represents Wasserstein + GP

            main_loss = surrogate_loss + self.cost_viol_loss_coef * viol_loss
            combine_value_loss = self.cost_value_loss_coef * cost_value_loss + self.value_loss_coef * value_loss
            entropy_loss = - self.entropy_coef * entropy_batch.mean()

            if self.imi_flag:
                imitation_loss = self.actor_critic.imitation_learning_loss(obs_batch, self.imi_weight)
                loss = main_loss + combine_value_loss + entropy_loss + imitation_loss + amp_loss
            else:
                loss = main_loss + combine_value_loss + entropy_loss + amp_loss

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if not self.actor_critic.fixed_std and self.min_std is not None:
                self.actor_critic.std.data = self.actor_critic.std.data.clamp(min=self.min_std)

            if self.amp_normalizer is not None:
                self.amp_normalizer.update(policy_state_raw.cpu().numpy())
                self.amp_normalizer.update(expert_state_raw.cpu().numpy())

            mean_value_loss += value_loss.item()
            mean_cost_value_loss += cost_value_loss.item()
            mean_viol_loss += viol_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_w_loss += w_loss.item()
            mean_gp_loss += gp_loss.item()
            mean_policy_pred += fake_scores.mean().item()
            mean_expert_pred += real_scores.mean().item()

            current_max = obs_batch.max().item()
            current_min = obs_batch.min().item()
            if current_max > obs_batch_max:
                obs_batch_max = current_max
            if current_min < obs_batch_min:
                obs_batch_min = current_min

            if self.imi_flag:
                mean_imitation_loss += imitation_loss.item()
            else:
                mean_imitation_loss += 0

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_cost_value_loss /= num_updates
        mean_viol_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_imitation_loss /= num_updates
        mean_w_loss /= num_updates
        mean_gp_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates

        self.storage.clear()

        return {
            "value_loss": mean_value_loss,
            "cost_value_loss": mean_cost_value_loss,
            "viol_loss": mean_viol_loss,
            "surrogate_loss": mean_surrogate_loss,
            "imitation_loss": mean_imitation_loss,
            "w_loss": mean_w_loss,
            "gp_loss": mean_gp_loss,
            "policy_score": mean_policy_pred,
            "expert_score": mean_expert_pred,
            "obs_max": obs_batch_max,
            "obs_min": obs_batch_min,
        }
