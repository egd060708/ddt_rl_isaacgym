import torch
import torch.nn as nn
from torch import autograd
import torch.nn.functional as F


class WAMPDiscriminator(nn.Module):
    """
    Wasserstein Adversarial Imitation Learning Discriminator (WGAN-GP style)
    Inspired by HumanMimic paper and standard WGAN-GP implementation.
    Uses Earth Mover's Distance with Gradient Penalty on interpolated samples.
    """
    def __init__(
        self,
        input_dim,
        amp_reward_coef,
        hidden_layer_sizes=[1024, 512, 256],
        device="cuda",
        lambda_gp=10.0,
        task_reward_lerp=0.0,
        amp_reward_scale=0.25,
    ):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.amp_reward_coef = amp_reward_coef
        self.lambda_gp = lambda_gp
        self.task_reward_lerp = task_reward_lerp
        self.amp_reward_scale = amp_reward_scale

        # Build trunk network (same as AMP but without final linear for flexibility)
        layers = []
        curr_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # Better for WGAN stability
            layers.append(nn.LeakyReLU(0.2))
            curr_dim = hidden_dim
        self.trunk = nn.Sequential(*layers).to(device)

        self.output_layer = nn.Linear(hidden_layer_sizes[-1], 1).to(device)

        self.trunk.train()
        self.output_layer.train()

    def forward(self, x):
        """Return critic score (real-valued, no sigmoid)"""
        h = self.trunk(x)
        d = self.output_layer(h)
        return d

    def compute_gradient_penalty(self, real_data, fake_data, lambda_gp=None):
        """Standard WGAN-GP gradient penalty on interpolated samples"""
        if lambda_gp is None:
            lambda_gp = self.lambda_gp

        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, device=self.device).expand_as(real_data)

        # Interpolate between real and fake
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)

        disc_interpolates = self.forward(interpolates)

        # Compute gradients
        ones = torch.ones(disc_interpolates.size(), device=self.device)
        gradients = autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty

    def compute_wasserstein_loss(self, real_scores, fake_scores):
        """Wasserstein loss: E[D(real)] - E[D(fake)]"""
        w_loss = fake_scores.mean() - real_scores.mean()
        return w_loss

    def predict_amp_reward(
        self, state, next_state, task_reward, normalizer=None, use_wasserstein=True
    ):
        """Predict imitation reward based on Wasserstein critic score"""
        with torch.no_grad():
            self.eval()
            if normalizer is not None:
                state = normalizer.normalize_torch(state, self.device)
                next_state = normalizer.normalize_torch(next_state, self.device)

            combined = torch.cat([state, next_state], dim=-1)
            d = self.forward(combined)

            if use_wasserstein:
                # For Wasserstein, reward is typically based on how close to expert
                # We can map higher critic score (more "real") to higher reward
                amp_reward = self.amp_reward_coef * torch.sigmoid(d) * 2.0  # scale to [0,2] range
            else:
                # Fallback to original AMP style
                amp_reward = self.amp_reward_coef * torch.clamp(
                    1.0 - self.amp_reward_scale * torch.square(d - 1.0), min=0.0
                )

            if self.task_reward_lerp > 0.0:
                reward = (1.0 - self.task_reward_lerp) * amp_reward + self.task_reward_lerp * task_reward.unsqueeze(-1)
            else:
                reward = amp_reward

            self.train()
        return reward.squeeze(-1), amp_reward.squeeze(-1), task_reward, d.squeeze(-1)

