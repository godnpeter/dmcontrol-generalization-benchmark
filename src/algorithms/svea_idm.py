import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import augmentations
import algorithms.modules as m
from algorithms.sac import SAC
import ipdb
import random


class SVEA_IDM(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.svea_alpha = args.svea_alpha
		self.svea_beta = args.svea_beta
		self.hard_aug_type = args.hard_aug_type
		if self.hard_aug_type == 'combo':
			self.combo_aug = [augmentations.random_overlay, augmentations.random_conv, augmentations.random_shift]

		self.aux_update_freq = args.aux_update_freq
		self.aux_lr = args.aux_lr
		self.aux_beta = args.aux_beta
		shared_cnn = self.critic.encoder.shared_cnn
		aux_cnn = m.HeadCNN(shared_cnn.out_shape, args.num_head_layers, args.num_filters).cuda()
		self.aux_encoder = m.Encoder(
			shared_cnn,
			aux_cnn,
			m.RLProjection(aux_cnn.out_shape, args.projection_dim)
		)
		self.pad_head = m.InverseDynamics(self.aux_encoder, action_shape, args.hidden_dim).cuda()
		self.init_pad_optimizer()
		self.train()

	def train(self, training=True):
		super().train(training)
		if hasattr(self, 'pad_head'):
			self.pad_head.train(training)
   
	def reset(self, step):
		super().reset(step)
		reset_seed = self.base_seed + step
		if self.args.do_policy_reset:
			self.aux_encoder.reset_projection_parameters(reset_seed)
			self.pad_head.reset_policy_parameters(reset_seed)
			print('Performed IDM head reset at step ', step)

	def init_pad_optimizer(self):
		self.pad_optimizer = torch.optim.Adam(
			self.pad_head.parameters(), lr=self.aux_lr, betas=(self.aux_beta, 0.999)
		)


	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		if self.svea_alpha == self.svea_beta:
			if self.hard_aug_type == 'random_overlay':
				obs = utils.cat(obs, augmentations.random_overlay(obs.clone()))
			elif self.hard_aug_type == 'random_conv':
				obs = utils.cat(obs, augmentations.random_conv(obs.clone()))
			elif self.hard_aug_type == 'combo':
				random_hard_aug = random.choice(self.combo_aug)
				obs = utils.cat(obs, random_hard_aug(obs.clone()))
			
			action = utils.cat(action, action)
			target_Q = utils.cat(target_Q, target_Q)

			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = (self.svea_alpha + self.svea_beta) * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
		else:
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = self.svea_alpha * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))

			obs_aug = augmentations.random_overlay(obs.clone())
			current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
			critic_loss += self.svea_beta * \
				(F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

		if L is not None:
			L.log('train_critic/loss', critic_loss, step)
			
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

	def update_inverse_dynamics(self, obs, obs_next, action, L=None, step=None):
		assert obs.shape[-1] == 84 and obs_next.shape[-1] == 84

		pred_action = self.pad_head(augmentations.random_overlay(obs.clone()), augmentations.random_overlay(obs_next.clone()))
		pad_loss = F.mse_loss(pred_action, action)

		self.pad_optimizer.zero_grad()
		pad_loss.backward()
		self.pad_optimizer.step()
		if L is not None:
			L.log('train/aux_loss', pad_loss, step)

	def update(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_svea()

		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()
   
		if step % self.aux_update_freq == 0:
			self.update_inverse_dynamics(obs, next_obs, action, L, step)
