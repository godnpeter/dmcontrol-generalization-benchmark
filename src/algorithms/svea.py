import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
# from new_augmentations import RandomConv, RandomOverlay, CutoutColor, RandomShiftsAug, RandomFlip, RandomRotate, Projection_Transformation
import utils
import augmentations
import algorithms.modules as m
from algorithms.sac import SAC
import ipdb
import random


class SVEA(SAC):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)
		self.svea_alpha = args.svea_alpha
		self.svea_beta = args.svea_beta
		self.hard_aug_type = args.hard_aug_type
		
		if self.hard_aug_type == 'combo':
			self.combo_aug = []
			for aug in args.combo_aug_type_list:
				if aug == 'random_shift':
					self.combo_aug.append(RandomShiftsAug(pad=4, random_shift_mode='bilinear'))
				elif aug == 'random_conv':
					self.combo_aug.append(RandomConv())
				elif aug == 'random_overlay':
					self.combo_aug.append(RandomOverlay(alpha=0.5))
				elif aug == 'cutout_color':
					self.combo_aug.append(CutoutColor())
				elif aug == 'flip':
					self.combo_aug.append(RandomFlip())
				elif aug == 'rotate':
					self.combo_aug.append(RandomRotate(degrees=45.0))
				elif aug == 'projection_transformation':
					self.combo_aug.append(Projection_Transformation())

			#self.combo_aug = [augmentations.random_overlay, augmentations.random_conv, augmentations.random_shift]
		# This isn't really worth it... hh
		elif self.hard_aug_type == 'sequential_combo':
			self.combo_aug = [augmentations.random_conv, augmentations.random_overlay]
			self.combo_aug_index = 0
			# Learn from random_shift only once
			self.random_hard_aug = augmentations.random_shift

	def reset(self, step):
		super().reset(step)
		if self.hard_aug_type == 'sequential_combo':
			self.random_hard_aug = self.get_next_combo_aug()
			print("Changed augmentation to : ", self.random_hard_aug.__name__)

	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
		with torch.no_grad():
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1,
								 target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		if self.svea_alpha == self.svea_beta:
			if self.hard_aug_type == 'random_overlay':
				obs = utils.cat(obs, augmentations.random_overlay(obs.clone()).cuda())
			elif self.hard_aug_type == 'random_conv':
				obs = utils.cat(obs, augmentations.random_conv(obs.clone()).cuda())
			elif self.hard_aug_type == 'combo':
				random_hard_aug = random.choice(self.combo_aug)
				obs = utils.cat(obs, random_hard_aug(obs.clone()).cuda())
			elif self.hard_aug_type == 'sequential_combo':
				obs = utils.cat(obs, self.random_hard_aug(obs.clone()).cuda())
			
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

	def update(self, replay_buffer, L, step):
		obs, action, reward, next_obs, not_done = replay_buffer.sample_svea()

		self.update_critic(obs, action, reward, next_obs, not_done, L, step)

		if step % self.actor_update_freq == 0:
			self.update_actor_and_alpha(obs, L, step)

		if step % self.critic_target_update_freq == 0:
			self.soft_update_critic_target()

	def get_next_combo_aug(self):
		combo_hard_aug = self.combo_aug[self.combo_aug_index]
		self.combo_aug_index = (self.combo_aug_index + 1) % len(self.combo_aug)
		return combo_hard_aug