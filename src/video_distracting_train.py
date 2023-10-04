import torch
import os
import ipdb
import numpy as np
import gym
import utils
import time
import wandb
import ipdb
import random
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

def evaluate(env, agent, video, num_episodes, L, step, test_env=False, final=False):
	episode_rewards = []
	for i in range(num_episodes):
		obs = env.reset()
		video.init(enabled=(i==0))
		done = False
		episode_reward = 0
		while not done:
			with utils.eval_mode(agent):
				action = agent.select_action(obs)
			obs, reward, done, _ = env.step(action)
			video.record(env, env._mode)
			episode_reward += reward

		if L is not None:
			_test_env = test_env if test_env else 'train'
			video.save(f'{step * args.action_repeat}{_test_env}.mp4')
			if final:
				L.log(f'eval/{_test_env}/final_episode_return', episode_reward, step)
			else:
				L.log(f'eval/{_test_env}/episode_return', episode_reward, step)
		episode_rewards.append(episode_reward)
	
	return np.mean(episode_rewards)


def main(args):
	# Set seed
	utils.set_seed_everywhere(args.seed)
	torch.set_num_threads(1)
	# Initialize environments
	gym.logger.set_level(40)

	train_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='train'
	)

	test_ch_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='color_hard',
		intensity=args.distracting_cs_intensity
	)

	test_ve_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='video_easy',
		intensity=args.distracting_cs_intensity
	)

	test_vh_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='video_hard',
		intensity=args.distracting_cs_intensity
	)
 
	test_distractingcs01_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='distracting_cs',
		intensity=0.1
	)

	test_distractingcs02_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='distracting_cs',
		intensity=0.2
	)
 
	test_distractingcs03_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='distracting_cs',
		intensity=0.3
	)

	test_distractingcs04_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='distracting_cs',
		intensity=0.4
	)
 
	test_distractingcs05_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='distracting_cs',
		intensity=0.5
	)

	train_envs = []
	if 'train' in args.train_env_list:
		train_envs.append(train_env)
	if 'color_hard' in args.train_env_list:
		train_envs.append(test_ch_env)
	if 'video_easy' in args.train_env_list:
		train_envs.append(test_ve_env)
	if 'video_hard' in args.train_env_list:
		train_envs.append(test_vh_env)

	# Create working directory
	if args.encoder_type == 'cnn':
		work_dir = os.path.join(args.log_dir, args.group_name, args.exp_name, \
							args.algorithm,  str(args.encoder_type)+ '_' + str(args.num_shared_layers) + '_' + args.hard_aug_type, \
							args.domain_name+'_'+args.task_name, str(args.seed))
	elif args.encoder_type == 'impala':
		work_dir = os.path.join(args.log_dir, args.group_name, args.exp_name, \
							args.algorithm,  str(args.encoder_type) + '_' + args.hard_aug_type, \
							args.domain_name+'_'+args.task_name, str(args.seed))
	else:
		raise NotImplementedError
	
	cropped_obs_shape = (3*args.frame_stack, args.image_crop_size, args.image_crop_size)
	print('Observations:', train_env.observation_space.shape)
	print('Cropped observations:', cropped_obs_shape)
	agent = make_agent(
		obs_shape=cropped_obs_shape,
		action_shape=train_env.action_space.shape,
		args=args
	)

	print('Working directory:', work_dir)
	assert not os.path.exists(os.path.join(work_dir, 'model','500000.pt')), 'specified working directory already exists'
	utils.make_dir(work_dir)
	model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
	utils.write_info(args, os.path.join(work_dir, 'info.log'))

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	replay_buffer = utils.ReplayBuffer(
		obs_shape=train_env.observation_space.shape,
		action_shape=train_env.action_space.shape,
		capacity=args.train_steps,
		batch_size=args.batch_size
	)
	
	start_step, episode, episode_reward, done = 0, 0, 0, True
	L = Logger(work_dir, args)
	start_time = time.time()
	for env_name in args.train_env_list:
		print("Starting training in train envs: ", env_name)
	
	#env = train_env
	for step in range(start_step, args.train_steps+1):
		if done:
			if step > start_step:
				L.log('train/duration', time.time() - start_time, step * args.action_repeat)
				start_time = time.time()
				L.dump(step * args.action_repeat)

			# Evaluate agent periodically
			if (step * args.action_repeat) % args.eval_freq == 0:
				print('Evaluating:', work_dir)
				eval_start_time = time.time()
				L.log('eval/episode', episode, step * args.action_repeat)
				evaluate(train_env, agent, video, args.eval_episodes, L, step * args.action_repeat)
				evaluate(test_ch_env, agent, video, args.eval_episodes, L, step * args.action_repeat, test_env='color_hard')
				evaluate(test_ve_env, agent, video, args.eval_episodes, L, step * args.action_repeat, test_env='video_easy')
				evaluate(test_vh_env, agent, video, args.eval_episodes, L, step * args.action_repeat, test_env='video_hard')
				L.log('eval/duration', time.time() - eval_start_time, step * args.action_repeat)
				L.dump(step * args.action_repeat)
			# Save agent periodically
			#if step > start_step and step % args.save_freq == 0:
				#torch.save(agent, os.path.join(model_dir, f'{step * args.action_repeat}.pt'))

			L.log('train/episode_return', episode_reward, step * args.action_repeat)

			env = random.choice(train_envs)
			# print('chosen env:', str(env._mode))
			obs = env.reset()
			done = False
			episode_reward = 0
			episode_step = 0
			episode += 1

			L.log('train/episode', episode, step * args.action_repeat)

		# Perform periodic reset
		if args.replay_ratio == 1:
			if (step % args.reset_interval_steps == 0) and (step != 0) and (step < (args.train_steps - 10)):
				agent.reset(step)
		else:
			if (step % (args.reset_interval_steps / args.replay_ratio) == 0) and (step != 0) and (step < (args.train_steps - 10)):
				agent.reset(step)
		
		# Sample action for data collection
		if step < args.init_steps:
			action = env.action_space.sample()
		else:
			with utils.eval_mode(agent):
				action = agent.sample_action(obs)

		# Run training update
		if step >= args.init_steps:
			num_updates = args.init_steps if step == args.init_steps else args.replay_ratio
			for _ in range(num_updates):
				agent.update(replay_buffer, L, step)

		# Take step
		next_obs, reward, done, _ = env.step(action)
		done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
		replay_buffer.add(obs, action, reward, next_obs, done_bool)
		episode_reward += reward
		obs = next_obs

		episode_step += 1

	evaluate(test_ch_env, agent, video, args.eval_episodes, L, step, test_env='color_hard', final=True)
	evaluate(test_ve_env, agent, video, args.eval_episodes, L, step, test_env='video_easy', final=True)
	evaluate(test_vh_env, agent, video, args.eval_episodes, L, step, test_env='video_hard', final=True)
	evaluate(test_distractingcs01_env, agent, video, args.eval_episodes, L, step , test_env='distractingcs_01', final=True)
	evaluate(test_distractingcs02_env, agent, video, args.eval_episodes, L, step , test_env='distractingcs_02', final=True)
	evaluate(test_distractingcs03_env, agent, video, args.eval_episodes, L, step , test_env='distractingcs_03', final=True)
	evaluate(test_distractingcs04_env, agent, video, args.eval_episodes, L, step , test_env='distractingcs_04', final=True)
	evaluate(test_distractingcs05_env, agent, video, args.eval_episodes, L, step, test_env='distractingcs_05', final=True)
	L.dump(step * args.action_repeat)
	torch.save(agent, os.path.join(model_dir, f'{step * args.action_repeat}.pt'))
	print('Saved model')
	print('Completed training for', work_dir)
	wandb.finish()


if __name__ == '__main__':
	args = parse_args()
	main(args)
