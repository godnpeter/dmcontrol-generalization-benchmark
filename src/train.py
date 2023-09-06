import torch
import os
import numpy as np
import gym
import utils
import time
import wandb
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
	env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='train'
	)
	test_ce_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='color_easy',
		intensity=args.distracting_cs_intensity
	) #if args.eval_mode is not None else None

	test_ch_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='color_hard',
		intensity=args.distracting_cs_intensity
	) #if args.eval_mode is not None else None

	test_ve_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='video_easy',
		intensity=args.distracting_cs_intensity
	) #if args.eval_mode is not None else None

	test_vh_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='video_hard',
		intensity=args.distracting_cs_intensity
	) #if args.eval_mode is not None else None

	# Create working directory
	work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, str(args.seed))
	print('Working directory:', work_dir)
	assert not os.path.exists(os.path.join(work_dir, 'train.log')), 'specified working directory already exists'
	utils.make_dir(work_dir)
	model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
	utils.write_info(args, os.path.join(work_dir, 'info.log'))

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	replay_buffer = utils.ReplayBuffer(
		obs_shape=env.observation_space.shape,
		action_shape=env.action_space.shape,
		capacity=args.train_steps,
		batch_size=args.batch_size
	)
	cropped_obs_shape = (3*args.frame_stack, args.image_crop_size, args.image_crop_size)
	print('Observations:', env.observation_space.shape)
	print('Cropped observations:', cropped_obs_shape)
	agent = make_agent(
		obs_shape=cropped_obs_shape,
		action_shape=env.action_space.shape,
		args=args
	)

	start_step, episode, episode_reward, done = 0, 0, 0, True
	L = Logger(work_dir, args)
	start_time = time.time()
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
				evaluate(env, agent, video, args.eval_episodes, L, step * args.action_repeat)
				evaluate(test_ce_env, agent, video, args.eval_episodes, L, step * args.action_repeat, test_env='color_easy')
				evaluate(test_ch_env, agent, video, args.eval_episodes, L, step * args.action_repeat, test_env='color_hard')
				evaluate(test_ve_env, agent, video, args.eval_episodes, L, step * args.action_repeat, test_env='video_easy')
				evaluate(test_vh_env, agent, video, args.eval_episodes, L, step * args.action_repeat, test_env='video_hard')
				L.log('eval/duration', time.time() - eval_start_time, step * args.action_repeat)
				L.dump(step * args.action_repeat)
			# Save agent periodically
			#if step > start_step and step % args.save_freq == 0:
				#torch.save(agent, os.path.join(model_dir, f'{step * args.action_repeat}.pt'))

			L.log('train/episode_return', episode_reward, step * args.action_repeat)

			obs = env.reset()
			done = False
			episode_reward = 0
			episode_step = 0
			episode += 1

			L.log('train/episode', episode, step * args.action_repeat)

		# Sample action for data collection
		if step < args.init_steps:
			action = env.action_space.sample()
		else:
			with utils.eval_mode(agent):
				action = agent.sample_action(obs)

		# Run training update
		if step >= args.init_steps:
			num_updates = args.init_steps if step == args.init_steps else 1
			for _ in range(num_updates):
				agent.update(replay_buffer, L, step)

		# Take step
		next_obs, reward, done, _ = env.step(action)
		done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
		replay_buffer.add(obs, action, reward, next_obs, done_bool)
		episode_reward += reward
		obs = next_obs

		episode_step += 1

	evaluate(test_ce_env, agent, video, args.eval_episodes, L, step, test_env='color_easy', final=True)
	evaluate(test_ch_env, agent, video, args.eval_episodes, L, step, test_env='color_hard', final=True)
	evaluate(test_ve_env, agent, video, args.eval_episodes, L, step, test_env='video_easy', final=True)
	evaluate(test_vh_env, agent, video, args.eval_episodes, L, step, test_env='video_hard', final=True)
	L.dump(step * args.action_repeat)
	torch.save(agent, os.path.join(model_dir, f'{step * args.action_repeat}.pt'))
	print('Saved model')
	print('Completed training for', work_dir)
	wandb.finish()


if __name__ == '__main__':
	args = parse_args()
	main(args)
