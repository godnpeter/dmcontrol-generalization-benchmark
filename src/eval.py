import torch
import torchvision
import os
import numpy as np
import gym
import utils
from copy import deepcopy
from tqdm import tqdm
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from video import VideoRecorder
import ipdb
import augmentations
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

def evaluate(env, agent, video, eval_mode='distractingcs', intensity=0.0):
	episode_rewards = []
	for i in tqdm(range(100)):

		ep_agent = agent
		obs = env.reset()
		if i % 100 == 0:
			video.init(enabled=True)
		else:
			video.init(enabled=False)
		done = False
		episode_reward = 0
		while not done:
			with utils.eval_mode(ep_agent):
				action = ep_agent.select_action(obs)
			next_obs, reward, done, _ = env.step(action)
			video.record(env, eval_mode)
			episode_reward += reward
			obs = next_obs

		if i % 100 == 0:
			video.save(f'eval_{eval_mode}{intensity}_{i}.mp4')
		
		episode_rewards.append(episode_reward)

	return np.mean(episode_rewards)


def main(args):
	# Set seed
	utils.set_seed_everywhere(args.seed)
	torch.set_num_threads(1)
	games = [('walker_walk'), ('walker_stand'), ('reacher_easy'), ('finger_spin'), \
                ('cheetah_run'),('cartpole_swingup'), ('cup_catch')]
	
	for game in games:
		args.domain_name, args.task_name = game.split('_')

		if args.domain_name == 'cup':
			args.domain_name = 'ball_in_cup'

		if args.domain_name == 'walker' and args.task_name == 'walk':
			args.action_repeat = 2
		elif args.domain_name == 'finger' and args.task_name == 'spin':
			args.action_repeat = 2
		elif args.domain_name == 'cartpole' and args.task_name == 'swingup':
			args.action_repeat = 8
		else:
			args.action_repeat = 4

		# Initialize environments
		gym.logger.set_level(40)

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
		
		print('Working directory:', work_dir)
		if not os.path.exists(work_dir):
			print('specified working directory does not exist')
			continue
		
		#assert os.path.exists(work_dir), 'specified working directory does not exist'
		model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
		video_dir = utils.make_dir(os.path.join(work_dir, 'distracting_video'))
		video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)

		# Check if evaluation has already been run
		#if args.eval_mode == 'distracting_cs':
		results_fp = os.path.join(work_dir, 'distractingcs_evalresults.pt')

		if os.path.exists(results_fp):
			print('specified run has already been done')
			continue
		#else:
		#		results_fp = os.path.join(work_dir, args.eval_mode+'.pt')
		#assert not os.path.exists(results_fp), f'{args.eval_mode} results already exist for {work_dir}'

		# Prepare agent
		assert torch.cuda.is_available(), 'must have cuda enabled'
		cropped_obs_shape = (3*args.frame_stack, args.image_crop_size, args.image_crop_size)
		agent = make_agent(
			obs_shape=cropped_obs_shape,
			action_shape=test_distractingcs02_env.action_space.shape,
			args=args
		)
		agent = torch.load(os.path.join(model_dir, str(500000)+'.pt'))
		agent.train(False)

		print(f'\nEvaluating {work_dir} for {100} episodes (mode: {test_distractingcs01_env._mode})')
		distractingcs_01_reward = evaluate(test_distractingcs01_env, agent, video, intensity=0.1)
		print(f'\nEvaluating {work_dir} for {100} episodes (mode: {test_distractingcs02_env._mode})')
		distractingcs_02_reward = evaluate(test_distractingcs02_env, agent, video, intensity=0.2)
		print(f'\nEvaluating {work_dir} for {100} episodes (mode: {test_distractingcs03_env._mode})')
		distractingcs_03_reward = evaluate(test_distractingcs03_env, agent, video, intensity=0.3)
		print(f'\nEvaluating {work_dir} for {100} episodes (mode: {test_distractingcs04_env._mode})')
		distractingcs_04_reward = evaluate(test_distractingcs04_env, agent, video, intensity=0.4)
		print(f'\nEvaluating {work_dir} for {100} episodes (mode: {test_distractingcs05_env._mode})')
		distractingcs_05_reward = evaluate(test_distractingcs05_env, agent, video, intensity=0.5)
		print('Distractingcs 01 Reward:', int(distractingcs_01_reward))
		print('Distractingcs 02 Reward:', int(distractingcs_02_reward))
		print('Distractingcs 03 Reward:', int(distractingcs_02_reward))
		print('Distractingcs 04 Reward:', int(distractingcs_04_reward))
		print('Distractingcs 05 Reward:', int(distractingcs_05_reward))

		# Save results
		torch.save({
			'args': args,
			'distractingcs 01 reward': distractingcs_01_reward,
			'distractingcs 02 reward': distractingcs_02_reward,
			'distractingcs 03 reward': distractingcs_03_reward,
			'distractingcs 04 reward': distractingcs_04_reward,
			'distractingcs 05 reward': distractingcs_05_reward
		}, results_fp)
		print('Saved results to', results_fp)


if __name__ == '__main__':
	args = parse_args()
	main(args)
