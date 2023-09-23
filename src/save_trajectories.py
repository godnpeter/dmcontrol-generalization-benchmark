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

def evaluate(env_dict, agent):
	episode_rewards = []
	eval_obs_dict = {}
	for i in tqdm(range(20)):
		trajectory_obs_dict = {}

		ep_agent = agent
		train_obs = env_dict['train'].reset()
		
		trajectory_obs_dict['train'] = [torch.from_numpy(np.array((train_obs))).float()]
		for eval_mode, eval_env in env_dict.items():
			if eval_mode == 'train':
				continue
			trajectory_obs_dict[eval_mode] = [torch.from_numpy(np.array(eval_env.reset())).float()]

		done = False
		episode_reward = 0
		while not done:
			with utils.eval_mode(ep_agent):
				action = ep_agent.select_action(train_obs)

			train_next_obs, reward, done, _ = env_dict['train'].step(action)

			trajectory_obs_dict['train'].append(torch.from_numpy(np.array((train_next_obs))).float())
			for eval_mode, eval_env in env_dict.items():
				if eval_mode == 'train':
					continue
				eval_next_obs, *_ = env_dict[eval_mode].step(action)
				trajectory_obs_dict[eval_mode].append(torch.from_numpy(np.array((eval_next_obs))).float())
    
			episode_reward += reward
			train_obs = train_next_obs
		
		eval_obs_dict[f'trajectory_{i}'] = trajectory_obs_dict

		episode_rewards.append(episode_reward)

	return eval_obs_dict


def main(args):
	# Set seed
	utils.set_seed_everywhere(args.seed)
	torch.set_num_threads(1)
	games = [('walker_walk')]
	
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

		eval_env_dict = {'train': train_env, 
      					'ch': test_ch_env, 
                   		've': test_ve_env, 
                     	'vh': test_vh_env, 
                      	'dcs01': test_distractingcs01_env, 
                       	'dcs02': test_distractingcs02_env,
      					'dcs03': test_distractingcs03_env, 
           				'dcs04': test_distractingcs04_env, 
               			'dcs05': test_distractingcs05_env}

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
		results_fp = os.path.join(work_dir, 'test_obs_dict.pt')

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

		test_obs_dict = evaluate(eval_env_dict, agent)
		ipdb.set_trace()
		# Save results
		torch.save({
			'test_obs_dict': test_obs_dict
		}, results_fp)
		print('Saved results to', results_fp)


if __name__ == '__main__':
	args = parse_args()
	main(args)
