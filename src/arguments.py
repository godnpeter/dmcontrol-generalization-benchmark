import argparse
import numpy as np
from ast import literal_eval

def str2bool(v):
    """Convert string representation of boolean to actual boolean."""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
	parser = argparse.ArgumentParser()

	# environment
	parser.add_argument('--domain_name', default='walker')
	parser.add_argument('--task_name', default='walk')
	parser.add_argument('--frame_stack', default=3, type=int)
	parser.add_argument('--action_repeat', default=4, type=int)
	parser.add_argument('--episode_length', default=1000, type=int)
	parser.add_argument('--eval_mode', default='color_hard', type=str)
	
	# agent
	parser.add_argument('--algorithm', default='sac', type=str)
	parser.add_argument('--train_steps', default='125k', type=str)
	parser.add_argument('--discount', default=0.99, type=float)
	parser.add_argument('--init_steps', default=1000, type=int)
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--hidden_dim', default=1024, type=int)
	parser.add_argument('--hard_aug_type', default='random_overlay', type=str)
	parser.add_argument("--combo_aug_type_list", type=str, default='random_shift,random_conv,random_overlay')
	parser.add_argument('--train_with_distraction', default='False', type=str2bool)
	parser.add_argument("--train_env_list", type=str, default='train,color_hard,video_easy')

	# optimizer
	parser.add_argument('--optimizer_type', default='adam', type=str)

	# actor
	parser.add_argument('--actor_lr', default=1e-3, type=float)
	parser.add_argument('--actor_beta', default=0.9, type=float)
	parser.add_argument('--actor_log_std_min', default=-10, type=float)
	parser.add_argument('--actor_log_std_max', default=2, type=float)
	parser.add_argument('--actor_update_freq', default=2, type=int)

	# critic
	parser.add_argument('--critic_lr', default=1e-3, type=float)
	parser.add_argument('--critic_beta', default=0.9, type=float)
	parser.add_argument('--critic_tau', default=0.01, type=float)
	parser.add_argument('--critic_target_update_freq', default=2, type=int)

	# architecture
	parser.add_argument('--num_shared_layers', default=11, type=int)
	parser.add_argument('--num_head_layers', default=0, type=int)
	parser.add_argument('--num_filters', default=32, type=int)
	parser.add_argument('--projection_dim', default=100, type=int)
	parser.add_argument('--encoder_tau', default=0.05, type=float)
	parser.add_argument('--encoder_type', default='cnn', type=str)
	parser.add_argument('--width_expansion', default=1, type=int)

	# entropy maximization
	parser.add_argument('--init_temperature', default=0.1, type=float)
	parser.add_argument('--alpha_lr', default=1e-4, type=float)
	parser.add_argument('--alpha_beta', default=0.5, type=float)

	# reset
	parser.add_argument('--reset_interval_steps', default=25000, type=int)
	parser.add_argument('--do_policy_reset', default='False', type=str2bool)
	parser.add_argument('--do_encoder_reset', default='False', type=str2bool)
	parser.add_argument('--shrink_alpha', default=0.8, type=float)
	parser.add_argument('--critic_target_reset', default='copy_critic', type=str)
	parser.add_argument('--replay_ratio', default=1, type=int)

	# auxiliary tasks
	parser.add_argument('--aux_lr', default=1e-3, type=float)
	parser.add_argument('--aux_beta', default=0.9, type=float)
	parser.add_argument('--aux_update_freq', default=2, type=int)

	# soda
	parser.add_argument('--soda_batch_size', default=256, type=int)
	parser.add_argument('--soda_tau', default=0.005, type=float)

	# svea
	parser.add_argument('--svea_alpha', default=0.5, type=float)
	parser.add_argument('--svea_beta', default=0.5, type=float)

	# eval
	parser.add_argument('--save_freq', default='100k', type=str)
	parser.add_argument('--eval_freq', default='50k', type=str)
	parser.add_argument('--eval_episodes', default=10, type=int)
	parser.add_argument('--distracting_cs_intensity', default=0., type=float)

	# misc
	parser.add_argument('--seed', default=None, type=int)
	parser.add_argument('--log_dir', default='logs', type=str)
	parser.add_argument('--save_video', default=True, action='store_true')

	# logger 
	parser.add_argument('--group_name', default='test', type=str)
	parser.add_argument('--exp_name', default='test', type=str)

	args = parser.parse_args()

	assert args.algorithm in {'sac', 'rad', 'curl', 'pad', 'soda', 'drq', 'svea', 'svea_idm', 'svea_bc'}, f'specified algorithm "{args.algorithm}" is not supported'

	assert args.eval_mode in {'train', 'color_easy', 'color_hard', 'video_easy', 'video_hard', 'distracting_cs', 'none'}, f'specified mode "{args.eval_mode}" is not supported'
	assert args.seed is not None, 'must provide seed for experiment'
	assert args.log_dir is not None, 'must provide a log directory for experiment'

	intensities = {0., 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}
	assert args.distracting_cs_intensity in intensities, f'distracting_cs has only been implemented for intensities: {intensities}'

	args.train_steps = int(args.train_steps.replace('k', '000'))
	args.save_freq = int(args.save_freq.replace('k', '000'))
	args.eval_freq = int(args.eval_freq.replace('k', '000'))

	# Access and use the arguments
	args.combo_aug_type_list = args.combo_aug_type_list.split(',')
	if args.hard_aug_type == 'combo':
		print("Combo Aug Type List:", args.combo_aug_type_list)

	args.train_env_list = args.train_env_list.split(',')
	if args.train_with_distraction:
		print("Train env List: ", args.train_env_list)

	if args.eval_mode == 'none':
		args.eval_mode = None

	if args.algorithm in {'rad', 'curl', 'pad', 'soda'}:
		args.image_size = 100
		args.image_crop_size = 84
	else:
		args.image_size = 84
		args.image_crop_size = 84
	
	return args
