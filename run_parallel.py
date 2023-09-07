import subprocess
import argparse
import json
import copy
import itertools
import os
import multiprocessing as mp
import multiprocessing
# from src.envs.atari import *
import numpy as np
import time
import ipdb


def run_script(script_name):
    print(script_name)
    subprocess.run(script_name, shell=True)

if __name__ == '__main__':
    '''
    total_devices: 해당 서버에 Gpu가 총 몇개인지
    내가 사용할 gpu 지정: CUDA_VISIBLE_DEVICES=0,1,3,4,6
    '''
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--num_seeds',    type=int,     default=5)
    parser.add_argument('--total_devices',  type=int,     default=8)  # 그 서버에 gpu가 몇개인지
    parser.add_argument('--num_exp_per_device',  type=int,  default=3)
    parser.add_argument('--algorithm',  type=str,  default='svea')
    parser.add_argument('--hard_aug_type', type=str, default='random_overlay')
    parser.add_argument('--num_shared_layers',  type=int,  default=11)
    parser.add_argument('--group_name', type=str, default='test')
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--num_games', type=int, default=7)
    parser.add_argument('--egl_device_id_equal_to_cuda_id', default=False, action='store_true')

    #parser.add_argument('--eval_mode', action='append',  default=[]) 

    args = vars(parser.parse_args())
    seeds = np.arange(args.pop('num_seeds')) 
    print('seeds:',seeds)
    print('algorithm: ', args['algorithm'])

    # gpu 몇개 사용할지는 CUDA_VISIBLE_DEVICES로 파악한다
    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    process_dict = {gpu_id: [] for gpu_id in available_gpus}
    print(f'available gpu_devices:{available_gpus}')

    num_devices = len(available_gpus)
    num_exp_per_device = args.pop('num_exp_per_device')
    pool_size = num_devices * num_exp_per_device 
        
    # number of game environment to run experiments
    num_games = args.pop('num_games')
    if num_games == 7:
        games = [('walker_walk'), ('walker_stand'), ('reacher_easy'), ('finger_spin'), \
                ('cheetah_run'),('cartpole_swingup'), ('cup_catch')]
    elif num_games == 4:
        games = [('walker_walk'), ('walker_stand'), ('reacher_easy'), ('cartpole_swingup') ]
    elif num_games == 3:
        games = [('cheetah_run'), ('finger_spin'), ('cup_catch') ]
    # create configurations for child run
    experiments = []
    for seed, game in itertools.product(*[seeds, games]):
        exp = copy.deepcopy(args)
        exp.pop('total_devices')
        exp['seed'] = seed
        exp['domain_name'], exp['task_name'] = game.split('_')

        if exp['domain_name'] == 'cup':
            exp['domain_name'] = 'ball_in_cup'

        if exp['domain_name'] == 'walker' and exp['task_name'] == 'walk':
            exp['action_repeat'] = 2
            exp['train_steps'] = '250k'
        elif exp['domain_name'] == 'finger' and exp['task_name'] == 'spin':
            exp['action_repeat'] = 2
            exp['train_steps'] = '250k'
        elif exp['domain_name'] == 'cartpole' and exp['task_name'] == 'swingup':
            exp['action_repeat'] = 8
            exp['train_steps'] = '62500'
        else:
            exp['action_repeat'] = 4
            exp['train_steps'] = '125k'
        
        experiments.append(exp)
        print(exp)
    # run parallell experiments
    # https://docs.python.org/3.5/library/multiprocessing.html#contexts-and-start-methods
    mp.set_start_method('spawn') 
    print('총 실험 개수:', len(experiments))

    # MUJOCO가 내가 지정한 gpu외에 다른 gpu를 먹지 않도록
    total_devices = args['total_devices']
    if total_devices == 4:
        
        egl_device = {
            '0':'1',
            '1':'0',
            '2':'3',
            '3':'2'
        }
    elif total_devices == 6:
        egl_device={
            '0': '5',
            '1': '4',
            '2': '3',
            '3': '2',
            '4': '1',
            '5': '0'
        }

    elif total_devices == 8:

        egl_device = {
            '0': '3',
            '1': '2',
            '2': '1',
            '3': '0',
            '4': '7',
            '5': '6',
            '6': '5',
            '7': '4'
        }
        
    
    for exp in experiments:
        wait = True
        while wait:
            for gpu_id, processes in process_dict.items():
                for process in processes:
                    if not process.is_alive():
                        print(f"Process {process.pid} on GPU {gpu_id} finished.")
                        processes.remove(process)
                        if gpu_id not in available_gpus:
                            available_gpus.append(gpu_id)
            
            for gpu_id, processes in process_dict.items():
                if len(processes) < num_exp_per_device:
                    wait = False    
                    break
            
            time.sleep(1)
        
        # get running processes in the gpu
        processes = process_dict[gpu_id]

        if args['egl_device_id_equal_to_cuda_id']:
            m_e_d_i = str(gpu_id)
        else:
            m_e_d_i = egl_device[gpu_id]
        
        cmd = 'CUDA_VISIBLE_DEVICES={} MUJOCO_EGL_DEVICE_ID={} \
                python src/train.py \
                --group_name={} \
                --exp_name={} \
                --algorithm={} \
                --domain_name={} \
                --task_name={} \
                --action_repeat={} \
                --train_steps={} \
                --seed={} \
                --hard_aug_type={} \
                --num_shared_layers={} \
                --save_video \
                '.format(
                    str(gpu_id),
                    str(m_e_d_i),
                    exp['group_name'],
                    exp['exp_name'],
                    exp['algorithm'],
                    exp['domain_name'],
                    exp['task_name'],
                    exp['action_repeat'],
                    exp['train_steps'],
                    exp['seed'],
                    exp['hard_aug_type'],
                    exp['num_shared_layers']
                )
        
        print(cmd)

        process = multiprocessing.Process(target=run_script, args=(cmd,))
        process.start()
        processes.append(process)

        # check if the GPU has reached its maximum number of processes
        if len(processes) == num_exp_per_device:
            available_gpus.remove(gpu_id)
