cd ..
cd ..

python run_distracting_parallel.py \
    --group_name 'train_envs:clean,color_hard,video_easy' \
    --exp_name 'drq_cnn11' \
    --algorithm 'drq' \
    --num_seeds 5 \
    --num_exp_per_device 3 \
    --total_devices 8 \
    --hard_aug_type 'none' \
    --num_shared_layers 11 \
    --num_games 4 \
    --train_with_distraction True \
    --train_env_list 'train,color_hard,video_easy'

python run_distracting_parallel.py \
    --group_name 'train_envs:clean,color_hard' \
    --exp_name 'drq_cnn11' \
    --algorithm 'drq' \
    --num_seeds 5 \
    --num_exp_per_device 3 \
    --total_devices 8 \
    --hard_aug_type 'none' \
    --num_shared_layers 11 \
    --num_games 4 \
    --train_with_distraction True \
    --train_env_list 'train,color_hard'


python run_distracting_parallel.py \
    --group_name 'train_envs:clean,color_hard,video_easy' \
    --exp_name 'svea_cnn11_randomoverlay' \
    --algorithm 'svea' \
    --num_seeds 5 \
    --num_exp_per_device 3 \
    --total_devices 8 \
    --hard_aug_type 'random_overlay' \
    --num_shared_layers 11 \
    --num_games 4 \
    --train_with_distraction True \
    --train_env_list 'train,color_hard,video_easy'


python run_distracting_parallel.py \
    --group_name 'train_envs:clean,color_hard' \
    --exp_name 'svea_cnn11_randomoverlay' \
    --algorithm 'svea' \
    --num_seeds 5 \
    --num_exp_per_device 3 \
    --total_devices 8 \
    --hard_aug_type 'random_overlay' \
    --num_shared_layers 11 \
    --num_games 4 \
    --train_with_distraction True \
    --train_env_list 'train,color_hard'