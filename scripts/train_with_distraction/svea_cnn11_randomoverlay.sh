cd ..
cd ..

python run_distracting_parallel.py \
    --group_name 'Hansen_repo' \
    --exp_name 'svea_cnn11_randomoverlay' \
    --algorithm 'svea' \
    --num_seeds 5 \
    --num_exp_per_device 2 \
    --total_devices 8 \
    --hard_aug_type 'random_overlay' \
    --num_shared_layers 11