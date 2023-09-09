cd ..
cd ..

python run_parallel.py \
    --group_name 'test' \
    --exp_name 'test' \
    --algorithm 'svea' \
    --num_seeds 10 \
    --num_exp_per_device 1 \
    --total_devices 8 \
    --hard_aug_type 'random_overlay' \
    --num_shared_layers 4