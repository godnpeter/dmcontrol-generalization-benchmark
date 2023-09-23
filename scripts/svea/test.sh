cd ..
cd ..

python run_parallel.py \
    --group_name 'test' \
    --exp_name 'test_cnn2' \
    --algorithm 'svea' \
    --num_seeds 100 \
    --num_exp_per_device 3 \
    --total_devices 8 \
    --hard_aug_type 'random_overlay' \
    --num_shared_layers 2 