cd ..
cd ..

python run_parallel.py \
    --group_name 'Hansen_repo' \
    --exp_name 'drq_cnn11_none' \
    --algorithm 'drq' \
    --num_seeds 5 \
    --num_exp_per_device 3 \
    --total_devices 8 \
    --hard_aug_type 'none' \
    --num_shared_layers 11