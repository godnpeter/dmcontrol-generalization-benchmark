cd ..
cd ..

python run_parallel.py \
    --group_name 'Hansen_repo' \
    --exp_name 'drq_cnn4_none' \
    --algorithm 'drq' \
    --num_seeds 5 \
    --num_exp_per_device 3 \
    --total_devices 6 \
    --hard_aug_type 'none' \
    --num_shared_layers 4 \
    --num_games 4 \
    --egl_device_id_equal_to_cuda_id