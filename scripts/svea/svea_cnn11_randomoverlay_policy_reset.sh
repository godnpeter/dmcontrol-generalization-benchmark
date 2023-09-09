cd ..
cd ..

python run_parallel.py \
    --group_name 'dmcgb_reset' \
    --exp_name 'svea_cnn11_randomoverlay_policy_reset_rr1' \
    --algorithm 'svea' \
    --num_seeds 5 \
    --num_exp_per_device 3 \
    --total_devices 8 \
    --hard_aug_type 'random_overlay' \
    --num_shared_layers 11 \
    --do_policy_reset 'true' \
    --do_encoder_reset 'false'