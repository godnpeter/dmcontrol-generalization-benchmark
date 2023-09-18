cd ..
cd ..

python run_parallel.py \
    --group_name 'dmcgb_reset' \
    --exp_name 'svea_cnn11_sequential_combo_encoder_policy_reset' \
    --algorithm 'svea' \
    --num_seeds 3 \
    --num_exp_per_device 3 \
    --total_devices 4 \
    --hard_aug_type 'sequential_combo' \
    --num_shared_layers 11 \
    --do_policy_reset 'true' \
    --do_encoder_reset 'true' \
    --shrink_alpha 0.8 \
    --num_games 7