cd ..
cd ..

python run_parallel.py \
    --group_name 'Hansen_repo' \
    --exp_name 'svea_cnn11_combo_encoder_policy_reset' \
    --algorithm 'svea' \
    --num_seeds 5 \
    --num_exp_per_device 3 \
    --total_devices 8 \
    --hard_aug_type 'combo' \
    --num_shared_layers 11 \
    --do_policy_reset 'true' \
    --do_encoder_reset 'true' \
    --shrink_alpha 0.8 \
    --num_games 5