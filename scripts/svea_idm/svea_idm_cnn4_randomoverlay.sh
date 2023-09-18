cd ..
cd ..

python run_parallel.py \
    --group_name 'svea_idm' \
    --exp_name 'svea_idm_cnn4_randomoverlay_idmrandomoverlay' \
    --algorithm 'svea_idm' \
    --num_seeds 5 \
    --num_games 4 \
    --num_exp_per_device 3 \
    --total_devices 4 \
    --hard_aug_type 'random_overlay' \
    --num_shared_layers 4 \
    --do_policy_reset 'false' \
    --do_encoder_reset 'false'