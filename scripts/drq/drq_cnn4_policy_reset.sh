cd ..
cd ..

python run_parallel.py \
    --group_name 'dmcgb_reset' \
    --exp_name 'drq_cnn4_none_policy_reset_rr1' \
    --algorithm 'drq' \
    --num_seeds 5 \
    --num_exp_per_device 3 \
    --total_devices 4 \
    --hard_aug_type 'none' \
    --num_shared_layers 4 \
    --do_policy_reset 'true' \
    --do_encoder_reset 'false'