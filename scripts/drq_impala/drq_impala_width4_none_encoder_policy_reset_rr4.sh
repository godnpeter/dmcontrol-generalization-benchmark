cd ..
cd ..

python run_parallel.py \
    --group_name 'dmcgb_impala_reset' \
    --exp_name 'drq_impala_width4_none_encoder_policy_reset_rr4' \
    --algorithm 'drq' \
    --num_seeds 5 \
    --num_exp_per_device 2 \
    --total_devices 8 \
    --hard_aug_type 'none' \
    --encoder_type 'impala' \
    --width_expansion 4 \
    --replay_ratio 4 \
    --shrink_alpha 0.5 \
    --reset_interval_steps 50000 \
    --do_policy_reset 'true' \
    --do_encoder_reset 'true' 