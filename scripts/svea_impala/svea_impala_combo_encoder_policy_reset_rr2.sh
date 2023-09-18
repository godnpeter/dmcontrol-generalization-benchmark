cd ..
cd ..

python run_parallel.py \
    --group_name 'dmcgb_reset' \
    --exp_name 'svea_impala_combo_encoder_policy_reset_rr2_resetinterval1e5' \
    --algorithm 'svea' \
    --num_seeds 5 \
    --num_exp_per_device 3 \
    --total_devices 8 \
    --hard_aug_type 'combo' \
    --encoder_type 'impala' \
    --replay_ratio 2 \
    --shrink_alpha 0.5 \
    --reset_interval_steps 100000 \
    --do_policy_reset 'true' \
    --do_encoder_reset 'true' 