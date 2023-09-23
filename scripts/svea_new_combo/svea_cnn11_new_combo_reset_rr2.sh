cd ..
cd ..

python run_parallel.py \
    --group_name 'svea_new_combo' \
    --exp_name 'svea_impala_wd2_combo_7types_encoder_policy_reset_rr2' \
    --algorithm 'svea' \
    --num_seeds 5 \
    --num_exp_per_device 3 \
    --total_devices 8 \
    --hard_aug_type 'combo' \
    --combo_aug_type_list random_shift random_conv random_overlay cutout_color flip rotate projection_transformation \
    --encoder_type 'impala' \
    --width_expansion 2 \
    --do_policy_reset 'true' \
    --do_encoder_reset 'true' \
    --shrink_alpha 0.5 \
    --replay_ratio 2 \
    --reset_interval_steps 100000