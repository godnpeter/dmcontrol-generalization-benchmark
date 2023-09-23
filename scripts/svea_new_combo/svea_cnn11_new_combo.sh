cd ..
cd ..

python run_parallel.py \
    --group_name 'svea_new_combo' \
    --exp_name 'svea_cnn11_combo_7types' \
    --algorithm 'svea' \
    --num_seeds 5 \
    --num_exp_per_device 3 \
    --total_devices 8 \
    --hard_aug_type 'combo' \
    --combo_aug_type_list random_shift random_conv random_overlay cutout_color flip rotate projection_transformation \
    --num_shared_layers 11