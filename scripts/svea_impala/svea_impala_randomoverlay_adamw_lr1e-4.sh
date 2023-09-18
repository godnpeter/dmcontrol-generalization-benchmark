cd ..
cd ..

python run_parallel.py \
    --group_name 'Hansen_repo' \
    --exp_name 'svea_impala_randomoverlay_adamw_lr1e-4' \
    --algorithm 'svea' \
    --num_seeds 5 \
    --num_exp_per_device 3 \
    --total_devices 8 \
    --hard_aug_type 'random_overlay' \
    --encoder_type 'impala' \
    --optimizer_type 'adamw' \
    --actor_lr 0.0001 \
    --critic_lr 0.0001