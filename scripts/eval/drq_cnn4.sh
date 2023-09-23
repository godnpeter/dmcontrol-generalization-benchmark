cd ..
cd ..

seeds=(0 1 2 3 4)

for seed in "${seeds[@]}"; do
	CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 python3 src/save_trajectories.py \
		--algorithm svea \
		--eval_episodes 10 \
		--seed "$seed" \
		--group_name 'Hansen_repo' \
		--exp_name 'drq_cnn4_none' \
		--encoder_type 'cnn' \
		--num_shared_layers 4 \
		--hard_aug_type 'none' \
		--domain_name 'cartpole' \
		--task_name 'swing_up' \
		--algorithm 'drq'
done

wait