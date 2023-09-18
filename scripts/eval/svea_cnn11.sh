cd ..
cd ..

seeds=(0 1 2 3 4)

for seed in "${seeds[@]}"; do
	CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 python3 src/eval.py \
		--algorithm svea \
		--eval_episodes 100 \
		--seed "$seed" \
		--group_name 'Hansen_repo' \
		--exp_name 'svea_cnn11_randomoverlay' \
		--encoder_type 'cnn' \
		--num_shared_layers 11 &
done

wait