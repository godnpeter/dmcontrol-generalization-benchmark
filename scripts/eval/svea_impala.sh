cd ..
cd ..

seeds=(0 1 2 3 4)

for seed in "${seeds[@]}"; do
	CUDA_VISIBLE_DEVICES=4 MUJOCO_EGL_DEVICE_ID=4 python3 src/eval.py \
		--algorithm svea \
		--eval_episodes 100 \
		--seed "$seed" \
		--group_name 'Hansen_repo' \
		--exp_name 'svea_impala_randomoverlay' \
		--encoder_type 'impala' &
done

wait