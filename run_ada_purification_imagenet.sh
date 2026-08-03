export CUDA_VISIBLE_DEVICES=0

export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export PATH="mpich-install/bin:$PATH"
export LD_LIBRARY_PATH="mpich-install/lib:$LD_LIBRARY_PATH"

# Linf Attacks
python eval_sde_adv_antiodepure_noise.py \
    --config "imagenet.yml" \
    --t 1000 \
    --epsilon 0.01569 \
    --lr 2 \
    --adv_batch_size 10 \
    --num_sub 500 \
    --domain "imagenet100" \
    --classifier_name "imagenet100-resnet50" \
    --vote_type "hard" \
    --lp_norm "Linf" \
    --iterations 1000 \
    --k_samples 5 \
    --adv_factor 2 \
    --scale_factor 0.005 \
    --similar_factor 0.5 \
    --gauss_factor 0.0002

# L2 Attacks
python eval_sde_adv_antiodepure_noise.py \
    --config "imagenet.yml" \
    --t 1000 \
    --epsilon 0.5 \
    --lr 2 \
    --adv_batch_size 10 \
    --num_sub 500 \
    --domain "imagenet100" \
    --classifier_name "imagenet100-resnet50" \
    --vote_type "hard" \
    --lp_norm "L2" \
    --iterations 1000 \
    --k_samples 5 \
    --adv_factor 50000 \
    --scale_factor 0.005 \
    --similar_factor 0.5 \
    --gauss_factor 0.0002