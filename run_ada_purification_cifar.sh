export CUDA_VISIBLE_DEVICES=0

export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Linf Attacks
python eval_adaptive_adv_purification_cmap.py \
    --config "cifar10.yml" \
    --t 1000 \
    --epsilon 0.03137 \
    --lr 2 \
    --adv_batch_size 5 \
    --num_sub 500 \
    --domain "cifar10" \
    --classifier_name "cifar10-wideresnet-28-10" \
    --vote_type "hard" \
    --lp_norm "Linf" \
    --iterations 1000 \
    --k_samples 10 \
    --adv_factor 2000000 \
    --scale_factor 1 \
    --similar_factor 2 \
    --gauss_factor 0.0005

# L2 Attacks
python eval_adaptive_adv_purification_cmap.py \
    --config "cifar10.yml" \
    --t 1000 \
    --epsilon 0.5 \
    --lr 2 \
    --adv_batch_size 5 \
    --num_sub 500 \
    --domain "cifar10" \
    --classifier_name "cifar10-wideresnet-28-10" \
    --vote_type "hard" \
    --lp_norm "L2" \
    --iterations 1000 \
    --k_samples 10 \
    --adv_factor 1000000000 \
    --scale_factor 1 \
    --similar_factor 2 \
    --gauss_factor 0.0005