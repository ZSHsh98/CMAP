export CUDA_VISIBLE_DEVICES=0

export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Linf Attacks
for ATK in AA_Attack PGD_EOT BPDA_EOT;do
    python eval_wb_adv_purification_cmap.py \
        --config "cifar10.yml" \
        --t 1000 \
        --epsilon 0.03137 \
        --lr 2 \
        --adv_batch_size 10 \
        --num_sub 500 \
        --num_steps 200 \
        --domain "cifar10" \
        --classifier_name "cifar10-wideresnet-28-10" \
        --attack_method "$ATK" \
        --attack_version "standard" \
        --vote_type "hard" \
        --lp_norm "Linf" \
        --iterations 200 \
        --k_samples 10 \
        --scale_factor 1 \
        --similar_factor 2 \
        --gauss_factor 0.0005
done

# L2 Attacks
for ATK in AA_Attack_L2 PGD_EOT_L2;do
    python eval_wb_adv_purification_cmap.py \
        --config "cifar10.yml" \
        --t 1000 \
        --epsilon 0.5 \
        --lr 2 \
        --adv_batch_size 10 \
        --num_sub 500 \
        --num_steps 200 \
        --domain "cifar10" \
        --classifier_name "cifar10-wideresnet-28-10" \
        --attack_method "$ATK" \
        --attack_version "standard" \
        --vote_type "hard" \
        --lp_norm "L2" \
        --iterations 200 \
        --k_samples 10 \
        --scale_factor 1 \
        --similar_factor 2 \
        --gauss_factor 0.0005
done

# Clean
python eval_wb_adv_purification_cmap.py \
    --config "cifar10.yml" \
    --t 1000 \
    --epsilon 0 \
    --lr 2 \
    --adv_batch_size 10 \
    --num_sub 500 \
    --num_steps 200 \
    --domain "cifar10" \
    --classifier_name "cifar10-wideresnet-28-10" \
    --vote_type "hard" \
    --lp_norm "L2" \
    --iterations 200 \
    --k_samples 10 \
    --scale_factor 1 \
    --similar_factor 2 \
    --gauss_factor 0.0005