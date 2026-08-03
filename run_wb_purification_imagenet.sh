export CUDA_VISIBLE_DEVICES=0

export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PAT

export PATH="mpich-install/bin:$PATH"
export LD_LIBRARY_PATH="mpich-install/lib:$LD_LIBRARY_PATH"

# Linf Attacks
for ATK in PGD_EOT BPDA_EOT;do
    python eval_wb_adv_purification_cmap.py \
        --config "imagenet.yml" \
        --t 1000 \
        --atk_t_cm 5 \
        --epsilon 0.01569 \
        --lr 2 \
        --adv_batch_size 5 \
        --num_sub 500 \
        --domain "imagenet100" \
        --classifier_name "imagenet100-resnet50" \
        --attack_method "$ATK" \
        --attack_version "standard" \
        --vote_type "hard" \
        --lp_norm "Linf" \
        --iterations 300 \
        --k_samples 5 \
        --scale_factor 0.005 \
        --similar_factor 0.5 \
        --gauss_factor 0.0002
done

# L2 Attacks
for ATK in PGD_EOT_L2;do
    python eval_wb_adv_purification_cmap.py \
        --config "imagenet.yml" \
        --t 1000 \
        --atk_t_cm 5 \
        --epsilon 0.5 \
        --lr 2 \
        --adv_batch_size 5 \
        --num_sub 500 \
        --domain "imagenet100" \
        --classifier_name "imagenet100-resnet50" \
        --attack_method "$ATK" \
        --attack_version "standard" \
        --vote_type "hard" \
        --lp_norm "L2" \
        --iterations 300 \
        --k_samples 5 \
        --scale_factor 0.005 \
        --similar_factor 0.5 \
        --gauss_factor 0.0002
done

# Clean
python eval_wb_adv_purification_cmap.py \
    --config "imagenet.yml" \
    --t 1000 \
    --atk_t_cm 5 \
    --epsilon 0 \
    --lr 2 \
    --adv_batch_size 5 \
    --num_sub 500 \
    --domain "imagenet100" \
    --classifier_name "imagenet100-resnet50" \
    --vote_type "hard" \
    --lp_norm "Linf" \
    --iterations 300 \
    --k_samples 5 \
    --scale_factor 0.005 \
    --similar_factor 0.5 \
    --gauss_factor 0.0002