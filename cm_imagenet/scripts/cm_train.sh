# export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_IGNORE_DISABLED_P2P=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PATH="/usr/local/mpich-4.1.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/mpich-4.1.2/lib:$LD_LIBRARY_PATH"

# imagenet 64
# mpiexec -n 8 python cm_train.py --training_mode consistency_training --target_ema_mode adaptive \
#     --start_ema 0.95 --scale_mode progressive --start_scales 2 --end_scales 200 \
#     --total_training_steps 800000 --loss_norm lpips --lr_anneal_steps 0 --teacher_model_path /path/to/edm_imagenet64_ema.pt \
#     --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 \
#     --teacher_dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 2048 \
#     --image_size 64 --lr 0.0001 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
#     --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 \
#     --weight_schedule uniform --data_dir /path/to/imagenet64

mpiexec -n 8 python scripts/cm_train.py --training_mode consistency_training --target_ema_mode adaptive \
    --start_ema 0.95 --scale_mode progressive --start_scales 2 --end_scales 200 \
    --total_training_steps 800000 --loss_norm lpips --lr_anneal_steps 0 \
    --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm True --dropout 0.0 \
    --teacher_dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 1024 \
    --image_size 64 --lr 0.00001 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
    --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 \
    --weight_schedule uniform --data_dir /zhangshuhai/imagenet100

# LSUN 256
# mpiexec -n 8 python scripts/cm_train.py --training_mode consistency_training --target_ema_mode adaptive --start_ema 0.95 --scale_mode progressive --start_scales 2 \
#     --end_scales 150 --total_training_steps 1000000 --loss_norm lpips --lr_anneal_steps 0  \
#     --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --teacher_dropout 0.1 --ema_rate 0.9999,0.99994,0.9999432189950708  \
#     --global_batch_size 96 --image_size 256 --lr 0.00005 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --schedule_sampler uniform \
#     --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir /zhangshuhai/imagenet100 #/path/to/bedroom256

# imagenet100 256
# mpiexec -n 4 python scripts/cm_train.py --training_mode consistency_training --target_ema_mode adaptive \
#     --start_ema 0.95 --scale_mode progressive --start_scales 2 --end_scales 150 \
#     --total_training_steps 100 --loss_norm lpips --lr_anneal_steps 1 \
#     --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm True --dropout 0.0 \
#     --teacher_dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 8 \
#     --image_size 256 --lr 0.00005 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 \
#     --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 \
#     --weight_schedule uniform --data_dir /zhangshuhai/imagenet100