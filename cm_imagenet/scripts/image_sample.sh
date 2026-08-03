export CUDA_VISIBLE_DEVICES=1
export PATH="/usr/local/mpich-4.1.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/mpich-4.1.2/lib:$LD_LIBRARY_PATH"

# imagenet64
## onestep
mpiexec -n 1 python scripts/image_sample.py --batch_size 16 --training_mode consistency_distillation \
    --sampler onestep --model_path cm_train_uncond_imagenet100_log/model200000.pt --attention_resolutions 32,16,8 \
    --class_cond False --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 \
    --num_head_channels 64 --num_res_blocks 3 --num_samples 64 --resblock_updown True \
    --use_fp16 True --weight_schedule uniform

# mpiexec -n 1 python scripts/image_sample.py --batch_size 64 --training_mode consistency_training \
#     --sampler multistep --ts 0,106,200 --steps 201 --model_path cm_train_uncond_imagenet100_log/model200000.pt --attention_resolutions 32,16,8 \
#     --class_cond False --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 \
#     --num_head_channels 64 --num_res_blocks 3 --num_samples 64 --resblock_updown True \
#     --use_fp16 True --weight_schedule uniform

# mpiexec -n 1 python scripts/image_sample.py --batch_size 64 --training_mode consistency_training \
#     --sampler multistep --ts 0,20,40,60,80,100,120,140,160,180,200 --steps 201 --model_path cm_train_uncond_imagenet100_log/model200000.pt --attention_resolutions 32,16,8 \
#     --class_cond False --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 \
#     --num_head_channels 64 --num_res_blocks 3 --num_samples 64 --resblock_updown True \
#     --use_fp16 True --weight_schedule uniform

# imagenet256
## onestep
# mpiexec -n 1 python scripts/image_sample.py --batch_size 32 --training_mode consistency_distillation \
#     --sampler onestep --model_path cm_train_uncond_log/model050000.pt --attention_resolutions 32,16,8 \
#     --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 \
#     --num_head_channels 64 --num_res_blocks 2 --num_samples 64 --resblock_updown True \
#     --use_fp16 True --weight_schedule uniform --ts 0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150 --steps 151

## multistep
# mpiexec -n 1 python scripts/image_sample.py --batch_size 32 --training_mode consistency_distillation \
#     --sampler multistep --ts 0,67,150 --steps 151 --model_path cm_train_uncond_log/ema_0.9999432189950708_050000.pt --attention_resolutions 32,16,8 \
#     --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 \
#     --num_head_channels 64 --num_res_blocks 2 --num_samples 64 --resblock_updown True --use_fp16 True --weight_schedule uniform


# mpiexec -n 1 python scripts/image_sample.py --batch_size 64 --training_mode consistency_distillation \
#     --sampler onestep --model_path /zhangshuhai/ODEPure/consistency_models/checkpoints/imagenet64/cd_imagenet64_lpips.pt --attention_resolutions 32,16,8 \
#     --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 \
#     --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples 64 \
#     --resblock_updown True --use_fp16 True --weight_schedule uniform
