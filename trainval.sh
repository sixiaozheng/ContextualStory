# story visualization

# pororo dataset
CONFIG=configs/visualization/training_contextualstory_sv_pororo.yaml
NAME=trainval_40k

accelerate launch --mixed_precision="bf16" --multi_gpu --main_process_port=29502 train.py \
    --config=$CONFIG -n=$NAME --wandb

NUM_GPU=1
accelerate launch --num_processes=$NUM_GPU --main_process_port=29502 eval.py \
    --config=$CONFIG -n=$NAME


# flintstones dataset
CONFIG=configs/visualization/training_contextualstory_sv_flintstones.yaml
NAME=trainval_80k

accelerate launch --mixed_precision="bf16" --multi_gpu --main_process_port=29502 train.py \
    --config=$CONFIG -n=$NAME --wandb

NUM_GPU=1
accelerate launch --num_processes=$NUM_GPU --main_process_port=29502 eval.py \
    --config=$CONFIG -n=$NAME


# story continuation
# pororo dataset
CONFIG=configs/continuation/training_contextualstory_sc_pororo.yaml
NAME=trainval_40k
accelerate launch --mixed_precision="bf16" --multi_gpu --main_process_port=29502 train.py \
    --config=$CONFIG -n=$NAME --wandb

NUM_GPU=1
accelerate launch --num_processes=$NUM_GPU --main_process_port=29502 eval.py \
    --config=$CONFIG -n=$NAME


# flintstones dataset
CONFIG=configs/continuation/training_contextualstory_sc_flintstones.yaml
NAME=trainval_80k
accelerate launch --mixed_precision="bf16" --multi_gpu --main_process_port=29502 train.py \
    --config=$CONFIG -n=$NAME --wandb

NUM_GPU=1
accelerate launch --num_processes=$NUM_GPU --main_process_port=29502 eval.py \
    --config=$CONFIG -n=$NAME