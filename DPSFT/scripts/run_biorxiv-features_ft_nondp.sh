set -x
prompt_len=${5:-64}
seq_len=${6:-512}
device_bs=${7:-4}
gpus=${8:-2}
dataset_name=${9:-biorxiv-conditions}
lr_scheduler=${10:-constant}
model_name=${11:-google/gemma-3-1b-pt}
model_str=${12:-gemma-3-1b}
python generate_train_command.py \
  --dataset_name ${dataset_name} \
  --model_name ${model_name} \
  --job_sess ${model_str}_${dataset_name}_nondp_bs-$1_step-$2_lr-$3-${lr_scheduler}_seed-42 \
  --eps 1000000000 \
  --delta 0.1 \
  --clip 1000000000 \
  --perdevice_bs ${device_bs} \
  --gpus ${gpus} \
  --max_instruction_len ${prompt_len} \
  --max_answer_len ${seq_len} \
  --total_bs $1 \
  --num_steps $2 \
  --lr $3 \
  --lr_scheduler ${lr_scheduler} \
  --prompt_style ${dataset_name}_generation \
  --main_process_port $4 \
  --seed 42

