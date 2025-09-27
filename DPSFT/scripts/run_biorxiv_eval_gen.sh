set -x
seqlen=${5:-1024}
device_bs=${6:-16}
gpus=${7:-2}
lr_scheduler=${9:-constant}
model_name=${10:-gpt2}
model_str=${11:-gpt2}
data_str=${12:-biorxiv-generated}
python generate_train_command.py \
  --dataset_name biorxiv-generated \
  --dataset_path $8 \
  --model_name ${model_name} \
  --job_sess ${model_str}_biorxiv-gen-${data_str}_nondp_bs-$1_step-$2_lr-$3-${lr_scheduler}_seed-42 \
  --eps 1000000000 \
  --delta 0.1 \
  --clip 1000000000 \
  --perdevice_bs ${device_bs} \
  --gpus ${gpus} \
  --max_instruction_len 16 \
  --max_seq_len ${seqlen} \
  --total_bs $1 \
  --num_steps $2 \
  --lr $3 \
  --lr_scheduler ${lr_scheduler} \
  --next_token_prediction_acc \
  --prompt_style biorxiv_evaluation \
  --main_process_port $4 \
  --seed 42 \
  # --eval_only
