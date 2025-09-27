set -x
model_name=${3:-google/gemma-3-1b-pt}
python merge_sft_peft.py \
	-m ${model_name} \
	-p /home/user/DPSFT/outputs/$1 \
	-e_id $2 -o /home/user/DPSFT/outputs/$1/model_epoch$2
