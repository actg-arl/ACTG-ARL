set -x
prompt_len=${6:-128}
seq_len=${7:-128}
prompt_str=${8:-biorxiv-complex8et-conditions}
n_gen=${9:-5000}
model_str=${10:-gemma-3-1b-pt}
python generation_biorxiv_gen.py \
    -m /home/user/DPSFT/outputs/$1/model_epoch$2 \
    -l ${seq_len} -d 0 -o ${prompt_str} -ps ${prompt_str} \
    -out generated_${prompt_str}_model-${model_str}_dp-eps-$3-np-$4-lr-$5_seqlen-${prompt_len}-${seq_len}_temp-1.0_tp-0.95_tk-0_eval_n-${n_gen}.csv \
    -n_gen ${n_gen} -bs 512 -tp 0.95
