set -x
eps=${6:-inf}
n_gen=${7:-5000}
epoch=${8:-79}
features_str=${9:-biorxiv-complex8et-conditions}
prompt_str=${10:-biorxiv-condgen}
model_name=${11:-gemma-3-1b-pt}
python generation_biorxiv-condgen.py \
    -m /home/user/DPSFT/outputs/$2/model_epoch${epoch} \
    -pf /home/user/DPSFT/generations_${features_str}/$1 \
    -pl 300 -sl 512 -d 0 -o ${prompt_str} -ps ${prompt_str} \
    -out generated_biorxiv-condgen_model-${model_name}_dp-eps-${eps}-np-$3-$4-$5_temp-1.0_tp-0.95_tk-0_eval_n-${n_gen}.jsonl \
    -n_gen ${n_gen} -bs 256 -tp 0.95

