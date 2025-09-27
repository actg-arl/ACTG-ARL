set -x
n_gen=${6:-5000}
model_str=${7:-gemma-3-1b-pt}
python generation_biorxiv_gen.py \
    -m /home/user/DPSFT/outputs/$1/model_epoch$2 \
    -l 512 -d 0 -o biorxiv_baseline -ps biorxiv \
    -out generated_biorxiv_model-${model_str}_dp-eps-$3-nm-$4_lr-$5_seqlen-512_n-${n_gen}.csv \
    -n_gen ${n_gen} -bs 512 -tp 0.95