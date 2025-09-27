set -x
mode=${3:-biorxiv-condgen}
ext=${4:-jsonl}
prefix=${5:-''}
text_column=${6:-generated_text}
python embed_biorxiv.py \
    --input_path /home/user/DPSFT/generations_${mode}/${prefix}generated_$1.${ext} \
    --output_embedding_path embeddings/${prefix}$1__specter2_len-512_embeddings_rerun.npy \
    --text_column_name ${text_column} \
    --batch_size 128 \
    --device $2