set -x
text_column=${4:-generated_text}
python infer_biorxiv_schema.py \
    --input_file /home/user/DPSFT/generations_$1/$3.$2 \
    --output_file /home/user/results/extracted/biorxiv_schema_v2_$3_rerun.csv \
    --text_column ${text_column}
