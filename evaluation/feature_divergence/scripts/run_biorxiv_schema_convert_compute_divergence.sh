set -x
version=${2:-_}
python parse_biorxiv_json_schema_to_df.py \
    -i /home/user/DPSFT/extracted/biorxiv_schema${version}$1 \
    -o /home/user/DPSFT/extracted/biorxiv_parsed_schema_rerun${version}$1
python eval_biorxiv_schema_div.py \
    -i /home/user/DPSFT/extracted/biorxiv_parsed_schema_rerun${version}$1 \
    -r /home/user/data/biorxiv/biorxiv_schema_v2_valid_test_gemini-2.5-flash_parsed.csv \
    -o /home/user/DPSFT/extracted/biorxiv_divergence_v2_rerun${version}$1

