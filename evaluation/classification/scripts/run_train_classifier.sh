set -x
python train_biorxiv_domain_scibert.py \
    --train_df_path $1 \
    --test_df_path biorxiv_domain_test_gemini-2.5-flash.csv