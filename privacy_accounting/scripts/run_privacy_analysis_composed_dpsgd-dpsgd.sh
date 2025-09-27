set -x
python privacy_analysis_composed_dpsgd-dpsgd.py \
    --N 28846 \
    --eps 1.0 \
    --bs1 2048 \
    --T1 1120 \
    --bs2 2048 \
    --T2 1120 \
    --eps1_targets 0.5 0.75