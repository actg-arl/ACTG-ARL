set -x
python privacy_analysis_composed_aim-dpsgd.py \
    --total_epsilon 4.0 \
    --total_delta 3.38e-6 \
    --dataset_size_s2 28846 \
    --batch_size_s2 2048 \
    --iterations_s2 1120 \
    --sigma_s2 3.3