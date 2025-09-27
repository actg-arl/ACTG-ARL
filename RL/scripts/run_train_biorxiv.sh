set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --multi_gpu --config_file=default_config.yaml --main_process_port=29519 \
    --num_machines 1  \
    --num_processes 8 \
    train_rl_ptx.py --log_with=wandb \
    --model_name=$HOME/mount-folder/xcloud/xcloud_dp_finetuning/outputs/gemma-3-1b-pt_biorxiv-complex8et-condgen_dp-eps-4.0_bs-2048_step-1120_lr-0.001-cosine_seed-42_biorxiv-complex8et-condgen_noredacted_modelgoogle-gemma-3-1b-pt_eps4.0_delta3.38e-06_bs2048_maxseq300-512_epoch3_lr0.001_clip1.0_np4.3_gpus8/model_epoch79/ \
    --reward_model_name=llmjudge_biorxiv-complex8et-conditions \
    --dataset_name=$HOME/mount-folder/code/dp_finetuning_0615_rl/generations_biorxiv-complex8et-conditions/generated_biorxiv-complex8et-conditions_AIM_eps-2.98_rho-0.1816_n-1000k_iter-3k.csv \
    --ptx_dataset_path=sft_fullprompt_9_generated_biorxiv-condgen_model-gemma-3-1b-pt_dp-eps-4-np-3.65-4.3-complex8et-aimn-1e-3-cosine-1_temp-1.0_tp-0.95_tk-0_eval_n-50000.csv \
    --adafactor=False \
    --save_freq=20 \
    --batch_size=64 \
    --mini_batch_size=4 \
    --gradient_accumulation_steps=16 \
    --ppo_epochs=4 \
    --seed=42 \
    --max_length=512 \
    --gen_bsize=64 \
    --scale_reward=1 \
    --learning_rate=5e-6 \
    --early_stopping=False \
    --output_dir=output_syn_gemma-3-1b_biorxiv-complex8et-condgen_bfloat16_kl-0.2-adaptive_eps-4_len-512_bs-64_mbs-64_lr-5e-6_seed-42_scale-reward-avg-more-data-ptx-decay-$1-$2 \
    --init_kl_coef=0.2 \
    --steps=2000 \
    --horizon=2000 \
    --adap_kl_ctrl=True \
    --min_length=10 \
    --wandb_project="ppo-syn-biorxiv" \
    --run_name="std-gemma-3-1b-biorxiv-complex8et-condgen_bfloat16_kl-0.2-adaptive_eps-4_len-512_bs-64_mbs-64_lr-5e-6_seed-42_scale-reward-avg-more-data-ptx-decay-$1-$2" \
    --gen_data_dir="gen_syn_all_samples_std_gemma-3-1b_enron-condgen_bfloat16_kl-0.2-adaptive_eps-4_len-512_bs-64_mbs-64_lr-5e-6_seed-42_scale-reward-avg-more-data-ptx-decay-$1-$2" \
    --ptx_coef_initial $1 \
    --ptx_coef_final $2
