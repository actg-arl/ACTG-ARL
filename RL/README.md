# Anchored RL for Improving Instruction Following in Conditional Generation


This repository contains the implementation of **Anchored RL**, an algorithm designed to improve the instruction-following capabilities of the conditional text generator.

Our approach combines multiple techniques:


1. **RL with rubric rewards**: improving the model's instruction following capabilities

2. **Mixed RL and SFT gradients**: keeping the model's generations anchored to the textual properties of the private data

3. **Best-of-N SFT data**: curating high-quality synthetic data for SFT by increasing test-time compute

This repository is adapted from [PrasannS/rlhf-length-biases](https://github.com/PrasannS/rlhf-length-biases).

## Setup

```
conda env create -f env_rl.yml
conda activate rl
cd rlutils
pip install -e .
cd ..
```

## Repository Structure

1. We inherit from TRL's ``PPOTrainer`` and ``PPOConfig`` to include our SFT objective. The custom classes ``PPOPTXTrainer`` and ``PPOPTXConfig`` are under [rlutils/rlutils](./rlutils/rlutils).

2. The core logic for rubric reward is in [rlutils/rlutils/rl_utils.py](./rlutils/rlutils/rl_utils.py).

## Example Scripts

- Perform Anchored RL training using a linear decaying scheme for $\gamma$ with a starting value of 2.0 and an end value of 0.2:

```
cd scripts
bash run_train_biorxiv.sh 2.0 0.2
```

- Merging the saved PEFT checkpoint for final generation purpose

```
python merge_peft_adapter.py \
    --base_model_name [base_model_name] \
    --adapter_model_name [model_dir]/step_{i} \
    --output_name [model_dir]/step_{i}_merged
```