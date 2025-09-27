# Supervised Fine-Tuning with DPSFT (Baseline SFT, Conditional SFT)

This code repository is adapted from [google-research/dp_instructions/dp_finetuning](https://github.com/google-research/google-research/tree/master/dp_instructions/dp_finetuning).

## Intro

This code implements DP supervised fine-tuning with LoRA. 
This implementation is built on the following packages: [transformers](https://github.com/huggingface/transformers), [datasets](https://github.com/huggingface/datasets), [accelerate](https://github.com/huggingface/accelerate), and [PEFT](https://github.com/huggingface/peft).

Currently, this code supports the following types of experiments:

- **(baseline-SFT)** Fine-tuning model for direct generation

- **(conditional-SFT-features)** Fine-tuning model to generate features

- **(conditional-SFT-text)** Fine-tuning model to generate text conditioned on features

We supply code supporting the following functionalities:

1. Perform supervised fine-tuning
2. Merge and save the PEFT model
3. Perform generation using the saved model 

## Datasets

This repository mainly support the baseline SFT and conditional SFT on three datasets: Enron email, PMC clinical notes, and bioRxiv abstracts. The datasets can be accessed at [../data](../data).

## Setup

```
pip install -r requirements.txt
```

## Example Scripts

### 1. Supervised fine-tuning

#### baseline-SFT

Fine-tune a gemma-3-1b-pt model on bioRxiv abstracts to directly generate the abstracts.

- Without DP:
    ```
    CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_biorxiv_ft_nondp.sh 256 448 1e-3 29500 512 4 2 biorxiv cosine
    ```
    Here, argument #1 is total batch size, #2 is training iterations, #3 is learning rate, #4 is main port, #5 is max sequence length, #6 is per-device batch size, #7 is the number of gpus used, #8 is the dataset name, #9 is the learning rate scheduler. More details can be referred to in the script.

- With DP:
    ```
    CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_biorxiv_ft_dp.sh 2048 1120 1e-3 29500 4.0 2.8362 512 4 2 biorxiv cosine
    ```

#### conditional-SFT-feature

Fine-tune a gemma-3-1b-pt model on bioRxiv features (schemas).

- Without DP:
    ```
    CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_biorxiv-features_ft_nondp.sh 256 672 1e-3 29500 128 128 16 2 biorxiv-complex8et-conditions cosine
    ```

- With DP:
    ```
    CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_biorxiv-features_ft_dp.sh 2048 1120 1e-3 29500 4 9.96 128 128 16 2 biorxiv-complex8et-conditions cosine
    ```

#### conditional-SFT-text

Fine-tune a gemma-3-1b-pt model on bioRxiv abstracts condiitional on their corresponding features (schemas).

- Without DP:
    ```
    CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_biorxiv-condgen_ft_nondp.sh 256 896 1e-3 29500 512 4 2 biorxiv-complex8et-condgen cosine
    ```

- With DP:
    ```
    CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_biorxiv-condgen_ft_dp.sh 2048 1120 1e-3 29500 4 4.3 512 4 2 biorxiv-complex8et-condgen cosine
    ```

### 2. Merge PEFT Models

Loads the peft model saved to ``peft_model_dir`` under ``peftmodel_epoch{ckpt_id}``, merges to the base model, and saves to ``model_epoch{ckpt_id}`` under the same directory.

```
bash scripts/run_merge.sh [peft_model_dir] [ckpt_id]
```

### 3. Generation 

#### Generation via the baseline-SFT model

Loads the fine-tuned and merged model, performs unconditional generation, and saves the generated bioRxiv abstracts to a local file.

```
bash scripts/run_biorxiv_gen_baseline.sh [model_dir] [ckpt_id] [eps] [noise_multiplier] [lr_str]
```

#### Generation via the conditional-SFT-feature model

Loads the fine-tuned and merged model, performs generation, and saves the generated bioRxiv features (schemas) to a local file.

```
bash scripts/run_biorxiv_gen_features.sh [model_dir] [ckpt_id] [eps] [noise_multiplier] [lr_str] [prompt_len] [generation_len] [dataset_name]
```

#### Generation via the conditional-SFT-text model

Loads the fine-tuned and merged model, performs generation conditioned on the generated features, and saves the generated bioRxiv abstracts to a local file.

```
bash scripts/run_biorxiv_condgen_text.sh [features_file] [model_dir] [noise-1] [noise-2] [lr-str] [eps] [n_gen] [ckpt_id]
```
