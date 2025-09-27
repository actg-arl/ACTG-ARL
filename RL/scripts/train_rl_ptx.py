"""Main script for RL training with a Pre-training Mix (PTX) objective."""

import logging
import os
import sys

from accelerate import Accelerator
from rlutils.data import (
    ansonly,
    build_biorxiv_complex8et_condgen_promptdata,
    build_enron_condgen_freeform_promptdata,
    build_moviejson_promptdata,
    collator,
    llmjudge,
    llmjudge_biorxiv,
    qaform,
)
from rlutils.ppo_ptx_trainer import PPOPTXTrainer
from rlutils.rl_utils import (
    ScriptArguments,
    load_models,
    train_loop,
)
from rlutils.data_utils import (
    TokenizedSupervisedInstructDataset,
)
from transformers import HfArgumentParser
from trl import set_seed
import wandb

# Configure logger to output to console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training_run.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def main(script_args: ScriptArguments):
  """Sets up and runs the PPO+PTX training loop.

  Args:
      script_args: Configuration object with all script parameters.
  """
  # --- 1. Setup and Initialization ---
  accelerator = Accelerator()
  set_seed(script_args.seed)

  # Ensure output directory exists and has a trailing slash
  os.makedirs(script_args.output_dir, exist_ok=True)
  if not script_args.output_dir.endswith("/"):
    script_args.output_dir += "/"

  # Initialize wandb on the main process
  if accelerator.is_main_process:
    os.environ["WANDB_TAGS"] = '["llamatrl"]'
    wandb.init(
        project=script_args.wandb_project,
        name=script_args.run_name,
        config=vars(script_args),
    )
    logger.info("Wandb initialized.")

  # --- 2. Load Models and Tokenizer ---
  config, tokenizer, model, optimizer = load_models(script_args)
  logger.info("Base models and tokenizer loaded successfully.")
  reward_model = None
  reward_tokenizer = None

  # --- 3. Load Datasets and Formatting ---
  logger.info(f"Building dataset for: {script_args.dataset_name}")
  dataset_name = script_args.dataset_name
  rmformat = qaform  # Default reward model format

  if "movie-json" in script_args.reward_model_name:
    dataset = build_moviejson_promptdata(tokenizer, seed=script_args.seed)
    reward_model = "movie-json"
    rmformat = ansonly
  elif "enron-freeform-conditions" in dataset_name:
    dataset = build_enron_condgen_freeform_promptdata(
        tokenizer, dataset_name, seed=script_args.seed
    )
    reward_model = "llmjudge_enron-freeform-conditions"
    rmformat = llmjudge
  elif "biorxiv-complex8et-conditions" in dataset_name:
    dataset = build_biorxiv_complex8et_condgen_promptdata(
        tokenizer, dataset_name, seed=script_args.seed
    )
    reward_model = "llmjudge_biorxiv-complex8et-conditions"
    rmformat = llmjudge_biorxiv
  else:
    raise ValueError(f"Unknown or unhandled dataset name: {dataset_name}")

  logger.info(f"Dataset loaded. Example: {dataset[0]}")

  # Load the separate pre-training dataset
  if "biorxiv" in script_args.ptx_dataset_path:
    pretrain_dataset = TokenizedSupervisedInstructDataset(
        "biorxiv",
        tokenizer,
        ptx_dataset_path=script_args.ptx_dataset_path,
        split="train",
        max_instruction_length=512,
        max_answer_length=512,
        num_proc=4,
    )
    logger.info("Pre-training dataset for bioRxiv loaded.")
  else:
    raise ValueError("Unsupported pre-training dataset.")

  # --- 4. Initialize Trainer ---
  ppo_trainer = PPOPTXTrainer(
      config=config,
      model=model,
      ref_model=None,
      tokenizer=tokenizer,
      dataset=dataset,
      pretrain_dataset=pretrain_dataset,
      data_collator=collator,
      optimizer=optimizer,
  )

  trainable_params = [
      n for n, p in ppo_trainer.model.named_parameters() if p.requires_grad
  ]
  logger.info(f"Number of trainable parameters: {len(trainable_params)}")

  # --- 5. Start Training ---
  logger.info("Starting training loop...")
  train_loop(
      script_args=script_args,
      ppo_trainer=ppo_trainer,
      reward_model=reward_model,
      tokenizer=tokenizer,
      qaform=rmformat,
      min_length=script_args.min_length,
      reward_tokenizer=reward_tokenizer,
  )
  logger.info("Training finished.")


if __name__ == "__main__":
  parser = HfArgumentParser(ScriptArguments)
  script_args = parser.parse_args_into_dataclasses()[0]
  main(script_args)
