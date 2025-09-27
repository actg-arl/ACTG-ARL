"""Merges a trained PEFT LoRA adapter into its base model and saves the

resulting full model and tokenizer to a specified directory.
"""

# --- Standard Library Imports ---
import argparse
import logging
import os
import sys

# --- Third-Party Library Imports ---
import peft
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Setup Standard Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
  """Main function to run the model merging process."""
  # --- Argument Parsing ---
  # Improved argument names and help text for clarity.
  parser = argparse.ArgumentParser(
      description="Merge a PEFT LoRA adapter into a base model."
  )
  parser.add_argument(
      "--base_model_path",
      "-m",
      type=str,
      required=True,
      help="Path to the base model (e.g., 'google/gemma-2-9b').",
  )
  parser.add_argument(
      "--lora_dir",
      "-p",
      type=str,
      required=True,
      help=(
          "Path to the directory containing the LoRA adapter weights (e.g.,"
          " 'outputs/run1/checkpoint-100')."
      ),
  )
  parser.add_argument(
    "--epoch_id",
    "-e_id",
    type=int,
    default=-1,
    help=(
        "Epoch ID of the LoRA adapter to merge. If -1, uses the latest epoch."
    ),
  )
  parser.add_argument(
      "--output_path",
      "-o",
      type=str,
      required=True,
      help="Path to save the merged model and tokenizer.",
  )
  parser.add_argument(
      "--special_tokens",
      type=str,
      choices=["none", "inst", "inst-sep"],
      default="none",
      help="Add special tokens to the tokenizer ('inst' or 'inst-sep').",
  )
  args = parser.parse_args()

  # --- Initial Setup ---
  logging.info(f"Base model: {args.base_model_path}")
  lora_path = os.path.join(args.lora_dir, f'peftmodel_epoch{args.epoch_id}')
  logging.info(f"LoRA adapter path: {lora_path}")
  logging.info(f"Output path: {args.output_path}")

  if not os.path.exists(lora_path):
    logging.error(f"LoRA path not found: '{lora_path}'")
    sys.exit(1)

  # --- Model and Tokenizer Loading ---
  try:
    logging.info("Loading base model and tokenizer...")
    compute_dtype = (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        device_map="auto",  # Distribute model across available devices
    )

    # Handle padding token for models that don't have one set.
    if tokenizer.pad_token_id is None:
      logging.warning(
          "pad_token_id not set. Using unk_token_id as pad_token_id."
      )
      tokenizer.pad_token_id = tokenizer.unk_token_id

  except Exception as e:
    logging.error(f"Failed to load base model or tokenizer: {e}")
    sys.exit(1)

  # --- Special Token Handling ---
  if args.special_tokens != "none":
    tokens_to_add = ["[INST]", "[/INST]"]
    if args.special_tokens == "inst-sep":
      tokens_to_add.append("[SEP]")

    tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
    model.resize_token_embeddings(len(tokenizer))
    logging.info(f"Added special tokens to tokenizer: {tokens_to_add}")

  # --- Load and Merge LoRA Adapter ---
  try:
    logging.info("Loading and merging LoRA adapter...")
    # Load the PEFT model with the adapter
    model = peft.PeftModel.from_pretrained(model, lora_path)
    # Merge the adapter layers into the base model
    model = model.merge_and_unload()
    logging.info("Successfully merged LoRA adapter.")
  except Exception as e:
    logging.error(
        f"Failed to load or merge the LoRA adapter from '{lora_path}': {e}"
    )
    sys.exit(1)

  # --- Save Final Model ---
  try:
    logging.info(f"Saving merged model and tokenizer to {args.output_path}...")
    # Ensure the output directory exists.
    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    logging.info("Save complete.")
  except Exception as e:
    logging.error(f"Failed to save the final model: {e}")
    sys.exit(1)

  logging.info("Script finished successfully.")


if __name__ == "__main__":
  main()
