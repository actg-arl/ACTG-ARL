"""A script to merge a PEFT adapter into a base transformer model and save the resulting model."""

from dataclasses import dataclass, field
import logging
import time

from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

# Set up a basic logger to show informative messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


@dataclass
class ScriptArguments:
  """Defines the command-line arguments for merging a PEFT model."""

  adapter_model_name: str = field(
      metadata={"help": "Path or Hub name of the PEFT adapter to merge."}
  )
  base_model_name: str = field(
      metadata={"help": "Path or Hub name of the base model."}
  )
  output_name: str = field(
      metadata={"help": "Local path to save the merged model and tokenizer."}
  )


def main():
  """Runs the main logic of the script."""
  # 1. Parse command-line arguments
  parser = HfArgumentParser(ScriptArguments)
  script_args = parser.parse_args_into_dataclasses()[0]
  logging.info(f"Starting model merge with arguments: {script_args}")

  # 2. Load the base model and tokenizer
  logging.info(f"Loading base model from '{script_args.base_model_name}'...")
  base_model = AutoModelForCausalLM.from_pretrained(
      script_args.base_model_name,
      return_dict=True,
      torch_dtype=torch.bfloat16,
      device_map="auto",  # Automatically place model on available devices
  )
  tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name)
  logging.info("Base model and tokenizer loaded successfully.")

  # 3. Load the PEFT adapter and merge it into the base model
  logging.info(
      f"Loading PEFT adapter from '{script_args.adapter_model_name}'..."
  )
  # The PeftModel class takes the base model as the first argument
  model = PeftModel.from_pretrained(base_model, script_args.adapter_model_name)
  model.eval()

  logging.info("Merging adapter weights into the base model...")
  model = model.merge_and_unload()
  logging.info("Model merged successfully.")

  # 4. Save the final merged model and tokenizer
  start_time = time.time()
  logging.info(
      f"Saving merged model and tokenizer to '{script_args.output_name}'..."
  )
  model.save_pretrained(script_args.output_name)
  tokenizer.save_pretrained(script_args.output_name)
  duration = time.time() - start_time
  logging.info(f"Model saved in {duration:.2f} seconds.")


if __name__ == "__main__":
  main()
