"""Embeds text from a file using a Hugging Face model Specter2

with an adapter, and saves the embeddings to a NumPy file.
"""

import argparse
import json
import logging
import os
import sys
from adapters import AutoAdapterModel
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# --- Setup Standard Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
  # --- Argument Parsing ---
  parser = argparse.ArgumentParser(
      description="Embed scientific text using Specter2."
  )
  parser.add_argument(
      "--input_path",
      type=str,
      required=True,
      help="Path to the input CSV or JSONL file.",
  )
  parser.add_argument(
      "--output_embedding_path",
      type=str,
      required=True,
      help="Path to save the output .npy file for embeddings.",
  )
  parser.add_argument(
      "--text_column_name",
      type=str,
      default="generated_text",
      help="Name of the column containing the text to embed.",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=16,
      help="Batch size for embedding generation.",
  )
  parser.add_argument(
      "--device", "-d", type=int, default=0, help="GPU device ID to use."
  )
  args = parser.parse_args()

  # --- Device Setup ---
  device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
  logging.info(f"Using device: {device}")

  # 1. Load the Specter2 model with its adapter
  logging.info("Loading Specter2 model...")
  tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
  model = AutoAdapterModel.from_pretrained("allenai/specter2_base")

  # load the adapter
  model.load_adapter(
      "allenai/specter2", source="hf", load_as="specter2", set_active=True
  )

  model.to(device)
  model.eval()

  # 2. Load biorxiv data
  logging.info(f"Reading data from {args.input_path}...")
  try:
    if args.input_path.endswith(".jsonl"):
      df = pd.read_json(args.input_path, lines=True)
    elif args.input_path.endswith(".csv"):
      df = pd.read_csv(args.input_path)
    else:
      logging.error(
          "Unsupported file format. Please provide a CSV or JSONL file."
      )
      sys.exit(1)
    texts = df[args.text_column_name].tolist()
    logging.info(f"Found {len(texts)} texts to embed.")
  except (FileNotFoundError, KeyError, Exception) as e:
    logging.error(f"Error reading input file '{args.input_path}': {e}")
    sys.exit(1)

  # Clean data to ensure all items are strings, replacing NaN/None with ""
  texts = [text if isinstance(text, str) else "" for text in texts]

  # 3. Generate embeddings in batches
  # Code logic referenced from https://huggingface.co/allenai/specter2
  logging.info("Starting embedding generation...")
  all_embeddings = []
  with torch.no_grad():
    for i in tqdm(
        range(0, len(texts), args.batch_size), desc="Embedding Batches"
    ):
      batch_texts = texts[i : i + args.batch_size]

      inputs = tokenizer(
          batch_texts,
          padding=True,
          truncation=True,
          return_tensors="pt",
          return_token_type_ids=False,
          max_length=512,
      ).to(device)

      output = model(**inputs)
      cls_embedding = output.last_hidden_state[:, 0, :]
      all_embeddings.append(cls_embedding.cpu())

  # 4. Concatenate and save the embeddings
  logging.info("Concatenating results...")
  final_embeddings = torch.cat(all_embeddings, dim=0).numpy()

  output_dir = os.path.dirname(args.output_embedding_path)
  os.makedirs(output_dir, exist_ok=True)

  logging.info(
      f"Saving {final_embeddings.shape[0]} embeddings to"
      f" {args.output_embedding_path}..."
  )
  np.save(args.output_embedding_path, final_embeddings)

  logging.info("Done.")


if __name__ == "__main__":
  main()
