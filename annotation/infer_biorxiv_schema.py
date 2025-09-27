"""Processes a dataset of abstracts to infer a schema using the Gemini API.

This script reads abstracts from an input file, sends them in parallel to
the Gemini API with a specified prompt, and saves the inferred schema
back to a new CSV file.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
import os
import random
import sys
import time
from google import genai
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
API_KEY = os.getenv('GOOGLE_API_KEY')
MODEL_NAME = 'gemini-2.5-flash-lite-preview-06-17'
MAX_WORKERS = 8
MAX_RETRIES = 5

# --- Setup Basic Logging ---
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# --- Silence Noisy Library Loggers ---
logging.getLogger('google_genai.models').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# --- Argument Parser Setup ---
parser = argparse.ArgumentParser(
    description='Infer schema for Biorxiv abstracts using the Gemini API.'
)
parser.add_argument(
    '--input_file',
    type=str,
    required=True,
    help='Path to the input CSV file containing abstracts.',
)
parser.add_argument(
    '--output_file',
    type=str,
    required=True,
    help='Path to save the output CSV file with inferred schema.',
)
parser.add_argument(
    '--prompt_file',
    type=str,
    default='./prompts/biorxiv_schema_extraction_prompt.txt',
    help='Path to the schema inference prompt file.',
)
parser.add_argument(
    '--text_column',
    type=str,
    default='abstract',
    help='Column name for the abstract content.',
)
parser.add_argument(
    '--sample_size',
    type=int,
    default=0,
    help=(
        'Number of rows to sample from the input file. Set to 0 to use the full'
        ' dataset.'
    ),
)
parser.add_argument('--output_column', type=str, default='schema')
args = parser.parse_args()


def infer_schema(
    abstract_text: str, client: genai.Client, prompt_template: str
) -> str:
  """Sends a single abstract to the Gemini API and returns the inferred schema.

  Includes error handling and retries.
  """
  for attempt in range(MAX_RETRIES):
    try:
      full_prompt = prompt_template.format(abstract_text=abstract_text)
      response = client.models.generate_content(
          model=MODEL_NAME,
          contents=full_prompt,
      )
      return response.text.strip()
    except Exception as e:
      wait_time = (2**attempt) + random.uniform(0, 1)
      logging.warning(
          f'API call failed: {e}. Retrying in {wait_time:.2f} seconds...'
          f' (Attempt {attempt + 1}/{MAX_RETRIES})'
      )
      time.sleep(wait_time)
  logging.error('Request failed after all retries.')
  return 'GENERATION_ERROR'

def main():
  # --- Read Input Data ---
  logging.info(f'Reading data from: {args.input_file}')
  if args.input_file.endswith('.csv'):
    df = pd.read_csv(args.input_file)
  elif args.input_file.endswith('.jsonl'):
    df = pd.read_json(args.input_file, lines=True)
  else:
    logging.error("Unsupported file format. Please provide a CSV or JSONL file.")
    sys.exit(1)

  if args.sample_size > 0:
    logging.info(f'Randomly sampling {args.sample_size} rows from the dataset.')
    df = df.sample(n=args.sample_size, random_state=42)

  # --- Initialize the Generative AI Client ---
  if not API_KEY:
    logging.error(
        'GOOGLE_API_KEY environment variable not set. Please set it before'
        ' running the script.'
    )
    sys.exit(1)

  client = genai.Client(api_key=API_KEY)

  # --- Define the prompt and the processing function ---
  try:
    with open(args.prompt_file, 'r') as f:
      prompt_template = f.read()
  except FileNotFoundError:
    logging.error(f'Prompt template not found at {args.prompt_file}')
    sys.exit(1)


  # --- Prepare the list of abstracts to process ---
  abstract_list = df[args.text_column].fillna('').tolist()
  infer_func = partial(
      infer_schema, client=client, prompt_template=prompt_template
  )

  # --- Run Inference ---
  start_time = time.time()

  logging.info(
      f'Starting schema inference for {len(df)} abstracts using up to'
      f' {MAX_WORKERS} parallel workers...'
  )

  with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    inferred_schema = list(
        tqdm(
            executor.map(infer_func, abstract_list),
            total=len(abstract_list),
        )
    )

  # --- Save Results ---
  df[f'inferred_{args.output_column}'] = inferred_schema

  logging.info(f'Saving results to: {args.output_file}')
  df.to_csv(args.output_file, index=False, encoding='utf-8')

  end_time = time.time()
  logging.info(f'Processing completed in {end_time - start_time:.2f} seconds.')
  logging.info('Done.')


if __name__ == '__main__':
  main()
