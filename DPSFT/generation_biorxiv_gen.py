"""Generates text samples from a Hugging Face model given a fixed prompt

and saves the results to a CSV file.
"""

# --- Standard Library Imports ---
import argparse
import csv
import logging
import os
import time

# --- Third-Party Library Imports ---
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
# --- Application-Specific Imports ---
from utils.data_utils import get_prompt_dict

# --- Setup Standard Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


def main():
  """Main function to run the text generation and saving process."""
  # --- Argument Parsing ---
  parser = argparse.ArgumentParser()
  parser.add_argument('--device', '-d', type=int, default=0)
  parser.add_argument('--model_name_or_path', '-m', type=str, required=True)
  parser.add_argument('--output_dir', '-o', type=str, default='')
  parser.add_argument(
      '--out_file', '-out', type=str, default='output.json', required=True
  )
  parser.add_argument(
      '--seq_len',
      '-l',
      type=int,
      default=128,
      help='max sequence length for generation',
  )
  parser.add_argument(
      '--bs', '-bs', type=int, default=64, help='batch size for generation'
  )
  parser.add_argument(
      '--n_gen',
      '-n_gen',
      type=int,
      default=64,
      help='number of generations per input',
  )
  parser.add_argument(
      '--top_p', '-tp', type=float, default=1.0, help='top_p for sampling'
  )
  parser.add_argument(
      '--top_k', '-tk', type=int, default=0, help='top_k for sampling'
  )
  parser.add_argument(
      '--temperature',
      '-temp',
      type=float,
      default=1.0,
      help='temperature for sampling',
  )
  parser.add_argument('--prompt_str', '-ps', type=str, default='biorxiv')
  args = parser.parse_args()

  # --- Device Setup and Model Loading ---
  device = f'cuda:{args.device}'

  compute_dtype = (
      torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
  )

  logging.info("Loading model...")

  model = AutoModelForCausalLM.from_pretrained(
      args.model_name_or_path,
      torch_dtype=compute_dtype,
      low_cpu_mem_usage=True,
      attn_implementation='eager',
  )
  model.eval()
  model.to(device)

  tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
  tokenizer.padding_side = 'left'

  # --- Prompt Preparation ---
  prompt_dict = get_prompt_dict(f'{args.prompt_str}_generation')
  text = prompt_dict['prompt']
  logging.info('----PROMPT----')
  logging.info(text)
  logging.info('---END PROMPT---')

  input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

  # --- Output Directory Setup ---
  out_folder = (
      f'generations_{args.output_dir}' if args.output_dir else 'generations'
  )
  os.makedirs(out_folder, exist_ok=True)

  n_gen = args.n_gen
  bs = args.bs
  csv_file = os.path.join(out_folder, args.out_file)

  # --- Generation and Saving to CSV ---
  t_start = time.time()
  with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)

    writer.writerow(['generated_text'])

    logging.info(f'Generating {n_gen} sequences and writing to {csv_file}...')

    for i in tqdm(range(0, n_gen, bs)):
      bs_cur = min(bs, n_gen - i)

      output = model.generate(
          input_ids,
          num_return_sequences=bs_cur,
          max_new_tokens=args.seq_len,
          do_sample=True,
          top_p=args.top_p,
          top_k=args.top_k,
          temperature=args.temperature,
      )
      output = output[:, input_ids.shape[1] :]

      output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

      for sample in output_text:
        writer.writerow([sample])

  logging.info('Generation and CSV writing complete.')
  logging.info('Total time: %s', time.time() - t_start)


if __name__ == '__main__':
  main()
