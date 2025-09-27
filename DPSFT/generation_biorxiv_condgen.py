"""Generates text samples from a Hugging Face model given a conditional prompt

formatted by a template and saves the results to a JSONL file.
"""

# --- Standard Library Imports ---
import argparse
import json
import logging
import os
import os.path as osp
import random
import time

# --- Third-Party Library Imports ---
import pandas as pd
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
      '--prompt_len',
      '-pl',
      type=int,
      default=128,
      help='max sequence length for generation',
  )
  parser.add_argument(
      '--seq_len',
      '-sl',
      type=int,
      default=512,
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
  parser.add_argument('--prompt_file', '-pf', type=str, required=True)
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
  parser.add_argument(
      '--begin_idx',
      '-b',
      type=int,
      default=0,
      help='begin index for input data',
  )
  parser.add_argument('--prompt_str', '-ps', type=str, default='biorxiv')
  args = parser.parse_args()

  # --- Device Setup and Model Loading ---
  device = f'cuda:{args.device}'

  compute_dtype = (
      torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
  )

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

  out_folder = (
      f'generations_{args.output_dir}' if args.output_dir else 'generations'
  )
  os.makedirs(out_folder, exist_ok=True)

  # --- Prompt Preparation ---
  prompt_dict = get_prompt_dict(f'{args.prompt_str}_generation')
  prompt = prompt_dict['prompt']
  logging.info('-----------PROMPT')
  logging.info(prompt)
  logging.info('-----END PROMPT-----------')

  # --- Input Data Preparation ---
  input_data = pd.read_csv(args.prompt_file)['generated_text'].tolist()

  n_gen = args.n_gen

  if len(input_data) > n_gen:
    random.seed(42)
    input_texts = random.sample(input_data, k=n_gen)
  else:
    input_texts = input_data

  input_texts = sorted(input_texts, key=lambda x: len(tokenizer.encode(x)))
  logging.info('rearranged input_texts by increasing tokenized length')

  # --- Output File Setup ---
  bs = args.bs
  jsonl_file = os.path.join(out_folder, args.out_file)

  if args.begin_idx > 0:
    n_gen = len(input_texts) - args.begin_idx
    input_texts = input_texts[args.begin_idx :]
    assert osp.exists(
        jsonl_file
    ), f'File {jsonl_file} does not exist. Please check the path.'
    write_mode = 'a'
  else:
    write_mode = 'w'

  # --- Generation and Saving ---
  t_start = time.time()
  with open(jsonl_file, write_mode) as f:

    logging.info(f'Generating {n_gen} sequences and writing to {jsonl_file}...')

    for i in tqdm(range(0, n_gen, bs)):
      bs_cur = min(bs, n_gen - i)

      cur_input_texts = input_texts[i : i + bs_cur]
      cur_prompts = [prompt.format(summary=text) for text in cur_input_texts]
      batch = tokenizer(
          cur_prompts,
          return_tensors='pt',
          padding=True,
          truncation=True,
          max_length=args.prompt_len,
      )
      batch = {k: v.to(device) for k, v in batch.items()}

      with torch.no_grad():
        output = model.generate(
            **batch,
            max_new_tokens=args.seq_len,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
        )
      output = output[:, batch['input_ids'].shape[1] :]

      output_texts = tokenizer.batch_decode(output, skip_special_tokens=True)

      for input_text, sample in zip(cur_input_texts, output_texts):
        f.write(
            json.dumps(
                {'input_text': input_text, 'generated_text': sample},
                ensure_ascii=False,
            )
            + '\n'
        )

  logging.info('Generation and writing complete.')
  logging.info('Total time: %s', time.time() - t_start)


if __name__ == '__main__':
  main()
