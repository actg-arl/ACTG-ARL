"""Utility functions for loading and formatting datasets for PPO training."""

import os

from datasets import Dataset, load_dataset


def build_moviejson_promptdata(tokenizer, num_samples=500_000, seed=42):
  """Creates a dataset of prompts for generating movie JSON objects."""
  tokenizer.pad_token = tokenizer.eos_token

  prompt = (
      "Generate a Json object about a movie. Generate only the json object, do"
      " not include any other text."
  )

  encoded = tokenizer(prompt, return_tensors="pt")

  input_ids = encoded["input_ids"][0]
  attention_mask = encoded["attention_mask"][0]

  # Replicate the same tokenized input
  input_ids_list = [input_ids.clone() for _ in range(num_samples)]
  attention_mask_list = [attention_mask.clone() for _ in range(num_samples)]
  queries = [prompt] * num_samples

  # Wrap into HuggingFace Dataset
  dataset = Dataset.from_dict({
      "input_ids": input_ids_list,
      "attention_mask": attention_mask_list,
      "query": queries,
  })

  dataset.set_format(type="torch")

  return dataset


def build_enron_condgen_freeform_promptdata(
    tokenizer, dataset_name, max_input_len=256, seed=42
):
  """Creates a dataset of prompts for generating Enron emails."""
  ds = load_dataset("csv", data_files=dataset_name)["train"]

  instruction = (
      "Generate a full email that matches the following summary of key"
      " characteristics."
  )
  prompt = "[INST]\n{instruction}\n\n{{summary_text}}\n[/INST]\n".format(
      instruction=instruction
  )

  def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(
        prompt.format(summary_text=sample["generated_text"])
    )[:max_input_len]
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample

  ds = ds.map(
      tokenize, batched=False, num_proc=16, remove_columns=["generated_text"]
  )
  ds.set_format(type="torch")

  return ds


def build_biorxiv_complex8et_condgen_promptdata(
    tokenizer, dataset_name, max_input_len=256, seed=42
):
  """Creates a dataset of prompts for generating bioRxiv abstracts."""
  ds = load_dataset("csv", data_files=dataset_name)["train"]

  instruction = (
      "Please generate a synthetic scientific abstract that matches the below"
      " JSON summary, in the style of a bioRxiv paper."
  )
  prompt = "<start_of_turn>user\n{instruction}\n\n{{summary_text}}\n<end_of_turn>\n<start_of_turn>model\n".format(
      instruction=instruction
  )

  def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(
        prompt.format(summary_text=sample["generated_text"])
    )[:max_input_len]
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample

  ds = ds.map(
      tokenize, batched=False, num_proc=16, remove_columns=["generated_text"]
  )
  ds.set_format(type="torch")

  return ds


# --- Simple Formatting Helpers ---


def qaform(q, r):
  """Formats text in a standard Question/Answer style."""
  return "Question: " + q + "\n\nAnswer: " + r


def anscat(q, r):
  """Concatenates the query and response."""
  return q + r


def ansonly(q, r):
  """Returns only the response."""
  return r


# --- Dataloader Collator ---


def collator(data):
  return dict((key, [d[key] for d in data]) for key in data[0])


# --- LLM-as-a-Judge Prompt Formatters ---


def llmjudge(q, r):
  """Formats a query using a loaded template for Enron scoring."""
  home_dir = os.environ["HOME"]

  prompt_template = open(
      os.path.join(
          home_dir,
          "mount-folder/code/prompt-feature-extraction_0616_rl/llm_judge_template.txt",
      )
  ).read()
  prefix = (
      "<bos>[INST]\nGenerate a full email that matches the following summary of"
      " key characteristics.\n\n"
  )
  suffix = "\n[/INST]\n"
  q = q.removeprefix(prefix).removesuffix(suffix)
  return prompt_template.format(
      condition_string=q,  # strip the original context from the instruction prompt
      email_string=r,
  )


def llmjudge_biorxiv(q, r):
  """Formats a query using a loaded template for bioRxiv scoring."""
  home_dir = os.environ["HOME"]

  prompt_template = open(
      os.path.join(
          home_dir,
          "mount-folder/code/prompt-feature-extraction_0616_rl/biorxiv_schema_extraction_prompt_v2.txt",
      )
  ).read()
  return prompt_template.format(abstract_text=r)
