"""Utilities for creating and processing datasets for supervised fine-tuning."""

import logging
import os

from datasets import Dataset, load_dataset
import torch

logger = logging.getLogger(__name__)


def clm_tokenize_function(
    examples,
    tokenizer,
    max_instruction_length=128,
    max_answer_length=384,
    ignore_index=-100,
):
  """Tokenize the text, applying separate max_length constraints to instructions and answers."""
  instructions = examples['prompt']
  answers = examples['completion']

  # Tokenize instructions and answers separately with truncation
  tokenized_instructions = tokenizer(
      instructions,
      max_length=max_instruction_length,
      padding=False,
      truncation=True,
  )
  tokenized_answers = tokenizer(
      answers,
      max_length=max_answer_length,
      padding=False,
      truncation=True,
      add_special_tokens=False,
  )

  # Initialize new lists for the final tokenized outputs
  all_input_ids = []
  all_attention_masks = []
  all_labels = []

  # Process each example individually
  for i in range(len(instructions)):
    instruction_ids = tokenized_instructions['input_ids'][i]
    answer_ids = tokenized_answers['input_ids'][i]

    # Concatenate tokenized parts and add EOS token
    input_ids = instruction_ids + answer_ids + [tokenizer.eos_token_id]
    attention_mask = [1] * len(input_ids)
    labels = list(input_ids)  # Start with a copy

    # Create the loss mask for the instruction part
    instruction_len = len(instruction_ids)
    labels[:instruction_len] = [ignore_index] * instruction_len

    all_input_ids.append(input_ids)
    all_attention_masks.append(attention_mask)
    all_labels.append(labels)

  # The final tokenized output is a dictionary
  tokenized_texts = {
      'input_ids': all_input_ids,
      'attention_mask': all_attention_masks,
      'labels': all_labels,
  }

  return tokenized_texts


class DataCollatorForSupervisedDataset(object):
  """Data collator for supervised dataset."""

  IGNORE_INDEX = -100

  def __init__(
      self,
      tokenizer,
      padding='longest',
      return_tensors='pt',
      device='cuda',
      padding_side='right',
      max_length=64,
  ):
    self.tokenizer = tokenizer
    self.padding = padding
    self.return_tensors = return_tensors
    self.device = device
    self.padding_side = padding_side
    self.max_length = max_length

  def __call__(self, instances):
    if self.padding not in ['longest', 'max_length']:
      raise ValueError(f'Padding {self.padding} is not supported.')
    if self.return_tensors != 'pt':
      raise ValueError(
          f'return_tensors {self.return_tensors} is not supported.'
      )

    input_ids, labels = tuple(
        [instance[key] for instance in instances]
        for key in ('input_ids', 'labels')
    )
    if self.return_tensors == 'pt':
      input_ids = [torch.tensor(input_id).long() for input_id in input_ids]
      labels = [torch.tensor(label).long() for label in labels]

    if self.padding_side == 'left':
      # reverse each input_id in input_ids
      input_ids = [torch.flip(input_id, dims=[0]) for input_id in input_ids]
      labels = [torch.flip(label, dims=[0]) for label in labels]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=self.IGNORE_INDEX
    )

    if self.padding_side == 'left':
      # reverse each input_id in input_ids
      input_ids = torch.flip(input_ids, dims=[1])
      labels = torch.flip(labels, dims=[1])

    input_ids = input_ids.to(self.device)
    labels = labels.to(self.device)

    if self.padding == 'max_length':
      pad_tensor = torch.zeros(
          (input_ids.shape[0], self.max_length - input_ids.shape[1]),
          dtype=torch.long,
      ).to(self.device)
      input_ids = torch.cat(
          [input_ids, pad_tensor + self.tokenizer.pad_token_id], dim=1
      )
      labels = torch.cat([labels, pad_tensor + self.IGNORE_INDEX], dim=1)

    index = [instance['index'] for instance in instances]
    return dict(
        index=index,
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
    )


class TokenizedSupervisedInstructDataset(Dataset):
  """Tokenize the concatenated text from dataset of (insturction, answer) pairs."""

  def __init__(
      self,
      dataset_or_name,
      tokenizer,
      ptx_dataset_path='sft_fullprompt_9_generated_biorxiv-condgen_model-gemma-3-1b-pt_dp-eps-1-np-15.18-12.75-complex8et-aimn-1e-3-cosine_temp-1.0_tp-0.95_tk-0_eval_n-50000.csv',
      split='train',
      max_instruction_length=128,
      max_answer_length=384,
      num_proc=4,
      tokenize_type='clm',
  ):

    dataset_name = dataset_or_name
    if dataset_name in ['biorxiv']:
      home_dir = os.environ['HOME']
      data_file = os.path.join(
          home_dir,
          f'mount-folder/code/dp_finetuning_0615_rl/generations_biorxiv-condgen/{ptx_dataset_path}',
      )
      text_dataset = load_dataset(
          'csv',
          data_files={
              'train': data_file,
          },
      )['train']
      logger.info(f'Loaded ptx_dataset from {ptx_dataset_path}')

    self.text_dataset = text_dataset
    self.tokenizer = tokenizer
    self.max_instruction_length = max_instruction_length
    self.max_answer_length = max_answer_length
    self.num_proc = num_proc
    self.tokenize_type = tokenize_type
    self.dataset_name = dataset_name
    self.dataset_split = split
    self.get_tokenized_dataset()

  def get_tokenized_dataset(self):
    processed_text_dataset = self.text_dataset

    tokenize_func = clm_tokenize_function
    self.tokenized_text_dataset = processed_text_dataset.map(
        lambda x: tokenize_func(
            x,
            self.tokenizer,
            max_instruction_length=self.max_instruction_length,
            max_answer_length=self.max_answer_length,
        ),
        batched=True,
        num_proc=self.num_proc,
    )

    if hasattr(processed_text_dataset, 'repeats'):
      new_input_ids = []
      new_attention_mask = []
      new_labels = []

      for i in range(len(self.tokenized_text_dataset)):
        new_input_ids.extend(
            [self.tokenized_text_dataset[i]['input_ids']]
            * processed_text_dataset.repeats[i]
        )
        new_attention_mask.extend(
            [self.tokenized_text_dataset[i]['attention_mask']]
            * processed_text_dataset.repeats[i]
        )
        new_labels.extend(
            [self.tokenized_text_dataset[i]['labels']]
            * processed_text_dataset.repeats[i]
        )

      # build a huggerface dataset with the repeated input_ids, attention_mask,
      # and labels. with the from_dict method.
      self.tokenized_text_dataset = Dataset.from_dict({
          'input_ids': new_input_ids,
          'attention_mask': new_attention_mask,
          'labels': new_labels,
      })
      # shuffle the dataset
      self.tokenized_text_dataset = self.tokenized_text_dataset.shuffle(seed=42)

  def __len__(self):
    return len(self.tokenized_text_dataset)

  def __getitem__(self, idx):
    # Return the tokenized text, attention mask, and labels
    if not isinstance(idx, list):
      idx = [idx]

    subset_dataset = self.tokenized_text_dataset[idx]
    input_ids = subset_dataset['input_ids']
    attention_mask = subset_dataset['attention_mask']
    labels = subset_dataset['labels']
    return {
        'index': idx,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }
