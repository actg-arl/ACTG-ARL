# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data utils."""

import json
import logging
import os
import os.path as osp

from datasets import Dataset, DatasetDict
from datasets import load_dataset, load_from_disk
import torch


def clm_tokenize_function(
    examples,
    tokenizer,
    max_instruction_length=128,
    max_answer_length=384,
    ignore_index=-100,
):
  """Tokenize the text, applying separate max_length constraints to instructions and answers."""
  instructions = examples['instruction']
  answers = examples['answer']

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


# pylint: disable=unused-argument
def eval_tokenize_function(
    examples, tokenizer, max_length=64, truncation=True, ignore_index=-100
):
  """Tokenize the eval data."""
  # examples are from huggingface datasets that contains
  # (instruction, answer) pairs.
  # during training, the loss of instruction tokens will be ignored
  instructions = examples['instruction']
  answers = examples['answer']

  # input_ids contain instruction only
  texts = instructions
  tokenized_texts = tokenizer(
      texts, max_length=max_length - 1, padding=False, truncation=truncation
  )
  tokenized_texts['labels'] = tokenizer(
      answers, max_length=max_length - 1, padding=False, truncation=truncation
  )['input_ids']

  return tokenized_texts


def _preprocess_text_dataset_fixed_prompt_instructions(
    hf_dataset, prompt_dict, split, column='text'
):
  if split == 'train':
    hf_dataset = hf_dataset['train']
  elif split in ['validation', 'valid', 'val']:
    hf_dataset = hf_dataset['validation']
  else:
    hf_dataset = hf_dataset['test']

  answers = hf_dataset[column]
  logging.info('--- prompt_dict begins ---')
  logging.info(prompt_dict['prompt'])
  logging.info('--- prompt_dict ends ---')

  formatted_instructions = [prompt_dict['prompt'] for _ in range(len(answers))]

  text_dataset = Dataset.from_dict(
      {'instruction': formatted_instructions, 'answer': answers}
  )

  return text_dataset


def _preprocess_text_dataset_conditional_prompt_instructions(
    hf_dataset, prompt_dict, split, feature_name, text_name
):
  if split == 'train':
    hf_dataset = hf_dataset['train']
  else:
    hf_dataset = hf_dataset['validation']

  features = hf_dataset[feature_name]
  answers = hf_dataset[text_name]

  logging.info('--- prompt_dict begins ---')
  logging.info(prompt_dict['prompt'])
  logging.info('--- prompt_dict ends ---')

  formatted_instructions = [
      prompt_dict['prompt'].format(feature=feature) for feature in features
  ]

  text_dataset = Dataset.from_dict(
      {'instruction': formatted_instructions, 'answer': answers}
  )
  return text_dataset


def _preprocess_text_dataset_enron_freeform_e2e_instructions(
    hf_dataset, prompt_dict, split
):
  if split == 'train':
    hf_dataset = hf_dataset['train']
  else:
    hf_dataset = hf_dataset['validation']

  summaries = hf_dataset['extracted_features']
  contents = hf_dataset['content']

  logging.info('--- prompt_dict begins ---')
  logging.info(prompt_dict['prompt'])
  logging.info('--- prompt_dict ends ---')

  formatted_instructions = [
      prompt_dict['prompt'] for _ in range(len(summaries))
  ]

  answers = []
  for summary, content in zip(summaries, contents):
    answers.append(
        '\n##### SUMMARY #####\n{summary}\n\n##### EMAIL #####\n{content}'
        .format(summary=summary, content=content)
    )

  text_dataset = Dataset.from_dict(
      {'instruction': formatted_instructions, 'answer': answers}
  )

  return text_dataset


def _preprocess_text_dataset_biorxiv_nl3_condgen_instructions(
    hf_dataset, prompt_dict, split, column='selected_schema'
):
  if split == 'train':
    hf_dataset = hf_dataset['train']
  else:
    hf_dataset = hf_dataset['validation']

  summaries = hf_dataset[column]
  answers = hf_dataset['abstract']

  logging.info('--- prompt_dict begins ---')
  logging.info(prompt_dict['prompt'])
  logging.info('--- prompt_dict ends ---')

  formatted_instructions = []
  for summary in summaries:
    json_data = json.loads(summary)
    prompt = prompt_dict['prompt'].format(
        token_counts=json_data['token_counts'],
        category=json_data['category'],
        title=json_data['title'],
    )
    formatted_instructions.append(prompt)

  text_dataset = Dataset.from_dict(
      {'instruction': formatted_instructions, 'answer': answers}
  )

  return text_dataset


def preprocess_text_dataset(
    text_dataset, dataset_name, prompt_template=None, split='train'
):
  """Preprocess text dataset."""
  # --- Movie JSON dataset ---
  if 'movie-json' in dataset_name:
    return _preprocess_text_dataset_fixed_prompt_instructions(
        text_dataset, prompt_dict=get_prompt_dict(prompt_template), split=split
    )
  # --- Mixture of PubMed and OpenReview ---
  elif 'pubmed_openreview_mixture' in dataset_name:
    return _preprocess_text_dataset_fixed_prompt_instructions(
        text_dataset, prompt_dict=get_prompt_dict(prompt_template), split=split
    )
  # --- Enron dataset ---
  elif dataset_name in [
      'enron',
      'enron_lenfilt_200-850',
      'enron_lenskew_5',
      'enron_sampled_5000',
  ]:
    return _preprocess_text_dataset_fixed_prompt_instructions(
        text_dataset,
        prompt_dict=get_prompt_dict(prompt_template),
        split=split,
        column='content',
    )
  elif dataset_name in [
      'enron-eval',
      'enron-generated',
      'enron-eval-generated',
  ]:
    return _preprocess_text_dataset_fixed_prompt_instructions(
        text_dataset,
        prompt_dict=get_prompt_dict(prompt_template),
        split=split,
        column='content',
    )
  elif dataset_name == 'enron-freeform-conditions':
    return _preprocess_text_dataset_fixed_prompt_instructions(
        text_dataset,
        prompt_dict=get_prompt_dict(prompt_template),
        split=split,
        column='extracted_features',
    )
  elif dataset_name == 'enron-condgen-freeform':
    return _preprocess_text_dataset_conditional_prompt_instructions(
        text_dataset,
        prompt_dict=get_prompt_dict(prompt_template),
        split=split,
        feature_name='extracted_features',
        text_name='content',
    )
  elif dataset_name == 'enron-freeform-e2e':
    return _preprocess_text_dataset_enron_freeform_e2e_instructions(
        text_dataset, prompt_dict=get_prompt_dict(prompt_template), split=split
    )
  elif dataset_name == 'enron-schema-conditions':
    return _preprocess_text_dataset_fixed_prompt_instructions(
        text_dataset,
        prompt_dict=get_prompt_dict(prompt_template),
        split=split,
        column='extracted_schema',
    )
  elif dataset_name == 'enron-condgen-schema':
    return _preprocess_text_dataset_conditional_prompt_instructions(
        text_dataset,
        prompt_dict=get_prompt_dict(prompt_template),
        split=split,
        feature_name='extracted_schema',
        text_name='content',
    )
  elif dataset_name == 'enron-condgen-topic':
    return _preprocess_text_dataset_conditional_prompt_instructions(
        text_dataset,
        prompt_dict=get_prompt_dict(prompt_template),
        split=split,
        feature_name='topic_keywords',
        text_name='content',
    )
  # --- PMC dataset ---
  elif dataset_name == 'PMC':
    return _preprocess_text_dataset_fixed_prompt_instructions(
        text_dataset,
        prompt_dict=get_prompt_dict(prompt_template),
        split=split,
        column='note',
    )
  elif dataset_name == 'PMC-conditions':
    return _preprocess_text_dataset_fixed_prompt_instructions(
        text_dataset,
        prompt_dict=get_prompt_dict(prompt_template),
        split=split,
        column='summary',
    )
  elif dataset_name == 'PMC-condgen':
    return _preprocess_text_dataset_conditional_prompt_instructions(
        text_dataset,
        prompt_dict=get_prompt_dict(prompt_template),
        split=split,
        feature_name='summary',
        text_name='note',
    )
  # --- bioRxiv dataset ---
  elif dataset_name == 'biorxiv':
    return _preprocess_text_dataset_fixed_prompt_instructions(
        text_dataset,
        prompt_dict=get_prompt_dict(prompt_template),
        split=split,
        column='abstract',
    )
  elif dataset_name == 'biorxiv-conditions':
    return _preprocess_text_dataset_fixed_prompt_instructions(
        text_dataset,
        prompt_dict=get_prompt_dict(prompt_template),
        split=split,
        column='selected_schema',
    )
  elif dataset_name == 'biorxiv-complex8-conditions':
    return _preprocess_text_dataset_fixed_prompt_instructions(
        text_dataset,
        prompt_dict=get_prompt_dict(prompt_template),
        split=split,
        column='schema',
    )
  elif dataset_name == 'biorxiv-complex8et-conditions':
    return _preprocess_text_dataset_fixed_prompt_instructions(
        text_dataset,
        prompt_dict=get_prompt_dict(prompt_template),
        split=split,
        column='schema',
    )
  elif dataset_name == 'biorxiv-category-conditions':
    return _preprocess_text_dataset_fixed_prompt_instructions(
        text_dataset,
        prompt_dict=get_prompt_dict(prompt_template),
        split=split,
        column='category',
    )
  elif dataset_name == 'biorxiv-condgen':
    return _preprocess_text_dataset_conditional_prompt_instructions(
        text_dataset,
        prompt_dict=get_prompt_dict(prompt_template),
        split=split,
        feature_name='selected_schema',
        text_name='abstract',
    )
  elif dataset_name == 'biorxiv-nl3-condgen':
    return _preprocess_text_dataset_biorxiv_nl3_condgen_instructions(
        text_dataset, prompt_dict=get_prompt_dict(prompt_template), split=split
    )
  elif dataset_name == 'biorxiv-complex8-condgen':
    return _preprocess_text_dataset_conditional_prompt_instructions(
        text_dataset,
        prompt_dict=get_prompt_dict(prompt_template),
        split=split,
        feature_name='schema',
        text_name='abstract',
    )
  elif dataset_name == 'biorxiv-complex8et-condgen':
    return _preprocess_text_dataset_conditional_prompt_instructions(
        text_dataset,
        prompt_dict=get_prompt_dict(prompt_template),
        split=split,
        feature_name='schema',
        text_name='abstract',
    )
  elif dataset_name == 'biorxiv-category-condgen':
    return _preprocess_text_dataset_conditional_prompt_instructions(
        text_dataset,
        prompt_dict=get_prompt_dict(prompt_template),
        split=split,
        feature_name='category',
        text_name='abstract',
    )
  elif dataset_name in ['biorxiv-generated']:
    return _preprocess_text_dataset_fixed_prompt_instructions(
        text_dataset,
        prompt_dict=get_prompt_dict(prompt_template),
        split=split,
        column='abstract',
    )
  else:
    raise f'Dataset {dataset_name} is not supported.'


# pylint: disable=invalid-name
def get_prompt_dict(prompt_template):
  """Get prompt dict."""
  # --- Unconditional generation ---
  if prompt_template == 'uncond_generation':
    PROMPT_DICT = {'type': 'uncond_generation', 'prompt': ''}
  # --- Movie JSON dataset ---
  elif prompt_template == 'moviejson-gen':
    PROMPT_DICT = {
        'type': 'movie-json',
        'prompt': (
            'Generate a JSON object about a movie. Generate only the JSON'
            ' object, do not include any other text.\n'
        ),
    }
  # --- Mixture of PubMed and OpenReview ---
  elif prompt_template == 'pubmed_openreview_mixture_generation':
    PROMPT_DICT = {
        'type': 'pubmed_openreview_mixture',
        'prompt': '<|endoftext|>',
    }
  # --- Enron dataset ---
  elif prompt_template == 'enron_generation':
    PROMPT_DICT = {'type': 'enron', 'prompt': 'Email Example:\n'}
  elif prompt_template == 'enron_evaluation':
    PROMPT_DICT = {'type': 'enron-eval', 'prompt': ''}
  elif prompt_template == 'enron-freeform-conditions_generation':
    PROMPT_DICT = {
        'type': 'enron_freeform_conditions',
        'prompt': (
            'A summary of key characteristics of an email (including'
            ' information of word count, sentiment, topic, and other'
            ' distinctive features):\n\n'
        ),
    }
  elif prompt_template == 'enron-schema-conditions_generation':
    PROMPT_DICT = {
        'type': 'enron_schema_conditions',
        'prompt': (
            '[INST]\nGenerate only a JSON object with these fields in'
            ' order:\nword_count, sentiment, tone, purpose, urgency, formality,'
            ' cta, attachments, email_type, sender_relation, topics,'
            ' main_action.\n[/INST]\n'
        ),
    }
  elif prompt_template == 'enron-condgen-freeform_generation':
    instruction = (
        'Generate a full email that matches the following summary of key'
        ' characteristics.'
    )

    PROMPT_DICT = {
        'type': 'enron_condgen_freeform',
        'prompt': '[INST]\n{instruction}\n\n{{feature}}\n[/INST]\n'.format(
            instruction=instruction
        ),
    }
  elif prompt_template == 'enron-condgen-schema_generation':
    instruction = (
        'Given the JSON object below, generate a complete email thread that'
        ' matches its attributes exactly. Output the email text (including any'
        ' quoted history).'
    )

    PROMPT_DICT = {
        'type': 'enron_condgen_schema',
        'prompt': '[INST]\n{instruction}\n\n{{feature}}\n[/INST]\n'.format(
            instruction=instruction
        ),
    }
  elif prompt_template == 'enron-condgen-topic_generation':
    instruction = (
        'Given the list of keywords below, generate a complete email thread'
        ' that matches the keywords. Output the email text (including any'
        ' quoted history).'
    )

    PROMPT_DICT = {
        'type': 'enron_condgen_topic',
        'prompt': '[INST]\n{instruction}\n\n{{feature}}\n[/INST]\n'.format(
            instruction=instruction
        ),
    }
  elif prompt_template == 'enron-freeform-e2e_generation':
    instruction = (
        'Generate a summary of the key characteristics of an email thread, and'
        ' then write out the full email. The summary should include information'
        ' of word count, sentiment, topic, and other distinctive features.'
    )

    PROMPT_DICT = {
        'type': 'enron_freeform_e2e',
        'prompt': '[INST]\n{instruction}\n[/INST]\n'.format(
            instruction=instruction
        ),
    }
  # --- PMC dataset ---
  elif prompt_template == 'PMC_generation':
    instruction = (
        'Please generate a clinical note presenting a thorough summary'
        ' encompassing the patient’s visit, medical history, symptoms,'
        ' administered treatments, and outcome of the intervention.'
    )
    PROMPT_DICT = {
        'type': 'PMC',
        'prompt': '<start_of_turn>user\n{instruction}\n<end_of_turn>\n<start_of_turn>model\n'.format(
            instruction=instruction
        ),
    }
  elif prompt_template == 'PMC-conditions_generation':
    instruction = (
        "Please generate a structured JSON summary of the patient's visit,"
        ' including the following sections: visit motivation, admission,'
        ' patient information, patient medical history, surgeries, symptoms,'
        ' medical examinations, diagnosis tests, treatments.'
    )
    PROMPT_DICT = {
        'type': 'PMC-conditions',
        'prompt': '<start_of_turn>user\n{instruction}\n<end_of_turn>\n<start_of_turn>model\n'.format(
            instruction=instruction
        ),
    }
  elif prompt_template == 'PMC-condgen_generation':
    instruction = (
        'Please generate a detailed clinical note based solely on the below'
        ' JSON summary, covering the patient’s visit, medical history,'
        ' symptoms, administered treatments, and outcome of the intervention.'
    )
    PROMPT_DICT = {
        'type': 'PMC-condgen',
        'prompt': '<start_of_turn>user\n{instruction}\n\n{{feature}}\n<end_of_turn>\n<start_of_turn>model\n'.format(
            instruction=instruction
        ),
    }
  # --- bioRxiv dataset ---
  elif prompt_template == 'biorxiv_generation':
    instruction = (
        'Generate a synthetic scientific abstract in the style of a bioRxiv'
        ' paper.'
    )
    PROMPT_DICT = {
        'type': 'biorxiv',
        'prompt': '<start_of_turn>user\n{instruction}\n<end_of_turn>\n<start_of_turn>model\n'.format(
            instruction=instruction
        ),
    }
  elif prompt_template == 'biorxiv-conditions_generation':
    instruction = (
        'Please generate a structured JSON summary of a scientific abstract,'
        ' including the following fields: token_counts, category, title.'
    )
    PROMPT_DICT = {
        'type': 'biorxiv-conditions',
        'prompt': '<start_of_turn>user\n{instruction}\n<end_of_turn>\n<start_of_turn>model\n'.format(
            instruction=instruction
        ),
    }
  elif prompt_template == 'biorxiv-complex8-conditions_generation':
    instruction = (
        'Please generate a structured JSON summary of a scientific abstract,'
        ' including the following fields: token_counts, primary_research_area,'
        ' model_organism, experimental_approach, dominant_data_type,'
        ' research_focus_scale, disease_mention, sample_size, research_goal.'
    )
    PROMPT_DICT = {
        'type': 'biorxiv-complex8-conditions',
        'prompt': '<start_of_turn>user\n{instruction}\n<end_of_turn>\n<start_of_turn>model\n'.format(
            instruction=instruction
        ),
    }
  elif prompt_template == 'biorxiv-complex8et-conditions_generation':
    instruction = (
        'Please generate a structured JSON summary of a scientific abstract,'
        ' including the following fields: primary_research_area,'
        ' model_organism, experimental_approach, dominant_data_type,'
        ' research_focus_scale, disease_mention, sample_size, research_goal.'
    )
    PROMPT_DICT = {
        'type': 'biorxiv-complex8et-conditions',
        'prompt': '<start_of_turn>user\n{instruction}\n<end_of_turn>\n<start_of_turn>model\n'.format(
            instruction=instruction
        ),
    }
  elif prompt_template == 'biorxiv-nl3-condgen_generation':
    instruction = (
        'Write a scientific abstract in the style of a bioRxiv paper, in the'
        ' {category} category, with a target token count of {token_counts}'
        ' tokens.\n\nTitle: {title}\n\nAbstract: '
    )
    PROMPT_DICT = {
        'type': 'biorxiv-nl3-condgen',
        'prompt': '<start_of_turn>user\n{instruction}\n<end_of_turn>\n<start_of_turn>model\n'.format(
            instruction=instruction
        ),
    }
  elif prompt_template == 'biorxiv-category-conditions_generation':
    instruction = 'Please generate a category of a scientific paper on bioRxiv.'
    PROMPT_DICT = {
        'type': 'biorxiv-category-conditions',
        'prompt': '<start_of_turn>user\n{instruction}\n<end_of_turn>\n<start_of_turn>model\n'.format(
            instruction=instruction
        ),
    }
  elif prompt_template == 'biorxiv-condgen_generation':
    instruction = (
        'Please generate a synthetic scientific abstract that matches the below'
        ' JSON summary, in the style of a bioRxiv paper.'
    )
    PROMPT_DICT = {
        'type': 'biorxiv-condgen',
        'prompt': '<start_of_turn>user\n{instruction}\n\n{{feature}}\n<end_of_turn>\n<start_of_turn>model\n'.format(
            instruction=instruction
        ),
    }
  elif prompt_template == 'biorxiv-category-condgen_generation':
    instruction = (
        'Please generate a synthetic scientific abstract that belongs to the'
        ' below category, in the style of a bioRxiv paper.'
    )
    PROMPT_DICT = {
        'type': 'biorxiv-category-condgen',
        'prompt': '<start_of_turn>user\n{instruction}\n\n{{feature}}\n<end_of_turn>\n<start_of_turn>model\n'.format(
            instruction=instruction
        ),
    }
  elif prompt_template == 'biorxiv_evaluation':
    PROMPT_DICT = {'type': 'biorxiv-eval', 'prompt': ''}
  else:
    raise ValueError(f'Prompt template {prompt_template} is not supported.')
  return PROMPT_DICT


class TokenizedSupervisedInstructDataset(Dataset):
  """Tokenize the concatenated text from dataset of (insturction, answer) pairs."""

  def __init__(
      self,
      dataset_or_name,
      dataset_path,
      tokenizer,
      split='train',
      max_instruction_length=128,
      max_answer_length=384,
      num_proc=4,
      tokenize_type='clm',
      prompt_template=None,
  ):

    # we shall build the text dataset from scratch
    # a processed text dataset contain two columns: instruction and answer
    if isinstance(dataset_or_name, str):
      dataset_name = dataset_or_name
      if dataset_name == 'movie-json':
        text_dataset = load_from_disk(
            '../data/wikipedia-movie-data-splits'
        )
      elif dataset_name in [
          'biorxiv',
          'biorxiv-category-conditions',
          'biorxiv-category-condgen',
      ]:
        data_dir = '../data/biorxiv'
        text_dataset = load_dataset(
            'csv',
            data_files={
                'train': os.path.join(data_dir, 'train.csv'),
                'validation': os.path.join(data_dir, 'validation.csv'),
                'test': os.path.join(data_dir, 'test.csv'),
            },
        )
      elif dataset_name in [
          'biorxiv-conditions',
          'biorxiv-condgen',
          'biorxiv-nl3-condgen',
      ]:
        data_dir = '../data/biorxiv'
        text_dataset = load_dataset(
            'csv',
            data_files={
                'train': osp.join(data_dir, 'train_features_v6_selected.csv'),
                'validation': osp.join(
                    data_dir, 'validation_features_v6_selected.csv'
                ),
                'test': osp.join(data_dir, 'test_features_v6_selected.csv'),
            },
        )
      elif dataset_name in [
          'biorxiv-complex8et-conditions',
          'biorxiv-complex8et-condgen',
      ]:
        data_dir = '../data/biorxiv'
        text_dataset = load_dataset(
            'csv',
            data_files={
                'train': osp.join(
                    data_dir,
                    'biorxiv_json_schema_v2et_train_gemini-2.5-flash_parsed.csv',
                ),
                'validation': osp.join(
                    data_dir,
                    'biorxiv_json_schema_v2et_valid_gemini-2.5-flash_parsed.csv',
                ),
            },
        )
      elif dataset_name == 'pubmed_openreview_mixture':
        text_dataset = load_from_disk(
            '../data/pubmed_openreview_mixture'
        )
      elif dataset_name in ['enron', 'enron-eval']:
        data_dir = '../data/enron_data'
        if dataset_name == 'enron':
          text_dataset = load_dataset(
              'csv',
              data_files={
                  'train': osp.join(data_dir, 'train.csv'),
                  'validation': osp.join(data_dir, 'valid.csv'),
                  'test': osp.join(data_dir, 'test.csv'),
              },
          )
        else:
          text_dataset = load_dataset(
              'csv',
              data_files={
                  'train': osp.join(data_dir, 'valid.csv'),
                  'validation': osp.join(data_dir, 'valid.csv'),
                  'test': osp.join(data_dir, 'test.csv'),
              },
          )
      elif dataset_name == 'enron-eval-generated':
        data_dir = '../data/enron_data'
        train_dataset_full = load_dataset(
            'csv',
            data_files={'train': osp.join(data_dir, 'valid.csv')},
        )['train']
        train_dataset = train_dataset_full.remove_columns(
            [col for col in train_dataset_full.column_names if col != 'content']
        )

        valid_dataset = load_dataset(
            'csv',
            data_files={'validation': dataset_path},
        )['validation']

        text_dataset = DatasetDict({
            'train': train_dataset,
            'validation': valid_dataset,
        })

      elif dataset_name in [
          'enron_lenfilt_200-850',
          'enron_lenskew_5',
          'enron_sampled_5000',
      ]:
        data_dir = '../data/enron_data'
        dataset_str = dataset_name.replace('enron_', '')
        text_dataset = load_dataset(
            'csv',
            data_files={
                'train': osp.join(data_dir, f'train_{dataset_str}.csv'),
                'validation': osp.join(data_dir, f'valid_{dataset_str}.csv'),
                'test': osp.join(data_dir, f'test_{dataset_str}.csv'),
            },
        )
      elif dataset_name in [
          'enron-freeform-conditions',
          'enron-condgen-freeform',
          'enron-freeform-e2e',
      ]:
        data_dir = '../data/enron_data'
        text_dataset = load_dataset(
            'csv',
            data_files={
                'train': osp.join(
                    data_dir, 'train_features_gemini-2.0-flash.csv'
                ),
                'validation': osp.join(
                    data_dir, 'valid_features_gemini-2.0-flash.csv'
                ),
                # "test": osp.join(data_dir, "test.csv"),
            },
        )
      elif dataset_name in [
          'enron-schema-conditions',
          'enron-condgen-schema',
      ]:
        data_dir = '../data/enron_data'
        text_dataset = load_dataset(
            'csv',
            data_files={
                'train': osp.join(
                    data_dir, 'extracted_schema_train_filtered.csv'
                ),
                'validation': osp.join(data_dir, 'extracted_schema_valid.csv'),
            },
        )
      elif dataset_name in [
          'enron-condgen-topic',
      ]:
        data_dir = '../data/enron_data'
        text_dataset = load_dataset(
            'csv',
            data_files={
                'train': osp.join(
                    data_dir, 'train_topics_BERTopic_Wikipedia.csv'
                ),
                'validation': osp.join(
                    data_dir, 'valid_topics_BERTopic_Wikipedia.csv'
                ),
            },
        )
      elif dataset_name in ['PMC', 'PMC-conditions', 'PMC-condgen']:
        text_dataset = load_dataset(
            'json',
            data_files={
                'train': (
                    '../data/PMC_data/PMC_train.jsonl'
                ),
                'validation': (
                    '../data/PMC_data/PMC_valid.jsonl'
                ),
            },
        )
      elif dataset_name == 'self_instruct':
        raise NotImplementedError
      elif dataset_name == 'enron-generated':
        data_dir = '../data/enron_data'
        # Load train dataset (only 'content' column)
        train_dataset = load_dataset(
            'csv',
            data_files={'train': dataset_path},
        )['train']

        # Load validation dataset and extract only 'content' column
        valid_dataset_full = load_dataset(
            'csv',
            data_files={'validation': osp.join(data_dir, 'valid.csv')},
        )['validation']
        valid_dataset = valid_dataset_full.remove_columns(
            [col for col in valid_dataset_full.column_names if col != 'content']
        )

        # Prepare together into one DatasetDict
        text_dataset = DatasetDict({
            'train': train_dataset,
            'validation': valid_dataset,
        })
      elif dataset_name == 'biorxiv-generated':
        data_dir = '../data/biorxiv'
        # Load train dataset (only 'content' column)
        if dataset_path.endswith('.csv'):
          train_dataset = load_dataset(
              'csv',
              data_files={'train': dataset_path},
          )['train'].rename_column('generated_text', 'abstract')
        elif dataset_path.endswith('.jsonl'):
          train_dataset = load_dataset(
              'json',
              data_files={'train': dataset_path},
          )['train'].rename_column('generated_text', 'abstract')
        else:
          raise ValueError(
              f'Unsupported dataset path {dataset_path} for biorxiv-generated'
              ' dataset.'
          )

        # Load validation dataset and extract only 'content' column
        valid_dataset_full = load_dataset(
            'csv',
            data_files={'validation': osp.join(data_dir, 'validation.csv')},
        )['validation']
        valid_dataset = valid_dataset_full.remove_columns([
            col for col in valid_dataset_full.column_names if col != 'abstract'
        ])

        # Prepare together into one DatasetDict
        text_dataset = DatasetDict({
            'train': train_dataset,
            'validation': valid_dataset,
        })
      elif (
          # dataset_name == 'chatbot_arena_instructions_train180k'
          'chatbot_arena_instructions_train180k' in dataset_name
          or 'labelled' in dataset_name
      ):
        text_dataset = DatasetDict.load_from_disk('../data/' + dataset_name)
      else:
        raise NotImplementedError

    # we alread have a text dataset, no need to build from scratch
    elif isinstance(dataset_or_name, dict):
      print('creating dataset from python dict')
      text_dataset = dataset_or_name
      dataset_name = 'customized_instructions'

    self.text_dataset = text_dataset
    self.tokenizer = tokenizer
    self.max_instruction_length = max_instruction_length
    self.max_answer_length = max_answer_length
    self.num_proc = num_proc
    self.tokenize_type = tokenize_type
    self.dataset_name = dataset_name
    self.dataset_split = split
    self.prompt_template = prompt_template
    self.get_tokenized_dataset()

  def get_tokenized_dataset(self):
    processed_text_dataset = preprocess_text_dataset(
        self.text_dataset,
        self.dataset_name,
        split=self.dataset_split,
        prompt_template=self.prompt_template,
    )

    print('processed_text_dataset', processed_text_dataset)
    print('example', processed_text_dataset[0:5])

    # tokenize the text dataset
    if self.tokenize_type == 'clm':
      # this option concatenates the instruction and answer, and tokenize
      # the concatenated text.
      # the loss on instruction is ignored
      tokenize_func = clm_tokenize_function
    else:
      # this option only tokenizes the instruction
      # usually used during inference
      tokenize_func = eval_tokenize_function

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


# Adapted from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
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
