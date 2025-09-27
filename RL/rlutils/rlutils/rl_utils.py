"""Utility functions for RL training experiments."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import logging
import os
import random
import re
from statistics import mean, stdev
import time
from typing import Optional

from accelerate import Accelerator
import numpy as np
from peft import LoraConfig, TaskType
import requests as http_requests
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    Adafactor,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from trl import AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

from .ppo_ptx_config import PPOPTXConfig

logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
  """The name of the Casual LM model we wish to fine with PPO"""

  min_length: Optional[int] = field(
      default=20, metadata={"help": "minimum length for generation"}
  )
  model_name: Optional[str] = field(
      default="facebook/opt-125m", metadata={"help": "the model name"}
  )
  adapter_model_name: Optional[str] = field(
      default=None, metadata={"help": "the model name"}
  )
  reward_model_name: Optional[str] = field(
      default="function:bagofwords", metadata={"help": "the reward model name"}
  )
  tokenizer_name: Optional[str] = field(
      default=None, metadata={"help": "the tokenizer name"}
  )
  dataset_name: Optional[str] = field(
      default="ultra", metadata={"help": "the dataset name"}
  )
  dataset_path: Optional[str] = field(
      default=None, metadata={"help": "the dataset path"}
  )
  ptx_dataset_path: Optional[str] = field(
      default="sft_fullprompt_9_generated_biorxiv-condgen_model-gemma-3-1b-pt_dp-eps-1-np-15.18-12.75-complex8et-aimn-1e-3-cosine_temp-1.0_tp-0.95_tk-0_eval_n-50000.csv",
      metadata={"help": "the ptx dataset path"},
  )
  val_data_path: Optional[str] = field(
      default="toxicity.csv", metadata={"help": "the validation dataset path"}
  )
  implicit: Optional[bool] = field(
      default=False,
      metadata={"help": "whether to use implicit reward model or not"},
  )
  log_with: Optional[str] = field(
      default=None, metadata={"help": "use 'wandb' to log with wandb"}
  )
  kl_penalty: Optional[str] = field(
      default="kl",
      metadata={"help": "kl penalty setup, can use dpoplus for that"},
  )
  learning_rate: Optional[float] = field(
      default=1.41e-5, metadata={"help": "the learning rate"}
  )
  max_length: Optional[int] = field(
      default=256, metadata={"help": "maximum length for generation"}
  )
  mini_batch_size: Optional[int] = field(
      default=128, metadata={"help": "the PPO minibatch size"}
  )
  batch_size: Optional[int] = field(
      default=32, metadata={"help": "the batch size"}
  )
  ppo_epochs: Optional[int] = field(
      default=4, metadata={"help": "the number of ppo epochs"}
  )
  gradient_accumulation_steps: Optional[int] = field(
      default=1, metadata={"help": "the number of gradient accumulation steps"}
  )
  adafactor: Optional[bool] = field(
      default=False, metadata={"help": "whether to use the adafactor optimizer"}
  )
  early_stopping: Optional[bool] = field(
      default=False, metadata={"help": "whether to early stop"}
  )
  save_rollouts: Optional[bool] = field(
      default=False, metadata={"help": "save rollouts, rewards to file"}
  )
  target_kl: Optional[float] = field(
      default=6, metadata={"help": "kl target for early stopping"}
  )
  load_in_8bit: Optional[bool] = field(
      default=False, metadata={"help": "whether to load in 8 bit"}
  )
  reward_baseline: Optional[float] = field(
      default=0,
      metadata={"help": "a baseline value that is subtracted from the reward"},
  )
  scale_reward: Optional[float] = field(
      default=0,
      metadata={
          "help": (
              "whether to omit outputs that don't fit in length context or not"
          )
      },
  )
  horizon: Optional[float] = field(
      default=10000, metadata={"help": "horizon for adaptive KL"}
  )
  save_freq: Optional[int] = field(
      default=None, metadata={"help": "n steps to save the model"}
  )
  output_dir: Optional[str] = field(
      default="checkpoints/debugging", metadata={"help": "model save directory"}
  )
  seed: Optional[int] = field(default=1, metadata={"help": "the seed"})
  steps: Optional[int] = field(
      default=10000, metadata={"help": "number of epochs"}
  )
  init_kl_coef: Optional[float] = field(
      default=0.2,
      metadata={
          "help": (
              "Initial KL penalty coefficient (used for adaptive and linear"
              " control)"
          )
      },
  )
  adap_kl_ctrl: Optional[bool] = field(default=True)
  ptx_coef: Optional[float] = field(
      default=0.1,
      metadata={"help": "coefficient for the PTX loss component"},
  )
  ptx_coef_initial: Optional[float] = field(
      default=0.1,
      metadata={"help": "initial coefficient for the PTX loss component"},
  )
  ptx_coef_final: Optional[float] = field(
      default=0.1,
      metadata={"help": "final coefficient for the PTX loss component"},
  )
  gen_bsize: Optional[int] = field(
      default=1,
      metadata={"help": "how many outputs to over-generate per sample"},
  )
  temperature: Optional[float] = field(
      default=1,
      metadata={"help": "sampling temperature for generation"},
  )
  generators_json: Optional[str] = field(
      default=None,
      metadata={
          "help": (
              "json file indicating which checkpoints to use for rollouts at"
              " various points"
          )
      },
  )
  wandb_project: Optional[str] = field(
      default="llamatrl", metadata={"help": "wandb project name"}
  )
  run_name: Optional[str] = field(
      default="llamatrl", metadata={"help": "wandb run name"}
  )
  gen_data_dir: Optional[str] = field(
      default=None, metadata={"help": "directory to save generated data"}
  )


API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
  raise ValueError("GEMINI_API_KEY environment variable not set.")


def load_models(script_args, dev=0):
  """Loads all models, tokenizer, and configuration objects for training."""

  current_device = Accelerator().local_process_index

  tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
  if getattr(tokenizer, "pad_token", None) is None:
    logger.warning("Tokenizer has no pad token; setting it to EOS token.")
    tokenizer.pad_token = tokenizer.eos_token

  config = PPOPTXConfig(
      model_name=script_args.model_name,
      learning_rate=script_args.learning_rate,
      log_with="wandb",
      batch_size=script_args.batch_size,
      mini_batch_size=script_args.mini_batch_size,
      gradient_accumulation_steps=script_args.gradient_accumulation_steps,
      early_stopping=script_args.early_stopping,
      target_kl=script_args.target_kl,
      ppo_epochs=script_args.ppo_epochs,
      seed=script_args.seed,
      cliprange=0.2,
      cliprange_value=0.2,
      vf_coef=0.1,
      horizon=script_args.horizon,
      target=script_args.target_kl,
      init_kl_coef=script_args.init_kl_coef,
      adap_kl_ctrl=script_args.adap_kl_ctrl,
      steps=script_args.steps,
      gamma=1,
      lam=0.95,
      kl_penalty=script_args.kl_penalty,
      remove_unused_columns=False,
      ptx_coef=script_args.ptx_coef,
      ptx_coef_initial=script_args.ptx_coef_initial,
      ptx_coef_final=script_args.ptx_coef_final,
  )
  lora_config = LoraConfig(
      r=16,
      lora_alpha=32,
      lora_dropout=0.05,
      bias="none",
      task_type="CAUSAL_LM",
      target_modules=["self_attn.q_proj", "self_attn.v_proj"],
  )
  model = AutoModelForCausalLMWithValueHead.from_pretrained(
      script_args.model_name,
      load_in_8bit=True if script_args.load_in_8bit else False,
      device_map={"": current_device},
      peft_config=lora_config,
      torch_dtype=torch.bfloat16,
      attn_implementation="eager",
  )

  optimizer = None
  if script_args.adafactor:
    optimizer = Adafactor(
        filter(lambda p: p.requires_grad, model.parameters()),
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        lr=config.learning_rate,
    )

  model.gradient_checkpointing_disable()
  return config, tokenizer, model, optimizer


def get_rollouts(
    ppo_trainer,
    question_tensors,
    output_length_sampler,
    script_args,
    generation_kwargs,
):
  """Generates responses from the policy model for a batch of queries."""
  kl_mask = [1] * len(question_tensors)
  get_unwrapped(ppo_trainer).set_adapter("default")

  generation_kwargs["length_sampler"] = output_length_sampler
  response_tensors = ppo_trainer.generate(
      question_tensors,
      batch_size=script_args.gen_bsize,
      return_prompt=False,
      **generation_kwargs,
  )
  return response_tensors, kl_mask


def get_unwrapped(ppo_trainer):
  """Unwraps the base model from the trainer's accelerator."""
  return ppo_trainer.accelerator.unwrap_model(
      ppo_trainer.model
  ).pretrained_model


def process_reward(
    texts,
    rmname,
    reward_model,
    script_args,
    response_tensors,
    metadata,
    reward_tokenizer,
    input_prompts=None,
    reference=None,
):
  """Calculates reward scores for a batch of texts using a specified model.

  This function acts as a dispatcher, routing texts to the correct reward
  calculation logic based on the reward model's name (`rmname`). It handles
  local pipelines, external API calls (e.g., LLM-as-a-judge), and custom
  scoring functions.
  """

  if rmname == "movie-json":

    def get_scores_movie_structure(s):
      """Calculates a reward score for a generated string based on two stages:

      1. JSON Parsability: +1.0 for a valid JSON object, -1.0 otherwise.
      2. Correct Field Names (if parsable): Rewards/penalties for matching,
      extra, or missing keys, plus a bonus for a perfect key match.
      """
      REWARD_PER_KEY_OPERATION = 0.1  # Value for 'x'
      ALL_KEYS_CORRECT_BONUS = (
          0.4  # Bonus if all expected keys are present and no extras
      )

      expected_keys = {"title", "year", "cast", "genres", "href", "extract"}

      # Stage 1: JSON Parsability
      try:
        data = json.loads(s)
        if not isinstance(data, dict):
          # Parsed correctly, but it's not a JSON object (e.g., it's a JSON array "[]")
          return -1.0
        # Successfully parsed as a JSON object
        parsability_reward = 1.0
      except json.JSONDecodeError:
        # String is not valid JSON
        return -1.0

      # Stage 2: Correct Field Names
      # This part is reached only if parsability_reward is 1.0.
      actual_keys = set(data.keys())

      # Calculate scores based on key comparisons
      correctly_present_keys = actual_keys.intersection(expected_keys)
      extra_keys = actual_keys.difference(expected_keys)
      missing_keys = expected_keys.difference(actual_keys)

      field_name_score = 0.0
      # Reward for each correctly present key
      field_name_score += len(correctly_present_keys) * REWARD_PER_KEY_OPERATION
      # Penalty for each extra key
      field_name_score -= len(extra_keys) * REWARD_PER_KEY_OPERATION
      # Penalty for each missing key
      field_name_score -= len(missing_keys) * REWARD_PER_KEY_OPERATION

      # Check for a perfect match (all expected keys present, no extra keys)
      is_perfect_key_match = len(extra_keys) == 0 and len(missing_keys) == 0

      if is_perfect_key_match:
        field_name_score += ALL_KEYS_CORRECT_BONUS

      total_reward = parsability_reward + field_name_score

      return round(total_reward, 4)

    rewards = [get_scores_movie_structure(s) for s in texts]

  elif rmname == "llmjudge_enron-freeform-conditions":

    MODEL_NAME = "gemini-2.5-flash-lite-preview-06-17"
    URL = (
        "https://generativelanguage.googleapis.com"
        f"/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    )
    MAX_WORKERS = 32
    MAX_RETRIES = 8

    def query(prompt: str) -> int:
      payload = {"contents": [{"parts": [{"text": prompt}]}]}
      headers = {"Content-Type": "application/json"}
      for attempt in range(1, MAX_RETRIES + 1):
        try:
          resp = http_requests.post(
              URL, json=payload, headers=headers, timeout=(10, 120)
          )
          resp.raise_for_status()
          data = resp.json()
          # pull out the generated text
          gen_text = data["candidates"][0]["content"]["parts"][0][
              "text"
          ].strip()
          # find the integer after "holistic_score":
          m = re.search(r'"holistic_score"\s*:\s*(\d+)', gen_text)
          if m:
            return float(m.group(1))
          # fallback if parsing failed
          return float(np.random.randint(1, 6))
        except Exception as e:
          wait_time = (2 ** (attempt - 1)) + random.random()
          logger.warning(
              f"[Judge API] error ({e}), retrying in {wait_time:.1f}s "
              f"(attempt {attempt}/{MAX_RETRIES})"
          )
          time.sleep(wait_time)

      # give up after MAX_RETRIES
      logger.error("[Judge API] all retries failed, returning random score")
      return np.random.randint(1, 6)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
      rewards = list(executor.map(query, texts))

    return rewards

  elif rmname == "llmjudge_biorxiv-complex8et-conditions":

    MODEL_NAME = "gemini-2.5-flash-lite-preview-06-17"
    URL = (
        "https://generativelanguage.googleapis.com"
        f"/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    )
    MAX_WORKERS = 32
    MAX_RETRIES = 8

    def query(input_text: str) -> int:
      prompt, input_json_text = input_text
      payload = {"contents": [{"parts": [{"text": prompt}]}]}
      headers = {"Content-Type": "application/json"}

      def scoring_biorxiv_feature_matching(gen_text, input_json_text):
        """Extracts the score from the generated text based on the input JSON."""

        logger.debug("-----gen_text------")
        logger.debug(gen_text)

        logger.debug("-----input_json_text------")
        logger.debug(input_json_text)

        try:
          parsed_json_gen = json.loads(gen_text[7:-3])
        except json.JSONDecodeError:
          try:
            parsed_json_gen = json.loads(gen_text)
          except json.JSONDecodeError:
            logger.warning(
                "Error parsing generated text JSON. Falling back to a random"
                " low score."
            )
            return np.random.randint(0, 5)

        input_json = json.loads(input_json_text)

        score = 0
        for key in input_json.keys():
          if key in parsed_json_gen and parsed_json_gen[key] == input_json[key]:
            score += 1
        return score

      for attempt in range(1, MAX_RETRIES + 1):
        try:
          resp = http_requests.post(
              URL, json=payload, headers=headers, timeout=(10, 120)
          )
          resp.raise_for_status()
          data = resp.json()
          # pull out the generated text
          gen_text = data["candidates"][0]["content"]["parts"][0][
              "text"
          ].strip()

          return scoring_biorxiv_feature_matching(gen_text, input_json_text)

        except Exception as e:
          wait_time = (2 ** (attempt - 1)) + random.random()
          logger.warning(
              f"[Judge API] error ({e}), retrying in {wait_time:.1f}s "
              f"(attempt {attempt}/{MAX_RETRIES})"
          )
          time.sleep(wait_time)

      # give up after MAX_RETRIES
      logger.error("[Judge API] all retries failed, returning random score")
      return np.random.randint(0, 5)

    prefix = (
        "<start_of_turn>user\nPlease generate a synthetic scientific abstract"
        " that matches the below JSON summary, in the style of a bioRxiv"
        " paper.\n\n"
    )
    prefix_w_bos = (
        "<bos><start_of_turn>user\nPlease generate a synthetic scientific"
        " abstract that matches the below JSON summary, in the style of a"
        " bioRxiv paper.\n\n"
    )
    suffix = "\n<end_of_turn>\n<start_of_turn>model\n"

    input_json_text = [
        q.removeprefix(prefix).removeprefix(prefix_w_bos).removesuffix(suffix)
        for q in input_prompts
    ]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
      rewards = list(executor.map(query, zip(texts, input_json_text)))

    return rewards

  else:
    raise NotImplementedError(f"Reward model '{rmname}' is not implemented.")

  return rewards


def train_loop(
    script_args,
    ppo_trainer,
    reward_model,
    tokenizer,
    qaform,
    min_length=20,
    reward_tokenizer=None,
):
  """The main training loop for PPO.

  This function orchestrates the entire RL process. It iterates through the
  dataset, generates responses from the policy model, queries the reward model
  for scores, and executes PPO optimization steps. It also handles logging
  statistics and saving model checkpoints.
  """
  generation_kwargs = {
      "min_length": -1,
      "top_k": 0.0,
      "top_p": 1.0,
      "do_sample": True,
      "temperature": script_args.temperature,
      "pad_token_id": tokenizer.eos_token_id,
  }

  current_device = Accelerator().local_process_index

  logger.info(f"Reward model for training: {reward_model}")

  min_len = script_args.max_length - 2

  output_length_sampler = LengthSampler(min_len, script_args.max_length)

  # compute moving averages over last 10 steps
  running_means = []
  running_stds = []
  rmname = script_args.reward_model_name

  ppo_trainer.save_pretrained(script_args.output_dir + f"step_0")
  total_epoch = 0
  pbar = tqdm(total=script_args.steps)
  while total_epoch < script_args.steps:
    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
      if script_args.ptx_coef_initial > script_args.ptx_coef_final:
        ptx_coef_current = script_args.ptx_coef_initial - (
            script_args.ptx_coef_initial - script_args.ptx_coef_final
        ) * (total_epoch / script_args.steps)
      else:
        ptx_coef_current = script_args.ptx_coef

      logger.info(f"Current PTX coefficient: {ptx_coef_current:.4f}")

      epoch_start_time = time.time()

      logger.debug(f"Batch keys: {list(batch.keys())}")

      question_tensors = batch["input_ids"]

      if total_epoch == 0:
        # SANITY CHECKING
        logger.info("PPO input")
        logger.info(tokenizer.batch_decode(question_tensors))

      # --- 1. generation ---
      timing = dict()
      t = time.time()
      with torch.no_grad():
        response_tensors, kl_mask = get_rollouts(
            ppo_trainer,
            question_tensors,
            output_length_sampler,
            script_args,
            generation_kwargs,
        )
      timing["time/ppo/sample_generation"] = time.time() - t

      # --- 2. scoring & reward processing ---
      t = time.time()

      batch["response"] = tokenizer.batch_decode(
          response_tensors, skip_special_tokens=True
      )

      if total_epoch == 0:
        logger.info("QAForm Input Example:")
        logger.info(qaform(batch["query"][0], batch["response"][0]))

      texts = [qaform(q, r) for q, r in zip(batch["query"], batch["response"])]

      rewards = process_reward(
          texts,
          rmname,
          reward_model,
          script_args,
          response_tensors,
          batch,
          reward_tokenizer,
          input_prompts=batch["query"],
      )

      if total_epoch == 0:
        logger.info(f"Initial rewards for first batch: {rewards}")
      logger.info(
          f"Reward Mean: {mean(rewards):.2f}, Reward Stdev:"
          f" {stdev(rewards):.2f}"
      )

      rewards = [torch.tensor(r).to(current_device) for r in rewards]

      # using running mean and running stdev
      rws = [float(f) for f in rewards]
      running_means.append(mean(rws))
      running_stds.append(stdev(rws))
      meanval = mean(running_means[-10:])
      sigma = mean(running_stds[-10:])

      logrewards = [float(r.item()) for r in rewards]

      # reward scaling from secrets of PPO
      if script_args.scale_reward == 1:
        rewards = [(rw - meanval) / (sigma + 1e-8) for rw in rewards]
        if total_epoch < 10:
          logger.debug(f"Scaled rewards (first 10 steps): {rewards}")
      timing["time/ppo/sample_scoring"] = time.time() - t

      # --- 3. PPO optimization step ----
      t = time.time()
      stats = ppo_trainer.step(
          question_tensors,
          response_tensors,
          rewards,
          timing,
          gen_data_dir=script_args.gen_data_dir,
          ptx_coef=ptx_coef_current,
      )
      timing["time/ppo/optimization_step"] = time.time() - t

      # --- 4. finalize timing, logging, and saving ---
      t = time.time()
      ppo_trainer.log_stats(stats, batch, logrewards)
      if (
          script_args.save_freq
          and (total_epoch + 1) % script_args.save_freq == 0
      ):
        ppo_trainer.save_pretrained(
            script_args.output_dir + f"step_{total_epoch+1}"
        )
      timing["time/ppo/log_save"] = time.time() - t

      timing["time/ppo/epoch_total"] = time.time() - epoch_start_time

      total_epoch += 1
      pbar.update(1)

  pbar.close()
