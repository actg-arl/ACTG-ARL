"""An extension of the TRL PPOTrainer to include a pre-training objective (PTX)."""

from itertools import cycle
from typing import List, Optional
import warnings
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from trl import PPOConfig, PPOTrainer
from trl.core import (
    WANDB_PADDING,
    PPODecorators,
    convert_to_scalar,
    logprobs_from_logits,
    masked_mean,
    stack_dicts,
    stats_to_np,
)
from .data_utils import DataCollatorForSupervisedDataset

class PPOPTXTrainer(PPOTrainer):
  """A PPOTrainer that incorporates a pretraining objective (PTX) to prevent catastrophic forgetting."""

  def __init__(self, *args, pretrain_dataset=None, **kwargs):
    super().__init__(*args, **kwargs)

    if pretrain_dataset is None:
      raise ValueError("A `pretrain_dataset` must be provided.")

    self.pretrain_data_collator = DataCollatorForSupervisedDataset(
        tokenizer=self.tokenizer, device=self.accelerator.device
    )

    self.pretrain_dataloader = self.accelerator.prepare(
        DataLoader(
            pretrain_dataset,
            batch_size=self.config.mini_batch_size,
            collate_fn=self.pretrain_data_collator,
            shuffle=True,
            drop_last=True,
        )
    )
    self.pretrain_dataloader_iter = cycle(self.pretrain_dataloader)

  def _calculate_ptx_loss(self):
    """Calculates the scaled pre-training (PTX) loss."""
    ptx_batch = next(self.pretrain_dataloader_iter)

    ptx_num_tokens = torch.sum(ptx_batch["labels"] != -100).detach()
    loss_denom = self.config.block_size

    ptx_outputs = self.model(**ptx_batch)
    ptx_logits = (
        ptx_outputs[0] if isinstance(ptx_outputs, tuple) else ptx_outputs.logits
    )

    # compute the cross-entropy loss
    vocab_size = ptx_logits.size(-1)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    unscaled_ptx_loss = loss_fct(
        ptx_logits[..., :-1, :].contiguous().view(-1, vocab_size),
        ptx_batch["labels"][..., 1:].contiguous().view(-1),
    )

    ptx_loss = unscaled_ptx_loss * (ptx_num_tokens / loss_denom)
    return ptx_loss

  def train_minibatch(
      self,
      old_logprobs: torch.FloatTensor,
      values: torch.FloatTensor,
      logprobs: torch.FloatTensor,
      logits: torch.FloatTensor,
      vpreds: torch.FloatTensor,
      mask: torch.LongTensor,
      advantages: torch.FloatTensor,
      returns: torch.FloatTensor,
      ptx_batch: Optional[dict] = None,
      ptx_coef: float = 0.1,
  ):
    """Performs a single optimization step on a mini-batch of data.

    This overridden method computes the standard PPO loss, adds the PTX
    loss, and then performs a backward pass and optimizer step.
    """
    self.model.train()

    # 1. Calculate PPO loss
    loss_p, loss_v, train_stats = self.loss(
        old_logprobs,
        values,
        logits,
        vpreds,
        logprobs,
        mask,
        advantages,
        returns,
    )
    ppo_loss = loss_p + loss_v

    # 2. Calculate PTX loss
    ptx_loss = self._calculate_ptx_loss()

    # 3. Combine losses
    total_loss = ppo_loss + ptx_coef * ptx_loss

    # 4. Backward pass and optimization
    self.accelerator.backward(total_loss)
    if self.config.max_grad_norm is not None:
      if self.accelerator.sync_gradients:
        self.accelerator.clip_grad_norm_(
            self.model_params, self.config.max_grad_norm
        )
    self.optimizer.step()
    self.optimizer.zero_grad()

    # 5. Update stats for logging
    train_stats.update({
        "loss/ptx": ptx_loss.detach(),
        "loss/total": total_loss.detach(),
    })

    return train_stats

  def _step_safety_checker(
      self,
      batch_size: int,
      queries: List[torch.LongTensor],
      responses: List[torch.LongTensor],
      scores: List[torch.FloatTensor],
  ):
      """
      Check if the input data is valid for training.

      Args:
          batch_size (int):
              Batch size from the config file.
          queries (List[`torch.LongTensor`]):
              List of tensors containing the encoded queries of shape (`query_length`)
          responses (List[`torch.LongTensor`]):
              List of tensors containing the encoded responses of shape (`response_length`)
          scores (List[`torch.FloatTensor`]):
              List of tensors containing the scores.
          masks (List[`torch.LongTensor`], *optional*):
              list of optional tensors containing the masks of shape (`response_length`)
      Returns:
          `tuple`: The input processed data.
      """
      for name, tensor_list in zip(["queries", "responses", "scores"], [queries, responses, scores]):
          if not isinstance(tensor_list, list):
              raise ValueError(f"{name} must be a list of tensors - got {type(tensor_list)}")
          if not isinstance(tensor_list[0], torch.Tensor):
              raise ValueError(f"Elements in {name} must be tensors - got {type(tensor_list[0])}")
          if batch_size is not None and len(tensor_list) != batch_size:
              raise ValueError(
                  f"Batch size ({batch_size}) does not match number of examples - but got {len(tensor_list)} for: {name}"
              )

      # add queries, scores and responses on the correct device
      queries = [tensor.to(self.current_device) for tensor in queries]
      responses = [tensor.to(self.current_device) for tensor in responses]
      scores = [tensor.to(self.current_device) for tensor in scores]

      # squeeze scores if needed
      for i, score in enumerate(scores):
          if score.dim() > 1:
              raise ValueError(f"Scores must be 1-dimensional - got {score.dim()} for {score}")
          elif score.dim() == 1:
              scores[i] = score.squeeze()

      return queries, responses, scores

  @PPODecorators.empty_device_cache()
  def step(
      self,
      queries: List[torch.LongTensor],
      responses: List[torch.LongTensor],
      scores: List[torch.FloatTensor],
      timing: dict,
      gen_data_dir: str,
      ptx_coef: float = 0.1,
      response_masks: Optional[List[torch.LongTensor]] = None,
      ptx_batch: dict = None,
  ):
      """
      Run a PPO optimisation step given a list of queries, model responses, and rewards.

      Args:
          queries (List[`torch.LongTensor`]):
              List of tensors containing the encoded queries of shape (`query_length`)
          responses (List[`torch.LongTensor`]):
              List of tensors containing the encoded responses of shape (`response_length`)
          scores (List[`torch.FloatTensor`]):
              List of tensors containing the scores.

      Returns:
          `dict[str, Any]`: A summary of the training statistics
      """
      bs = self.config.batch_size

      queries, responses, scores = self._step_safety_checker(bs, queries, responses, scores)

      # if we want to push best model to the hub
      if hasattr(self, "highest_reward"):
          if self.compare_step % self.config.compare_steps == 0:
              curr_mean_reward = torch.tensor(scores).mean()
              # if the best reward ever seen
              if curr_mean_reward > self.highest_reward:
                  self.highest_reward = curr_mean_reward
                  # push model to hub
                  self.push_to_hub(**self.push_to_hub_kwargs)
          self.compare_step += 1

      t0 = time.time()

      t = time.time()

      model_inputs = self.prepare_model_inputs(queries, responses)

      print(
          f"[Rank {self.accelerator.process_index}] "
          f"input_ids shape: {model_inputs['input_ids'].shape}, "
          f"dtype: {model_inputs['input_ids'].dtype}"
      )
      self.accelerator.wait_for_everyone()

      if self.is_distributed:
          print('in distriuted setting')
          pad_first = self.tokenizer.padding_side == "left"

          model_inputs["input_ids"] = self.accelerator.pad_across_processes(
              model_inputs["input_ids"],
              dim=1,
              pad_index=self.tokenizer.pad_token_id,
              pad_first=pad_first,
          )
          model_inputs["attention_mask"] = self.accelerator.pad_across_processes(
              model_inputs["attention_mask"], dim=1, pad_index=0, pad_first=pad_first
          )
          if self.is_encoder_decoder:
              model_inputs["decoder_input_ids"] = self.accelerator.pad_across_processes(
                  model_inputs["decoder_input_ids"],
                  dim=1,
                  pad_index=self.tokenizer.pad_token_id,
                  pad_first=pad_first,
              )
              model_inputs["decoder_attention_mask"] = self.accelerator.pad_across_processes(
                  model_inputs["decoder_attention_mask"],
                  dim=1,
                  pad_index=0,
                  pad_first=pad_first,
              )

      model_inputs_names = list(model_inputs.keys())

      full_kl_penalty = self.config.kl_penalty == "full"

      with torch.no_grad():
          all_logprobs, logits_or_none, values, masks = self.batched_forward_pass(
              self.model,
              queries,
              responses,
              model_inputs,
              response_masks=response_masks,
              return_logits=full_kl_penalty,
          )

          # for when the model is a peft model
          if self.is_peft_model and hasattr(
              self.accelerator.unwrap_model(self.model).pretrained_model,
              "disable_adapter",
          ):
              with self.accelerator.unwrap_model(self.model).pretrained_model.disable_adapter():
                  ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                      self.model, queries, responses, model_inputs, return_logits=full_kl_penalty,
                  )
          elif self.is_peft_model and not hasattr(self.model.pretrained_model, "disable_adapter"):
              raise ValueError(
                  "You are using a `peft` version that does not support `disable_adapter`. Please update your `peft` version to the latest version."
              )

          else:
              ref_logprobs, ref_logits_or_none, _, _ = self.batched_forward_pass(
                  self.ref_model, queries, responses, model_inputs, return_logits=full_kl_penalty,
              )

      timing["time/ppo/forward_pass"] = time.time() - t

      with torch.no_grad():
          t = time.time()
          if full_kl_penalty:
              active_full_logprobs = logprobs_from_logits(logits_or_none, None, gather=False)
              ref_full_logprobs = logprobs_from_logits(ref_logits_or_none, None, gather=False)

              rewards, non_score_reward, kls = self.compute_rewards(
                  scores, active_full_logprobs, ref_full_logprobs, masks
              )
          else:
              rewards, non_score_reward, kls = self.compute_rewards(scores, all_logprobs, ref_logprobs, masks)
          timing["time/ppo/compute_rewards"] = time.time() - t

          t = time.time()
          values, advantages, returns = self.compute_advantages(values, rewards, masks)
          timing["time/ppo/compute_advantages"] = time.time() - t

      # upcast to float32 to avoid dataset issues
      batch_dict = {
          "queries": queries,
          "responses": responses,
          "logprobs": all_logprobs.to(torch.float32),
          "values": values.to(torch.float32),
          "masks": masks,
          "advantages": advantages,
          "returns": returns,
      }
      batch_dict.update(model_inputs)

      t = time.time()
      all_stats = []
      early_stop = False
      for _ in range(self.config.ppo_epochs):
          if early_stop:
              break
          b_inds = np.random.permutation(bs)
          for backward_batch_start in range(0, bs, self.config.backward_batch_size):
              backward_batch_end = backward_batch_start + self.config.backward_batch_size
              backward_batch_inds = b_inds[backward_batch_start:backward_batch_end]

              for mini_batch_start in range(0, self.config.backward_batch_size, self.config.mini_batch_size):
                  mini_batch_end = mini_batch_start + self.config.mini_batch_size
                  mini_batch_inds = backward_batch_inds[mini_batch_start:mini_batch_end]
                  mini_batch_dict = {
                      "logprobs": batch_dict["logprobs"][mini_batch_inds],
                      "values": batch_dict["values"][mini_batch_inds],
                      "masks": batch_dict["masks"][mini_batch_inds],
                      # hacks: the queries and responses are ragged.
                      "queries": [batch_dict["queries"][i] for i in mini_batch_inds],
                      "responses": [batch_dict["responses"][i] for i in mini_batch_inds],
                      "advantages": batch_dict["advantages"][mini_batch_inds],
                      "returns": batch_dict["returns"][mini_batch_inds],
                  }
                  for k in model_inputs_names:
                      mini_batch_dict[k] = batch_dict[k][mini_batch_inds]
                  with self.accelerator.accumulate(self.model):
                      model_inputs = {k: mini_batch_dict[k] for k in model_inputs_names}

                      logprobs, logits, vpreds, _ = self.batched_forward_pass(
                          self.model,
                          mini_batch_dict["queries"],
                          mini_batch_dict["responses"],
                          model_inputs,
                          return_logits=True,
                      )
                      train_stats = self.train_minibatch(
                          mini_batch_dict["logprobs"],
                          mini_batch_dict["values"],
                          logprobs,
                          logits,
                          vpreds,
                          mini_batch_dict["masks"],
                          mini_batch_dict["advantages"],
                          mini_batch_dict["returns"],
                          ptx_batch=ptx_batch,
                          ptx_coef=ptx_coef,
                      )
                      all_stats.append(train_stats)

          # typically, early stopping is done at the epoch level
          if self.config.early_stopping:
              policykl = train_stats["policy/policykl"]
              early_stop = self._early_stop(policykl)
              if early_stop:
                  break

      timing["time/ppo/optimize_step"] = time.time() - t

      t = time.time()
      train_stats = stack_dicts(all_stats)

      # reshape advantages/ratios such that they are not averaged.
      train_stats["policy/advantages"] = torch.flatten(train_stats["policy/advantages"]).unsqueeze(0)
      train_stats["policy/advantages"] = torch.nan_to_num(train_stats["policy/advantages"], WANDB_PADDING)
      train_stats["policy/ratio"] = torch.flatten(train_stats["policy/ratio"]).unsqueeze(0)

      stats = self.record_step_stats(
          scores=scores,
          logprobs=all_logprobs,
          ref_logprobs=ref_logprobs,
          non_score_reward=non_score_reward,
          train_stats=train_stats,
          kl_coef=self.kl_ctl.value,
          masks=masks,
          queries=queries,
          responses=responses,
          kls=kls,
      )
      # Gather/Reduce stats from all processes
      if self.is_distributed:
          stats = self.gather_stats(stats)
      stats = stats_to_np(stats)
      timing["time/ppo/calc_stats"] = time.time() - t
      stats["ppo/learning_rate"] = self.optimizer.param_groups[0]["lr"]

      # Update the KL control - multiply the batch_size by the number of processes
      if self.accelerator.is_main_process:
          print('objective/kl', stats["objective/kl"])
          print('steps', self.config.batch_size * self.accelerator.num_processes)
      self.kl_ctl.update(
          stats["objective/kl"],
          self.config.batch_size * self.accelerator.num_processes,
      )

      # Log the total ppo time
      timing["time/ppo/total"] = time.time() - t0
      stats.update(timing)

      # post-process stats for tensorboard and other loggers
      if self.config.log_with != "wandb":
          stats = convert_to_scalar(stats)

      if self.lr_scheduler is not None:
          self.lr_scheduler.step()

      return stats


  def record_step_stats(self, kl_coef: float, **data):
      """
      Record training step statistics.


      Args:
          kl_coef (`float`):
              KL coefficient
          data (`dict`):
              Dictionary of training step data

      Returns:
          stats (`dict`):
              Dictionary of training step statistics
      """
      mask = data.pop("masks")

      kls = data.pop("kls")
      kl_list = ((kls) * mask).sum(axis=-1)
      mean_kl = kl_list.mean()
      mean_entropy = (-data["logprobs"] * mask).sum(axis=-1).mean()

      mean_non_score_reward = masked_mean(
          data["non_score_reward"], mask
      )  # non_score_reward is size `batch_size`, `response_length`
      mean_scores = torch.tensor(data["scores"]).mean()  # scores is size `batch_size`
      std_scores = torch.tensor(data["scores"]).std()

      if mean_kl.item() < -1.0:
          # warn users
          warnings.warn(
              f"KL divergence is starting to become negative: {mean_kl.item():.2f} - this might be a precursor for failed training."
              " sometimes this happens because the generation kwargs are not correctly set. Please make sure"
              " that the generation kwargs are set correctly, or review your training hyperparameters."
          )

      stats = {
          "objective/kl": mean_kl,
          "objective/kl_dist": kl_list,
          "objective/logprobs": data["logprobs"],
          "objective/ref_logprobs": data["ref_logprobs"],
          "objective/kl_coef": kl_coef,
          "objective/entropy": mean_entropy,
          "ppo/mean_non_score_reward": mean_non_score_reward,
          "ppo/mean_scores": mean_scores,
          "ppo/std_scores": std_scores,
      }

      # Log text properties
      query_lens = torch.tensor([len(query) for query in data["queries"]], dtype=torch.float)
      response_lens = torch.tensor([len(response) for response in data["responses"]], dtype=torch.float)

      stats["tokens/queries_len_mean"] = torch.mean(query_lens).cpu().numpy().item()
      stats["tokens/queries_len_std"] = torch.std(query_lens).cpu().numpy().item()
      stats["tokens/queries_dist"] = query_lens.cpu().numpy()
      stats["tokens/responses_len_mean"] = torch.mean(response_lens).cpu().numpy().item()
      stats["tokens/responses_len_std"] = torch.std(response_lens).cpu().numpy().item()
      stats["tokens/responses_dist"] = response_lens.cpu().numpy()

      for k, v in data["train_stats"].items():
          stats[f"ppo/{k}"] = torch.mean(v, axis=0)
      stats["ppo/val/var_explained"] = 1 - stats["ppo/val/error"] / stats["ppo/returns/var"]
      return stats

