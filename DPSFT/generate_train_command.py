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

"""Script to generate commands."""

import argparse
import os

from utils import dp_utils

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dataset_name', type=str, default='chatbot_arena_instructions_train180k'
)
parser.add_argument(
    '--dataset_path',
    type=str,
    default='',
)
parser.add_argument('--model_name', type=str, default='yahma/llama-7b-hf')
parser.add_argument(
    '--job_sess',
    type=str,
    default='debug',
    help='session of this job, will be prefix of sess',
)
parser.add_argument(
    '--perdevice_bs', type=int, default=4, help='number of batchsize per gpu'
)
parser.add_argument('--gpus', type=int, default=1, help='number of gpus to use')
parser.add_argument(
    '--max_seq_len', type=int, default=2048, help='max sequence length'
)
parser.add_argument(
    '--max_instruction_len',
    type=int,
    default=128,
    help='max instruction length',
)
parser.add_argument(
    '--max_answer_len', type=int, default=384, help='max answer length'
)
parser.add_argument('--total_bs', type=int, default=32, help='total batchsize')
parser.add_argument(
    '--num_epochs', type=int, default=3, help='number of epochs'
)
parser.add_argument(
    '--num_steps', type=int, default=-1, help='number of epochs'
)
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument(
    '--lr_scheduler',
    type=str,
    default='constant',
    help='learning rate scheduler',
)
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument(
    '--clip', type=float, default=-1, help='per-example gradient clip norm'
)
parser.add_argument(
    '--eps',
    type=float,
    default=-1,
    help='target epsilon value for (eps,delta)-DP',
)
parser.add_argument(
    '--delta',
    type=float,
    default=5e-7,
    help='target delta value for (eps,delta)-DP',
)
parser.add_argument(
    '--prompt_style',
    type=str,
    default=None,
    help=(
        'style of the prompt, could be vicuna style (vicuna) or empty prompt'
        ' for unconditional generation (uncond_generation)'
    ),
)
parser.add_argument(
    '--no_eval_at_start',
    action='store_true',
    help='whether not to run evaluation at epoch 0',
)
parser.add_argument(
    '--eval_only',
    action='store_true',
)
parser.add_argument('--next_token_prediction_acc', action='store_true')
parser.add_argument(
    '--main_process_port',
    type=int,
    help='main process port for accelerate launch',
    default=29500,
)
parser.add_argument(
    '--seed',
    type=int,
    help='seed for random number generator',
    default=0,
)
parser.add_argument(
    '--noise_multiplier',
    type=float,
    default=0.0,
)
parser.add_argument(
    '--sess',
    type=str,
    default='',
)

args = parser.parse_args()


if args.dataset_name == 'movie-json':
  NUM_TRAIN_SAMPLES = 26963
elif args.dataset_name == 'pubmed_openreview_mixture':
  NUM_TRAIN_SAMPLES = 76069
elif args.dataset_name in [
    'enron',
    'enron-eval',
    'enron-generated',
    'enron-eval-generated',
    'enron_lenfilt_200-850',
    'enron_lenskew_5',
    'enron_sampled_5000',
    'enron-freeform-conditions',
    'enron-condgen-freeform',
    'enron-freeform-e2e',
    'enron-schema-conditions',
    'enron-condgen-schema',
    'enron-condgen-topic',
]:
  NUM_TRAIN_SAMPLES = 33508
elif args.dataset_name in ['PMC', 'PMC-conditions', 'PMC-condgen']:
  NUM_TRAIN_SAMPLES = 25000
elif args.dataset_name in [
    'biorxiv',
    'biorxiv-conditions',
    'biorxiv-condgen',
    'biorxiv-category-conditions',
    'biorxiv-category-condgen',
    'biorxiv-nl3-condgen',
    'biorxiv-complex8-conditions',
    'biorxiv-complex8-condgen',
    'biorxiv-complex8et-conditions',
    'biorxiv-complex8et-condgen',
]:
  NUM_TRAIN_SAMPLES = 28846
elif args.dataset_name in [
    'biorxiv-generated',
]:
  NUM_TRAIN_SAMPLES = 5000
else:
  raise NotImplementedError(f'Unknown dataset: {args.dataset_name}')

prefix = '/home/user/results/'

os.makedirs(prefix + 'logs', exist_ok=True)

if args.gpus == 1:
  accelerate_cfg_file = 'accelerate_configs/accelerate_config_nofsdp.cfg'
else:
  accelerate_cfg_file = (
      f'accelerate_configs/accelerate_config_nofsdp_gpu{args.gpus}.cfg'
  )

dataset_name = args.dataset_name
dataset_path = args.dataset_path
model_name = args.model_name
perdevice_bs = args.perdevice_bs
max_seq_len = f'{args.max_instruction_len}-{args.max_answer_len}'
total_bs = args.total_bs
accumulation_steps = max(1, total_bs // (perdevice_bs * args.gpus))
assert total_bs % perdevice_bs == 0
num_epochs = args.num_epochs
num_steps = args.num_steps
lr = args.lr
wd = args.wd
clip = args.clip
eps = args.eps
delta = args.delta
np = args.noise_multiplier
gpus = args.gpus
job_sess = args.job_sess
prompt_style = args.prompt_style
lr_scheduler = args.lr_scheduler

if num_steps > 0:
  steps = num_steps
else:
  steps = int(num_epochs / (total_bs / NUM_TRAIN_SAMPLES))

if np != 0:
  print(
      f'Noise multiplier is {np} for {eps, delta}-DP. Config:'
      f' batchsize={total_bs}, dataset size={NUM_TRAIN_SAMPLES},'
      f' num_steps={num_steps}.'
  )
elif eps > 0:
  print('eps, delta:', eps, delta)
  print('steps:', steps)
  print('total_bs, num_train_samples:', total_bs, NUM_TRAIN_SAMPLES)
  np = dp_utils.get_noise_multiplier(
      eps, delta, steps, total_bs / NUM_TRAIN_SAMPLES
  )
  print(
      f'Noise multiplier is {np} for ({eps,delta})-DP. Config:'
      f' batchsize={total_bs}, dataset size={NUM_TRAIN_SAMPLES},'
      f' num_steps={num_steps}.'
  )
else:
  np = -1

model_name_in_file = model_name.split('/')[-1]

prompt_style_dict = {
    # --- movie json dataset ---
    'movie-json': 'moviejson-gen',
    # --- mixture of pubmed and openreview ---
    'pubmed_openreview_mixture': 'pubmed_openreview_mixture_generation',
    # --- enron dataset ---
    'enron': 'enron_generation',
    'enron_lenfilt_200-850': 'enron_generation',
    'enron_lenskew_5': 'enron_generation',
    'enron_sampled_5000': 'enron_generation',
    'enron-freeform-conditions': 'enron-freeform-conditions_generation',
    'enron-condgen-freeform': 'enron-condgen-freeform_generation',
    'enron-freeform-e2e': 'enron-freeform-e2e_generation',
    'enron-schema-conditions': 'enron-schema-conditions_generation',
    'enron-condgen-schema': 'enron-condgen-schema_generation',
    'enron-generated': 'enron_evaluation',
    'enron-condgen-topic': 'enron-condgen-topic_generation',
    # --- PMC dataset ---
    'PMC': 'PMC_generation',
    'PMC-conditions': 'PMC-conditions_generation',
    'PMC-condgen': 'PMC-condgen_generation',
    # --- biorxiv dataset ---
    'biorxiv': 'biorxiv_generation',
    'biorxiv-conditions': 'biorxiv-conditions_generation',
    'biorxiv-category-conditions': 'biorxiv-category-conditions_generation',
    'biorxiv-category-condgen': 'biorxiv-category-condgen_generation',
    'biorxiv-condgen': 'biorxiv-condgen_generation',
    'biorxiv-nl3-condgen': 'biorxiv-nl3-condgen_generation',
    'biorxiv-complex8-conditions': 'biorxiv-complex8-conditions_generation',
    'biorxiv-complex8-condgen': 'biorxiv-condgen_generation',
    'biorxiv-complex8et-conditions': 'biorxiv-complex8et-conditions_generation',
    'biorxiv-complex8et-condgen': 'biorxiv-condgen_generation',
    'biorxiv-generated': 'biorxiv_evaluation',
}

if dataset_name in prompt_style_dict:
  assert prompt_style == prompt_style_dict[dataset_name], (
      f'Prompt style {prompt_style} does not match expected for dataset'
      f' {dataset_name}: {prompt_style_dict[dataset_name]}'
  )
  sess = '{}_noredacted_model{}_eps{}_delta{}_bs{}_maxseq{}_epoch{}_lr{}_clip{}_np{}_gpus{}'.format(
      dataset_name,
      model_name_in_file,
      eps,
      delta,
      total_bs,
      max_seq_len,
      num_epochs,
      lr,
      clip,
      np,
      gpus,
  )

else:
  raise NotImplementedError(f'Unknown dataset {dataset_name}')

sess = job_sess + '_' + sess

prepend = ''
hf_login_str = os.environ.get('HF_LOGIN_STR', '')
if not hf_login_str:
  raise ValueError(
      'Please add your huggingface login token to the environment variable'
      ' or generate_train_command.py'
  )

add_str = ''
if np > 0 or args.no_eval_at_start:
  add_str = '--no_eval_at_start'
if args.eval_only:
  add_str += ' --eval_only'
if args.next_token_prediction_acc:
  add_str += ' --next_token_prediction_acc'
if args.dataset_path:
  add_str += f' --dataset_path {dataset_path}'

command = f"""{prepend} accelerate launch \\
    --main_process_port {args.main_process_port} \\
    --config_file {accelerate_cfg_file} \\
    ./train_clm.py \\
    --dataset_name {dataset_name} \\
    --model_name_or_path {model_name} \\
    --output_dir {prefix}/outputs/{sess} \\
    --seed {args.seed} \\
    --per_device_train_batch_size {perdevice_bs} \\
    --gradient_accumulation_steps {accumulation_steps} \\
    --learning_rate {lr} \\
    --lr_scheduler_type {lr_scheduler} \\
    --num_train_epochs {num_epochs} \\
    --num_train_steps {num_steps} \\
    --num_warmup_steps 30 \\
    --max_instruction_length {args.max_instruction_len} \\
    --max_answer_length {args.max_answer_len} \\
    --log_freq 5 \\
    --gradient_ckpt \\
    --clip_norm {clip} \\
    --weight_decay {wd} \\
    --delta {delta} \\
    --noise_multiplier {np} \\
    --prompt_style {prompt_style} \\
    --access_token {hf_login_str} \\
    --qbits 16 \\
    {add_str} \\
    2>&1 | tee -a {prefix}/logs/{sess}.txt
"""

os.system(command)
