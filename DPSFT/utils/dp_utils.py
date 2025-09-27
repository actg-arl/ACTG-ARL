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

"""Utils for DP."""

from prv_accountant import PRVAccountant
from prv_accountant.privacy_random_variables import PoissonSubsampledGaussianMechanism
import torch


def linear_forward_hook(module, intsr, outtsr):  # pylint: disable=unused-argument
  module.input = intsr[0]


# pylint: disable=invalid-name
def linear_backward_hook(layer, grad_input, grad_output):  # pylint: disable=unused-argument
  """Backward hook for network layer."""
  grad_output = grad_output[0]  # n, len, outdim
  grad_input = layer.input  # n, len, indim

  layer_batch_dim = 0

  A = grad_input
  B = grad_output
  if A.dtype != B.dtype:
    # during forward, some activates are casted to fp32 for stability.
    # Convert them back for gradient computation
    A = A.to(B.dtype)

  # Compute per-sequence gradients
  # The gradients of tokens in the same sequence are summed up
  # k: tokens-per-sample
  # n: batch size
  if layer_batch_dim == 1:
    gs = torch.einsum('kn...i,kn...j->nij', B, A)
    if layer.bias is not None:
      gs_bias = torch.einsum('kn...i->ni', B)
  else:
    gs = torch.einsum('n...i,n...j->nij', B, A)
    if layer.bias is not None:
      gs_bias = torch.einsum('n...k->nk', B)

  layer.weight.grad_sample = gs.float()
  if layer.bias is not None:
    layer.bias.grad_sample = gs_bias.float()


def make_lora_model_dp(model):
  # register forward and backward hooks for lora branch
  for module in model.modules():
    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
      module.lora_A['default'].register_forward_hook(linear_forward_hook)
      module.lora_A['default'].register_backward_hook(linear_backward_hook)
      module.lora_B['default'].register_forward_hook(linear_forward_hook)
      module.lora_B['default'].register_backward_hook(linear_backward_hook)


def get_grad_norm(params):
  """Get the gradient norm of each example."""
  # params: all trainable parameters
  # when lora is enabled, the params only contain lora parameters
  for p in params:
    if hasattr(p, 'grad_sample'):
      # n is the batch size
      n = p.grad_sample.shape[0]
      break
  grad_norm_list = torch.zeros(n).cuda()
  for p in params:
    if hasattr(p, 'grad_sample'):
      flat_g = p.grad_sample.reshape(n, -1)
      current_norm_list = torch.norm(flat_g, dim=1)
      grad_norm_list += torch.square(current_norm_list)
    else:
      raise ValueError('DP enabled but no grad_sample found')
  grad_norm_list = torch.sqrt(grad_norm_list)

  return grad_norm_list


def clip_grad_sample(params, clipping):
  """Clip the gradient of each example."""
  for p in params:
    if hasattr(p, 'grad_sample'):
      n = p.grad_sample.shape[0]
      break
  grad_norm_list = torch.zeros(n).cuda()
  for p in params:
    if hasattr(p, 'grad_sample'):
      flat_g = p.grad_sample.reshape(n, -1)
      current_norm_list = torch.norm(flat_g, dim=1)
      grad_norm_list += torch.square(current_norm_list)
  grad_norm_list = torch.sqrt(grad_norm_list)
  scaling = clipping / grad_norm_list
  scaling[scaling > 1] = 1

  for p in params:
    if hasattr(p, 'grad_sample'):
      p_dim = len(p.shape)
      scaling = scaling.view([n] + [1] * p_dim)
      p.grad_sample *= scaling

  return grad_norm_list


def get_epsilon_prv(noise_multiplier, delta, steps, sampling_prob):
  """Get the epsilon for running dp-sgd."""
  prv = PoissonSubsampledGaussianMechanism(
      noise_multiplier=noise_multiplier, sampling_probability=sampling_prob
  )
  accountant = PRVAccountant(
      prvs=[prv],
      max_self_compositions=[steps],
      eps_error=0.1,
      delta_error=delta / 10,
  )
  _, _, eps_up = accountant.compute_epsilon(
      delta=delta, num_self_compositions=[steps]
  )
  return eps_up


def search_for_sigma(
    current_sigma, eps, delta, steps, sampling_prob, precision
):
  """Search for the sigma that gives the closest epsilon to the target."""
  while current_sigma > 0:
    current_eps = get_epsilon_prv(current_sigma, delta, steps, sampling_prob)
    if current_eps < eps:
      current_sigma -= precision
    else:
      current_sigma += precision
      return current_sigma
  return precision


def get_noise_multiplier_archive(eps, delta, steps, sampling_prob, init_sigma=25):
  """Get the noise multiplier for running dp-sgd."""
  try:
    current_sigma = init_sigma
    current_sigma = search_for_sigma(
        current_sigma, eps, delta, steps, sampling_prob, precision=1
    )
    current_sigma = search_for_sigma(
        current_sigma, eps, delta, steps, sampling_prob, precision=0.1
    )
    current_sigma = search_for_sigma(
        current_sigma, eps, delta, steps, sampling_prob, precision=0.01
    )

    if current_sigma == 0.01:
      raise ValueError(
          'Cannot find a valid sigma for the given epsilon and delta.'
      )
  except Exception as e:
    print('error using prv accountant, falling back to pld accountant')
    import dp_accounting
    def calculate_sigma_sgd(epsilon, delta, ratio, iters, orders=None):
        def make_subsampling_gaussian_event(z: float):
            return dp_accounting.SelfComposedDpEvent(
                event=dp_accounting.PoissonSampledDpEvent(
                    ratio, 
                    event=dp_accounting.GaussianDpEvent(noise_multiplier=z)),
                count=int(iters))

        sigma = dp_accounting.calibrate_dp_mechanism(
            dp_accounting.pld.PLDAccountant,
            make_subsampling_gaussian_event,
            epsilon, delta, dp_accounting.LowerEndpointAndGuess(0, 1))
        return sigma
    current_sigma = calculate_sigma_sgd(
      eps, delta, sampling_prob, steps
    )
  return current_sigma

def get_noise_multiplier(eps, delta, steps, sampling_prob, init_sigma=25):
  """Get the noise multiplier for running dp-sgd."""
  if eps >= 100:
    print('fall into the mode of using noise_multiplier directly')
    return 0
    # return 1 / eps
    
  try:
    print('try using prv accountant...')
    current_sigma = find_noise_multiplier(sampling_prob, steps, eps, delta, eps_error=0.01)
    print('prv accountant worked! - current_sigma:', current_sigma)
  except Exception as e:
    print('error using prv accountant, falling back to pld accountant')
    import dp_accounting
    def calculate_sigma_sgd(epsilon, delta, ratio, iters, orders=None):
        def make_subsampling_gaussian_event(z: float):
            return dp_accounting.SelfComposedDpEvent(
                event=dp_accounting.PoissonSampledDpEvent(
                    ratio, 
                    event=dp_accounting.GaussianDpEvent(noise_multiplier=z)),
                count=int(iters))

        sigma = dp_accounting.calibrate_dp_mechanism(
            dp_accounting.pld.PLDAccountant,
            make_subsampling_gaussian_event,
            epsilon, delta, dp_accounting.LowerEndpointAndGuess(0, 1))
        return sigma
    current_sigma = calculate_sigma_sgd(
      eps, delta, sampling_prob, steps
    )
  return current_sigma



def find_noise_multiplier(sampling_probability: float, num_steps: int, target_epsilon: float, target_delta: float,
                          eps_error: float=0.1) -> float:
    """
    Find a noise multiplier that satisfies a given target epsilon.

    :param float sampling_probability: Probability of a record being in batch for Poisson sampling
    :param int num_steps: Number of optimisation steps
    :param float target_epsilon: Desired target epsilon
    :param float target_delta: Value of DP delta
    :param float eps_error: Error allowed for final epsilon
    """
    from prv_accountant import Accountant
    from scipy import optimize
    def compute_epsilon(mu: float) -> float:
        acc = Accountant(
            noise_multiplier=mu,
            sampling_probability=sampling_probability,
            delta=target_delta,
            max_compositions=num_steps,
            eps_error=eps_error/2
        )
        return acc.compute_epsilon(num_steps)

    mu_max = 100.0

    mu_R = 1.0
    eps_R = float('inf')
    while eps_R > target_epsilon:
        import numpy as np
        mu_R *= np.sqrt(2)
        try:
            eps_R = compute_epsilon(mu_R)[2]
        except (OverflowError, RuntimeError):
            pass
        if mu_R > mu_max:
            raise RuntimeError("Finding a suitable noise multiplier has not converged. "
                               "Try increasing target epsilon or decreasing sampling probability.")

    mu_L = mu_R
    eps_L = eps_R
    while eps_L < target_epsilon:
        mu_L /= np.sqrt(2)
        eps_L = compute_epsilon(mu_L)[0]

    has_converged = False 
    bracket = [mu_L, mu_R]
    while not has_converged:
        mu_err = (bracket[1]-bracket[0])*0.01
        mu_guess = optimize.root_scalar(lambda mu: compute_epsilon(mu)[2]-target_epsilon, bracket=bracket, xtol=mu_err).root
        bracket = [mu_guess-mu_err, mu_guess+mu_err]
        eps_up = compute_epsilon(mu_guess-mu_err)[2]
        eps_low = compute_epsilon(mu_guess+mu_err)[0]
        has_converged = (eps_up - eps_low) < 2*eps_error
    assert compute_epsilon(bracket[1])[2] < target_epsilon + eps_error

    return bracket[1]


# eps = 5.94
# delta = 5e-7
# steps = int(10/(4096/180000))
# sampling_prob = 4096/180000


def clip_and_accumulate_perexample_grads(
    require_grad_params, accumulated_steps, clip_norm, accelerator
):
  """Clip and accumulate per-example gradients."""
  if accelerator.scaler is not None:
    # get the scale of mixed precision training
    mixed_precision_scale = accelerator.scaler.get_scale()
  else:
    mixed_precision_scale = 1.0
  for p in require_grad_params:
    if hasattr(p, 'grad_sample'):
      # convert to fp32
      p.grad_sample = p.grad_sample.float()
      # undo mixed precision scaling
      p.grad_sample /= mixed_precision_scale
    else:
      raise RuntimeError('DP enabled but no grad_sample found')

  # clip gradients
  grad_norms = clip_grad_sample(require_grad_params, clip_norm)

  # accumulate gradients
  for p in require_grad_params:
    if hasattr(p, 'grad_sample'):
      if accumulated_steps == 0:
        p.accumulated_grad = torch.sum(p.grad_sample, dim=0)
      else:
        p.accumulated_grad += torch.sum(p.grad_sample, dim=0)
      p.grad_sample = None

  return grad_norms
