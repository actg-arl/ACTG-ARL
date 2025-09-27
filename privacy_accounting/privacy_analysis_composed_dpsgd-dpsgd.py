"""
Finds optimal noise multiplier pairs for a two-stage differential privacy
mechanism, composed of two stages of DP-SGD, to achieve a target epsilon.

This script uses the prv_accountant library for privacy accounting.
"""

# --- Standard Library Imports ---
import argparse
import itertools
import logging
import os
import warnings

# --- Third-Party Library Imports ---
import numpy as np
import pandas as pd
from prv_accountant import Accountant, PRVAccountant
from prv_accountant.privacy_random_variables import PoissonSubsampledGaussianMechanism
from scipy import optimize

# --- Configuration ---
warnings.filterwarnings("ignore", message="overflow encountered in exp")
NOISE_MULTIPLIER_MAX = 400.0

# --- Setup Basic Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# --- Helper & Core DP Functions ---
def get_epsilon_single_stage(params, N):
  """Computes epsilon for a single-stage DP mechanism."""
  noise_multiplier, sampling_probability, num_steps = params
  target_delta = 1 / (N * np.log(N))
  if noise_multiplier <= 0 or num_steps == 0:
    return float("inf")

  prv = PoissonSubsampledGaussianMechanism(
      noise_multiplier=noise_multiplier,
      sampling_probability=sampling_probability,
  )
  accountant = PRVAccountant(
      prvs=[prv],
      max_self_compositions=[num_steps],
      eps_error=0.01,
      delta_error=target_delta / 10,
  )
  try:
    _, _, eps_upper_bound = accountant.compute_epsilon(
        delta=target_delta, num_self_compositions=[num_steps]
    )
    return eps_upper_bound
  except (ValueError, RuntimeError):
    return float("inf")


def get_epsilon_composed(params1, params2, N, precision_error=0.005):
  """Computes total epsilon for a two-stage composed DP mechanism."""
  nm1, sp1, iters1 = params1
  nm2, sp2, iters2 = params2
  target_delta = 1 / (N * np.log(N))
  if nm1 <= 0 or nm2 <= 0 or iters1 == 0 or iters2 == 0:
    return float("inf")

  prv1 = PoissonSubsampledGaussianMechanism(
      noise_multiplier=nm1, sampling_probability=sp1
  )
  prv2 = PoissonSubsampledGaussianMechanism(
      noise_multiplier=nm2, sampling_probability=sp2
  )

  accountant = PRVAccountant(
      prvs=[prv1, prv2],
      max_self_compositions=[iters1, iters2],
      eps_error=precision_error,
      delta_error=target_delta / 10,
  )
  try:
    _, _, total_eps = accountant.compute_epsilon(
        delta=target_delta, num_self_compositions=[iters1, iters2]
    )
    return total_eps
  except (ValueError, RuntimeError):
    return float("inf")


def fast_find_noise_for_stage1(
    sampling_probability, num_steps, target_epsilon, N
):
  """Finds the noise multiplier for a single stage to meet a target epsilon."""
  target_delta = 1 / (N * np.log(N))

  def compute_epsilon(mu: float):
    return Accountant(
        noise_multiplier=mu,
        sampling_probability=sampling_probability,
        delta=target_delta,
        max_compositions=num_steps,
        eps_error=0.05,
    ).compute_epsilon(num_steps)

  mu_R = 1.0
  eps_R = float("inf")
  while eps_R > target_epsilon:
    mu_R *= np.sqrt(2)
    if mu_R > NOISE_MULTIPLIER_MAX:
      raise RuntimeError("Noise search did not converge (Stage 1 Upper).")
    try:
      eps_R = compute_epsilon(mu_R)[2]
    except (OverflowError, RuntimeError):
      pass

  mu_L = mu_R
  eps_L = eps_R
  while eps_L < target_epsilon:
    mu_L /= np.sqrt(2)
    if mu_L < 1e-3:
      raise RuntimeError("Noise search did not converge (Stage 1 Lower).")
    eps_L = compute_epsilon(mu_L)[0]

  try:
    return optimize.root_scalar(
        lambda mu: compute_epsilon(mu)[2] - target_epsilon, bracket=[mu_L, mu_R]
    ).root
  except ValueError:
    return np.nan


def fast_find_noise_for_composed(params1, sp2, iters2, total_target_epsilon, N):
  """Finds the noise for stage 2, given the fixed parameters of stage 1."""

  def objective(mu2: float):
    return (
        get_epsilon_composed(params1, (mu2, sp2, iters2), N)
        - total_target_epsilon
    )

  if get_epsilon_single_stage(params1, N) > total_target_epsilon:
    return np.nan

  mu_R = 1.0
  while objective(mu_R) > 0:
    mu_R *= np.sqrt(2)
    if mu_R > NOISE_MULTIPLIER_MAX:
      return np.nan

  mu_L = mu_R
  while objective(mu_L) < 0:
    mu_L /= np.sqrt(2)
    if mu_L < 1e-3:
      mu_L = 0.0
      break

  if mu_L == 0.0 and objective(mu_R) > 0:
    return np.nan

  try:
    return optimize.root_scalar(objective, bracket=[mu_L, mu_R]).root
  except ValueError:
    return np.nan


def main():
  # --- Argument Parsing ---
  parser = argparse.ArgumentParser(
      description="Find noise pairs for a two-stage DP process."
  )
  parser.add_argument(
      "--N", type=int, required=True, help="Total number of samples."
  )
  parser.add_argument(
      "--eps", type=float, required=True, help="Total target epsilon."
  )
  parser.add_argument(
      "--bs1",
      type=int,
      nargs="+",
      required=True,
      help="List of batch sizes for stage 1.",
  )
  parser.add_argument(
      "--T1",
      type=int,
      nargs="+",
      required=True,
      help="List of iterations for stage 1.",
  )
  parser.add_argument(
      "--eps1_targets",
      type=float,
      nargs="+",
      required=True,
      help="List of target epsilons for stage 1.",
  )
  parser.add_argument(
      "--bs2",
      type=int,
      nargs="+",
      required=True,
      help="List of batch sizes for stage 2.",
  )
  parser.add_argument(
      "--T2",
      type=int,
      nargs="+",
      required=True,
      help="List of iterations for stage 2.",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      default="./output",
      help="Directory to save the output CSV file.",
  )

  args = parser.parse_args()

  # --- Iterate Over Parameter Combinations and Process ---
  s1_params = list(itertools.product(args.bs1, args.T1, args.eps1_targets))
  s2_params = list(itertools.product(args.bs2, args.T2))

  results = []
  logging.info(f"Starting search for N={args.N}, Total Epsilon={args.eps}")
  logging.info("-" * 50)

  for i, (bs1, T1, eps1_target) in enumerate(s1_params):
    if eps1_target >= args.eps:
      continue
    sp1 = bs1 / args.N

    log_prefix = (
        f"[S1 {i+1}/{len(s1_params)} | bs={bs1}, T={T1},"
        f" eps_target={eps1_target:.1f}]"
    )
    logging.info(log_prefix)

    try:
      nm1 = fast_find_noise_for_stage1(sp1, T1, eps1_target, args.N)
    except RuntimeError as e:
      nm1 = np.nan

    if np.isnan(nm1):
      logging.warning(f"  -> Could not find nm1. Skipping.")
      continue

    params1 = (nm1, sp1, T1)
    eps1_actual = get_epsilon_single_stage(params1, args.N)
    logging.info(
        f"  -> Found nm1={nm1:.2f} (actual eps1={eps1_actual:.2f}). Searching"
        " for matching Stage 2...",
    )

    for bs2, T2 in s2_params:
      sp2 = bs2 / args.N
      nm2 = fast_find_noise_for_composed(params1, sp2, T2, args.eps, args.N)

      if not np.isnan(nm2):
        logging.info(
            f"  -> SUCCESS: Found nm2={nm2:.2f} for S2(bs={bs2}, T={T2})"
        )
        params2 = (nm2, sp2, T2)
        eps2_individual = get_epsilon_single_stage(params2, args.N)
        total_eps = get_epsilon_composed(params1, params2, args.N)

        if abs(total_eps - args.eps) < 0.05:
          results.append({
              "bs1": bs1,
              "T1": T1,
              "nm1": round(nm1, 2),
              "eps1_ind": round(eps1_actual, 2),
              "bs2": bs2,
              "T2": T2,
              "nm2": round(nm2, 2),
              "eps2_ind": round(eps2_individual, 2),
              "total_eps": round(total_eps, 3),
          })

  # --- Output and Save Results ---
  logging.info("=" * 50)
  df = pd.DataFrame(results)
  if not df.empty:
    df = df.sort_values(by=["eps1_ind", "T1", "T2"]).reset_index(drop=True)
    logging.info("Found the following training schemes:")
    logging.info('\n'+df.to_string())

    os.makedirs(args.output_dir, exist_ok=True)
    eps1_targets_str = "+".join([str(x) for x in args.eps1_targets])
    output_filename = f"two_stage_schemes_N-{args.N}_eps-{args.eps}-eps1-{eps1_targets_str}.csv"
    output_path = os.path.join(args.output_dir, output_filename)

    df.to_csv(output_path, index=False)
    logging.info(f"Results saved to {output_path}")
  else:
    logging.warning(
        "No valid training schemes were found with the given parameters."
    )


if __name__ == "__main__":
  main()
