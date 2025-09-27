"""Finds the optimal noise multiplier (sigma) for a single Gaussian mechanism (Stage 1)

when composed with a multi-step subsampled Gaussian mechanism (Stage 2)
to meet a total (epsilon, delta) budget.

This script uses the dp-accounting library for privacy accounting.
"""

# --- Standard Library Imports ---
import argparse
import logging
import sys
from typing import Tuple

# --- Third-Party Library Imports ---
from dp_accounting import dp_event
from dp_accounting.rdp import rdp_privacy_accountant

# --- Setup Basic Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


def find_sigma_for_first_stage(
    total_epsilon: float,
    total_delta: float,
    subsampling_rate_stage2: float,
    iterations_stage2: int,
    sigma_stage2: float,
    sigma_search_min: float = 1e-5,
    sigma_search_max: float = 1000.0,
    search_precision: int = 50,
) -> float:
  """Finds the required sigma for Stage 1 via binary search.

  Args:
      total_epsilon: The target epsilon for the combined two-stage process.
      total_delta: The target delta for the combined two-stage process.
      subsampling_rate_stage2: The sampling rate (q) for the second stage.
      iterations_stage2: The number of composition steps for the second stage.
      sigma_stage2: The noise multiplier (sigma) for the second stage.
      sigma_search_min: The lower bound for the binary search on sigma.
      sigma_search_max: The upper bound for the binary search on sigma.
      search_precision: The number of iterations for the binary search.

  Returns:
      The calculated sigma for the first stage mechanism.
  """
  # Define the event for the second, known stage
  subsampled_event_stage2 = dp_event.PoissonSampledDpEvent(
      sampling_probability=subsampling_rate_stage2,
      event=dp_event.GaussianDpEvent(noise_multiplier=sigma_stage2),
  )

  # Objective function for the binary search
  def get_total_epsilon(sigma_stage1: float) -> float:
    """Calculates the total epsilon for a given sigma_stage1."""
    accountant = rdp_privacy_accountant.RdpAccountant()

    # Stage 1: A single Gaussian mechanism application
    event_stage1 = dp_event.GaussianDpEvent(noise_multiplier=sigma_stage1)
    accountant.compose(event_stage1)

    # Stage 2: The subsampled Gaussian mechanism composed multiple times
    accountant.compose(subsampled_event_stage2, count=iterations_stage2)

    return accountant.get_epsilon(target_delta=total_delta)

  # Binary search for sigma_stage1
  low = sigma_search_min
  high = sigma_search_max

  for _ in range(search_precision):
    mid_sigma = (low + high) / 2
    try:
      epsilon_guess = get_total_epsilon(mid_sigma)
    except (OverflowError, ValueError):
      # If accountant fails (e.g., due to very low sigma), treat as infinite epsilon
      epsilon_guess = float("inf")

    if epsilon_guess > total_epsilon:
      low = mid_sigma
    else:
      high = mid_sigma

  # 'high' is the lowest sigma that meets the budget
  return high


def main():
  """Main function to parse arguments and run the calculation."""
  parser = argparse.ArgumentParser(
      description="Calculate Stage 1 sigma for a two-stage DP mechanism."
  )
  # --- Overall Budget ---
  parser.add_argument(
      "--total_epsilon",
      type=float,
      required=True,
      help="Target epsilon for the combined process.",
  )
  parser.add_argument(
      "--total_delta",
      type=float,
      required=True,
      help="Target delta for the combined process.",
  )
  # --- Stage 2 Parameters ---
  parser.add_argument(
      "--dataset_size_s2",
      type=int,
      required=True,
      help="Dataset size for Stage 2.",
  )
  parser.add_argument(
      "--batch_size_s2", type=int, required=True, help="Batch size for Stage 2."
  )
  parser.add_argument(
      "--iterations_s2",
      type=int,
      required=True,
      help="Number of iterations for Stage 2.",
  )
  parser.add_argument(
      "--sigma_s2",
      type=float,
      required=True,
      help="Noise multiplier (sigma) for Stage 2.",
  )
  args = parser.parse_args()

  sampling_rate_s2 = args.batch_size_s2 / args.dataset_size_s2

  logging.info("Target Budget:")
  logging.info(f"  - Epsilon: {args.total_epsilon}")
  logging.info(f"  - Delta:   {args.total_delta}\n")
  logging.info("Stage 2 (Subsampled Gaussian) Parameters:")
  logging.info(
      "  - Sampling Rate (q):"
      f" {sampling_rate_s2:.4f} ({args.batch_size_s2}/{args.dataset_size_s2})"
  )
  logging.info(f"  - Iterations:        {args.iterations_s2}")
  logging.info(f"  - Noise (sigma):     {args.sigma_s2}\n")

  # --- Find Sigma for Stage 1 ---
  required_sigma_stage1 = find_sigma_for_first_stage(
      total_epsilon=args.total_epsilon,
      total_delta=args.total_delta,
      subsampling_rate_stage2=sampling_rate_s2,
      iterations_stage2=args.iterations_s2,
      sigma_stage2=args.sigma_s2,
  )

  # --- Verification and Final Output ---
  logging.info("=" * 40)
  logging.info("          RESULTS & VERIFICATION")
  logging.info("=" * 40)
  logging.info(f"Required Sigma for Stage 1: {required_sigma_stage1:.6f}")

  # The ρ (rho) parameter for Gaussian mechanism is 1 / (2 * sigma^2)
  rho_stage1 = 1 / (2 * required_sigma_stage1**2)
  logging.info(f"Corresponding Rho for Stage 1 (ρ): {rho_stage1:.6f}")

  # Verify the calculation
  final_accountant = rdp_privacy_accountant.RdpAccountant()
  final_accountant.compose(
      dp_event.GaussianDpEvent(noise_multiplier=required_sigma_stage1)
  )
  subsampled_event2 = dp_event.PoissonSampledDpEvent(
      sampling_probability=sampling_rate_s2,
      event=dp_event.GaussianDpEvent(noise_multiplier=args.sigma_s2),
  )
  final_accountant.compose(subsampled_event2, count=args.iterations_s2)
  final_epsilon = final_accountant.get_epsilon(target_delta=args.total_delta)

  logging.info(
      f"Verification: Total Epsilon after composition: {final_epsilon:.4f}"
  )
  logging.info("=" * 40)


if __name__ == "__main__":
  main()
