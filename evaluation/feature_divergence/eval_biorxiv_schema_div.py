"""Computes statistical divergence (TVD and JSD) between two datasets.

This script calculates 1-way marginal divergences for a specified set of
columns and the mean 2-way marginal divergence across all column pairs.
"""

# --- Standard Library Imports ---
import argparse
from collections import Counter
from itertools import combinations
import logging
import os
import sys

# --- Third-Party Library Imports ---
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon as jsd

# --- Setup Standard Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# --- Global Constants ---
# Fields to be used for 1-way marginal divergence calculations.
FIELDS_FOR_1WAY_DIVERGENCE = [
    'map_primary_research_area',
    'map_model_organism',
    'map_experimental_approach',
    'map_dominant_data_type',
    'map_research_focus_scale',
    'map_disease_mention',
    'map_sample_size',
    'map_research_goal',
]


# --- Core Functions ---


def compute_category_divergence(
    series1: pd.Series, series2: pd.Series
) -> tuple[float, float, int, int]:
  """Computes TVD and JSD between two categorical series."""
  # Compute normalized distributions
  counts1 = Counter(series1.dropna())
  counts2 = Counter(series2.dropna())

  all_categories = set(counts1.keys()).union(counts2.keys())

  dist1 = np.array([counts1.get(cat, 0) for cat in all_categories]) / len(
      series1
  )
  dist2 = np.array([counts2.get(cat, 0) for cat in all_categories]) / len(
      series2
  )

  # Compute Total Variation Distance (TVD)
  tvd = 0.5 * np.sum(np.abs(dist1 - dist2))

  # Compute Jensen-Shannon Divergence (JSD)
  jsd_value = jsd(dist1, dist2)

  return tvd, jsd_value, len(counts1), len(counts2)


def mean_2way_divergence(
    df1: pd.DataFrame, df2: pd.DataFrame
) -> tuple[float, float]:
  """Computes the mean TVD and JSD over all 2-way marginals."""
  jsd_vals = []
  tvd_vals = []

  # Assumes df1 and df2 have the same columns for comparison
  for c1, c2 in combinations(df1.columns, 2):
    # Build the joint samples as tuples
    joint1 = list(zip(df1[c1], df1[c2]))
    joint2 = list(zip(df2[c1], df2[c2]))

    tvd_val, jsd_val, _, _ = compute_category_divergence(
        pd.Series(joint1), pd.Series(joint2)
    )
    tvd_vals.append(tvd_val)
    jsd_vals.append(jsd_val)

  return np.mean(tvd_vals), np.mean(jsd_vals)


def main():
  """Main function to run the data divergence analysis."""
  parser = argparse.ArgumentParser(
      description='Compute statistical divergence between two datasets.'
  )
  parser.add_argument(
      '--input_path',
      '-i',
      required=True,
      help='Path to the generated data CSV file.',
  )
  parser.add_argument(
      '--reference_file',
      '-r',
      default='biorxiv_schema_v2_valid_test_gemini-2.5-flash_parsed.csv',
      help='Path to the reference (real) data CSV file.',
  )
  parser.add_argument(
      '--output_file',
      '-o',
      required=True,
      help='Path to save the output divergence scores CSV.',
  )
  args = parser.parse_args()

  # --- Load Data ---
  try:
    logging.info(f'Loading generated data from: {args.input_path}')
    df_generated = pd.read_csv(args.input_path)

    logging.info(f'Loading reference data from: {args.reference_file}')
    df_reference = pd.read_csv(args.reference_file)
  except FileNotFoundError as e:
    logging.error(f'File not found: {e}')
    sys.exit(1)
  except Exception as e:
    logging.error(f'Error reading data files: {e}')
    sys.exit(1)

  # --- 1-Way Marginal Divergence Calculation ---
  logging.info('Computing 1-way marginal divergences...')
  divergence_results = {}
  for field in FIELDS_FOR_1WAY_DIVERGENCE:
    # The generated data has a different column name convention
    generated_col_name = field.replace('map_', '')

    if field not in df_reference.columns:
      logging.warning(f"Column '{field}' not in reference data. Skipping.")
      continue
    if generated_col_name not in df_generated.columns:
      logging.warning(
          f"Column '{generated_col_name}' not in generated data. Skipping."
      )
      continue

    tvd, jsd, unique_ref, unique_gen = compute_category_divergence(
        df_reference[field], df_generated[generated_col_name]
    )
    divergence_results[field] = {
        'tvd': tvd,
        'jsd': jsd,
        'unique_real': unique_ref,
        'unique_condgen': unique_gen,
    }

  # --- 2-Way Marginal Divergence Calculation ---
  logging.info('Computing mean 2-way marginal divergences...')

  # Prepare generated dataframe for 2-way analysis by aligning column names.
  # This block is preserved exactly from the original script's logic.
  df_generated_renamed = df_generated.copy()
  df_generated_renamed.fillna('NA (added)', inplace=True)

  cols_to_drop = [
      col
      for col in df_generated_renamed.columns
      if 'map_' + col not in FIELDS_FOR_1WAY_DIVERGENCE
  ]
  df_generated_renamed.drop(columns=cols_to_drop, inplace=True)

  rename_dict = {
      col.replace('map_', ''): col for col in FIELDS_FOR_1WAY_DIVERGENCE
  }
  df_generated_renamed.rename(columns=rename_dict, inplace=True)

  # Ensure both dataframes have the same columns for comparison
  common_cols = list(
      set(df_reference.columns) & set(df_generated_renamed.columns)
  )

  mean_tvd, mean_jsd = mean_2way_divergence(
      df_reference[common_cols], df_generated_renamed[common_cols]
  )
  divergence_results['2way'] = {
      'tvd': mean_tvd,
      'jsd': mean_jsd,
      'unique_real': None,
      'unique_condgen': None,
  }

  # --- Save Results ---
  try:
    output_df = pd.DataFrame.from_dict(divergence_results, orient='index')
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)

    logging.info(f'Saving divergence results to {args.output_file}')
    output_df.to_csv(args.output_file)
    logging.info('Analysis complete.')
  except IOError as e:
    logging.error(f'Could not write to output file: {e}')
    sys.exit(1)


if __name__ == '__main__':
  main()
