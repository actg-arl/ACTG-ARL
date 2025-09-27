from typing import Mapping

from absl import logging
import mbi
import pandas as pd


import domain
import transformations
import accounting
import common
import aim

def run(
    attribute_domains: Mapping[
        str, domain.CategoricalAttribute | domain.NumericalAttribute
    ],
    df: pd.DataFrame,
    epsilon: float,
    delta: float,
    rho: float,
    seed: int = 0,
    numerical_bins: int = 32,
    one_way_marginal_budget_fraction: float = 0.1,
    num_records: int | None = None,  # New parameter
    **kwargs,
) -> pd.DataFrame:
  """Runs a mechanism end-to-end, with discretization and categorical encoding.

  Args:
    attribute_domains: A mapping from column names to attribute domains. Every
      key in this mapping must be a column of `df`.
    df: The dataset to generate synthetic data for.
    epsilon: The epsilon-DP parameter.
    delta: The delta-DP parameter.
    seed: The seed for the random number generator.
    numerical_bins: The number of bins to use for discretization.
    one_way_marginal_budget_fraction: The fraction of the total privacy budget
      to use for one-way marginal queries.
    **kwargs: Additional keyword arguments for AIM.

  Returns:
    A synthetic dataset.
  """
  assert 0 <= one_way_marginal_budget_fraction <= 1
 
  #####################################
  # Map the data to a discrete domain #
  #####################################
  transform_fns = {}
  discrete_domains = {}
  for col, attr in attribute_domains.items():
    if col not in df.columns:
      raise ValueError(f'Column {col} not found in dataset.')

    if isinstance(attr, domain.CategoricalAttribute):
      logging.info('Encoding categorical column: %s', col)
      transform_fns[col] = transformations.discrete_encoder(attr)
      discrete_domains[col] = attr.size
    else:
      # This is a numerical attribute.
      logging.info('Discretizing numerical column: %s', col)
      cat_attr, to_categorical = (
          transformations.create_uniform_discretize_transformation(
              attr, numerical_bins
          )
      )
      logging.info('Encoding discretized numerical column: %s', col)
      to_discrete = transformations.discrete_encoder(cat_attr)
      # transform_fn(x) = to_discrete(to_categorical(x))
      transform_fns[col] = to_discrete @ to_categorical
      discrete_domains[col] = cat_attr.size

  discrete = pd.DataFrame()
  for col in transform_fns:
    discrete[col] = df[col].apply(transform_fns[col])
  data = mbi.Dataset(discrete, mbi.Domain.fromdict(discrete_domains))

  if not rho:
    rho = accounting.zcdp_rho(epsilon, delta)
  print('rho', rho)
  one_way_rho = one_way_marginal_budget_fraction * rho
  rho_remaining = rho - one_way_rho
  #######################################################################
  # Measure 1-way marginals and compress domain by merging rare values. #
  #######################################################################
  one_way_marginal_queries = [(col,) for col in data.domain]
  per_query_rho = one_way_rho / len(one_way_marginal_queries)
  per_query_sigma = accounting.zcdp_gaussian_sigma(per_query_rho)
  one_way_measurements = common.measure_marginals_with_noise(
      data, one_way_marginal_queries, per_query_sigma
  )
  logging.info('[SynthKit Tabular]: Measured one-way marginals.')
  compressed_domain, compressed_one_way_measurements, compress_transforms = (
      common.get_domain_compression_transformations(one_way_measurements)
  )

  total_measurement = common.convert_to_total_measurement(one_way_measurements)

  for col in compress_transforms:
    transform_fns[col] = compress_transforms[col] @ transform_fns[col]
  logging.info(
      '[SynthKit Tabular]: Estimated Total %d',
      total_measurement.noisy_measurement,
  )
  compressed_data = mbi.Dataset(
      transformations.apply(data.df, compress_transforms),
      compressed_domain,
  )
  logging.info('[SynthKit Tabular]: Compressed domain: %s', compressed_domain)

  measurements = [total_measurement] + list(compressed_one_way_measurements)

  # Run the mechanism on the discretized data.
  model = aim.adaptive_iterative_mechanism(
      compressed_data,
      rho_remaining,
      seed=seed,
      initial_measurements=measurements,
      **kwargs,
  )
  # If num_records is specified, use it. Otherwise, use the noisy total.
  n_records_to_generate = (
      num_records
      if num_records is not None
      else int(total_measurement.noisy_measurement)
  )
  logging.info(
      '[SynthKit Tabular]: Generating %d synthetic records.',
      n_records_to_generate,
  )
  synthetic_data = model.synthetic_data(rows=n_records_to_generate)

  # Convert synthetic data back to the original domain.
  synthetic_columns = {}
  for col in transform_fns:
    synthetic_columns[col] = pd.Series(
        [transform_fns[col].inverse(x) for x in synthetic_data.df[col]],
        dtype=df[col].dtype,
    )

  return pd.DataFrame(synthetic_columns)
