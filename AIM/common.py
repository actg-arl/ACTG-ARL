
import functools

import mbi
import numpy as np
import scipy

import transformations


def exponential_mechanism(
    quality_scores: np.ndarray,
    epsilon: float,
    sensitivity: float,
    prng: np.random.RandomState = np.random,
    monotonic: bool = False,
) -> int:
  """Returns an index chosen by the exponential mechanism."""
  coef = 1.0 if monotonic else 0.5
  scores = coef * epsilon / sensitivity * quality_scores
  probas = np.exp(scores - scipy.special.logsumexp(scores))
  return prng.choice(quality_scores.size, p=probas)


def measure_marginals_with_noise(
    data: mbi.Projectable,
    marginal_queries: list[tuple[str, ...]],
    sigma: float,
    weights: np.ndarray | None = None,
) -> list[mbi.LinearMeasurement]:
  """Measures the given marginal queries with the Gaussian mechanism.

  Args:
    data: The sensitive dataset whose marginals are to be measured.
    marginal_queries: The list of marginal queries to measure, represented as a
      list of tuples of column names.
    sigma: The standard deviation of the Gaussian noise to add.
    weights: The weights to use for each marginal query.
      If None, use uniform weights.

  Returns:
    The list of LinearMeasurements.
  """
  if weights is None:
    weights = np.ones(len(marginal_queries))
  weights = np.array(weights) / np.linalg.norm(weights)
  if len(weights) != len(marginal_queries):
    raise ValueError(
        'The number of weights must be equal to the number of marginal queries.'
    )
  measurements = []
  for proj, wgt in zip(marginal_queries, weights):
    x = data.project(proj).datavector()
    y = x + np.random.normal(loc=0, scale=sigma / wgt, size=x.size)
    measurements.append(mbi.LinearMeasurement(y, proj, sigma / wgt))
  return measurements


def _weighted_identity(weights, x: mbi.Factor):
  # We make this a global function so that it can be pickle-serialized.
  return x.datavector() * weights


def compressed_measurement(
    one_way_measurement: mbi.LinearMeasurement,
    size: int,
    transform_fn: transformations.DataTransformation[int, int],
) -> mbi.LinearMeasurement:
  """Returns a measurement defined over the compressed domain.

  Args:
    one_way_measurement: The measurement to compress.
    size: The size of the compressed domain.
    transform_fn: The domain compression transformation.

  Returns:
    A measurement defined over the compressed domain.
  """
  if len(one_way_measurement.clique) != 1:
    raise ValueError(
        'The measurement must be defined with respect to a one-way marginal,'
        f' got {one_way_measurement.clique}.'
    )
  y = one_way_measurement.noisy_measurement
  mapping = np.array([transform_fn(i) for i in range(y.size)])
  y2 = np.bincount(mapping, weights=y, minlength=size)
  coefs = np.sqrt(np.bincount(mapping, minlength=size))
  return mbi.LinearMeasurement(
      y2 / coefs,
      one_way_measurement.clique,
      one_way_measurement.stddev,
      query=functools.partial(_weighted_identity, 1.0 / coefs)
  )


def compression_transformation(
    measurement: mbi.LinearMeasurement,
) -> tuple[int, transformations.DataTransformation[int, int]]:
  """Returns a domain compression transformation for the given measurement."""
  mask = measurement.noisy_measurement < 3 * measurement.stddev
  size, transform_fn = transformations.create_rare_value_merging_transformation(
      mask
  )
  return size, transform_fn


def convert_to_total_measurement(
    measurements: list[mbi.LinearMeasurement],
) -> mbi.LinearMeasurement:
  """Converts a list of measurements to a measurement of total records."""
  # Note: This is a hack to get around the fact that
  # mbi.estimation.minimum_variance_unbiased_total does not work on compressed
  # measurements.
  total = mbi.estimation.minimum_variance_unbiased_total(measurements)
  return mbi.LinearMeasurement(
      noisy_measurement=total,
      clique=(),
      stddev=1.0  # ideally we'd get this from minimum_variance_unbiased_total.
  )


def get_domain_compression_transformations(
    one_way_measurements: list[mbi.LinearMeasurement],
) -> tuple[
    mbi.Domain,
    list[mbi.LinearMeasurement],
    dict[str, transformations.DataTransformation[int, int]],
]:
  """Returns a new domain and transformations for compressing the domain.

  Args:
    one_way_measurements: List of one-way measurements over the original domain.

  Returns: A tuple of three elements:
    - The new (compressed) domain.
    - The list of measurements defined over the compressed domain.
    - A dictionary mapping each column of the original domain to a
      transformation that maps values in that column to values in the
      compressed domain.
  """
  column_transforms = {}
  sizes = {}
  new_measurements = []
  for measurement in one_way_measurements:
    col = measurement.clique[0]
    size, transform_fn = compression_transformation(measurement)
    sizes[col] = size
    column_transforms[col] = transform_fn
    new_measurements.append(
        compressed_measurement(measurement, size, transform_fn)
    )
  return mbi.Domain.fromdict(sizes), new_measurements, column_transforms
