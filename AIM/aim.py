"""Implementation of the Adaptive+Iterative Mechanism (AIM)."""

import itertools
import time
from typing import Iterable, Mapping, TypeAlias

from absl import logging
import jax.numpy as jnp
import mbi
import mbi.estimation
import mbi.junction_tree
import more_itertools
import numpy as np

import accounting
import common


MarginalQuery: TypeAlias = tuple[str, ...]


def _downward_closure(
    marginal_queries: Iterable[MarginalQuery],
) -> Iterable[MarginalQuery]:
  """Returns the downward closure of the given marginal queries.

  Given a collection of sets, the downward closure is the set of all sets that
  are subsets of any of the given sets.

  Example Usage:
  >>> _downward_closure([('a', 'b'), ('a', 'c')])
  [('a',), ('b',), ('c',), ('a', 'b'), ('a', 'c')]

  Args:
    marginal_queries: The marginal queries to compute the downward closure of.

  Returns:
    The downward closure of the given marginal queries, without the empty tuple.
  """
  ans = set()
  for proj in marginal_queries:
    ans.update(more_itertools.powerset(proj))
  return list(sorted(ans - {()}, key=len))


def _hypothetical_model_size(
    domain: mbi.Domain, cliques: list[MarginalQuery]
) -> float:
  """Returns the size of the graphical model implied by the given cliques.

  Example Usage:
  >>> domain = mbi.Domain.fromdict({'a': 10, 'b': 10, 'c': 10})
  >>> cliques = [('a', 'b'), ('a', 'c')]
  >>> _hypothetical_model_size(domain, cliques)
  200

  Args:
    domain: The domain of the graphical model.
    cliques: The cliques of the graphical model.

  Returns:
    The sum of sizes of all maximal cliques in the implied junction tree, in
    megabytes.
  """
  jtree, _ = mbi.junction_tree.make_junction_tree(domain, cliques)
  maximal_cliques = mbi.junction_tree.maximal_cliques(jtree)
  cells = sum(domain.size(cl) for cl in maximal_cliques)
  size_mb = cells * 8 / 2**20
  return size_mb


def _compile_workload(
    workload: Mapping[MarginalQuery, float],
) -> Mapping[MarginalQuery, float]:
  """Compiles an input workload into a set of candidate measurements for AIM.

  Args:
    workload: A dictionary mapping marginal queries to weights representing the
      importance of each marginal query.

  Returns:
    A dictionary mapping marginal queries in the downward closure of the
    workload to weights representing the importance of each marginal query.
  """

  def score(cl):
    return sum(
        workload[workload_cl] * len(set(cl) & set(workload_cl))
        for workload_cl in workload
    )

  return {cl: score(cl) for cl in _downward_closure(workload.keys())}


def _filter_candidates(
    candidates: Mapping[MarginalQuery, float],
    model: mbi.MarkovRandomField,
    size_limit: float,
) -> Mapping[MarginalQuery, float]:
  """Filters the given candidates that lead to tractable graphical models.

  Args:
    candidates: The candidate marginal queries.
    model: The current graphical model.
    size_limit: The size limit in megabytes for the new graphical model, if a
      given candidate is selected.

  Returns:
    A collection of new candidates that pass the size_limit filter.
  """
  ans = {}
  free_cliques = _downward_closure(model.cliques)
  for cl in candidates:
    cond1 = (
        _hypothetical_model_size(model.domain, model.cliques + [cl])
        <= size_limit
    )
    cond2 = cl in free_cliques
    if cond1 or cond2:
      ans[cl] = candidates[cl]
  return ans


def _worst_approximated(
    candidates: Mapping[MarginalQuery, float],
    answers: mbi.CliqueVector,
    estimates: mbi.CliqueVector,
    eps: float,
    sigma: float,
    domain: mbi.Domain,
) -> MarginalQuery:
  """Returns the worst approximated candidate in the given candidates."""
  errors = {}
  for cl in candidates:
    wgt = candidates[cl]
    diff = answers[cl].datavector() - estimates[cl].datavector()
    bias = jnp.sqrt(2 / jnp.pi) * sigma * domain.size(cl)
    errors[cl] = wgt * (jnp.linalg.norm(diff, ord=1) - bias)

  max_sensitivity = max(
      candidates.values(),
  )  # if all weights are 0, could be a problem
  keys, values = list(errors.keys()), np.array(list(errors.values()))
  idx = common.exponential_mechanism(
      values, eps, max_sensitivity, monotonic=True
  )
  return keys[idx]


def adaptive_iterative_mechanism(
    data: mbi.Projectable,
    rho: float,
    *,
    workload: (
        Mapping[MarginalQuery, float] | Iterable[MarginalQuery] | None
    ) = None,
    max_rounds: int | None = None,
    seed: int = 0,
    pgm_iters: int = 1000,
    max_model_size: int = 80,
    initial_measurements: list[mbi.LinearMeasurement] | None = None,
) -> mbi.MarkovRandomField:
  """Generate synthetic data via the AIM mechanism.

  Details are described in the paper:
  [AIM: An Adaptive and Iterative Mechanism for Differentially Private Synthetic
  Data](https://arxiv.org/abs/2201.12677). This mechanism is a competitive
  algorithm within the broader SELECT-MEASURE-GENERATE paradigm. It is an
  MWEM-style algorithm (Multiplicative Weights + Exponential Mechanism), that
  iteratively improves the estimate of the data distribution by selecting
  marginal queries that are poorly approximated by the current model. It is a
  scalable algorithm that can handle high-dimensional datasets, but it can be
  time consuming to run (hours). The runtime/utility trade-off can be controlled
  by the max_model_size parameter. For quick experimentation, we recommend
  setting max_model_size = 1, for production use cases, we recommend setting
  max_model_size >= 80.

  Args:
    data: The dataset to generate synthetic data for.
    rho: the zCDP privacy budget.
    workload: A collection of marginal queries (and weights) the synthetic
      data should be tailored to.
    max_rounds: The maximum number of rounds to run the mechanism.
    seed: The seed for the random number generator.
    pgm_iters: The number of iterations for the mirror descent algorithm.
    max_model_size: The maximum size of the graphical model in megabytes.
      Controls the utility/runtime trade-off.
    initial_measurements: A list of initial measurements taken over the data.

  Returns:
    The synthetic data generated by the AIM mechanism.
  """
  logging.info('[AIM]: Starting Mechanism.')
  if workload is None:
    workload = [
        cl
        for cl in itertools.combinations(data.domain, 3)
        if data.domain.size(cl) <= 1e6
    ]

  if isinstance(workload, list):
    workload = {cl: 1.0 for cl in workload}

  np.random.seed(seed)

  #########################################################################
  # Compile workload into candidate measurements, and precompute answers. #
  #########################################################################
  candidates = _compile_workload(workload)
  answers = mbi.CliqueVector.from_projectable(data, list(candidates))
  logging.info('[AIM]: Calculated workload-query answers.')
  terminate = False
  rho_remaining = rho
  max_rounds = max_rounds or 16 * len(data.domain)
  rho_per_round = rho / max_rounds
  measurements = initial_measurements or []
  model = mbi.estimation.mirror_descent(
      data.domain, measurements, iters=pgm_iters
  )

  t = 0
  while not terminate:
    t += 1
    if rho_remaining < 2 * rho_per_round:
      logging.info('[AIM] Final round, Using all remaining privacy budget.')
      rho_per_round = rho_remaining
      terminate = True

    ############################################################################
    # Select a marginal query that is worst approximated by the current model. #
    ############################################################################
    t0 = time.time()
    rho_remaining -= rho_per_round
    sigma = accounting.zcdp_gaussian_sigma(0.9*rho_per_round)
    epsilon = accounting.zcdp_exponential_eps(0.1*rho_per_round)
    size_limit = max_model_size * (rho - rho_remaining) / rho
    small_candidates = _filter_candidates(candidates, model, size_limit)

    estimates = mbi.marginal_oracles.bulk_variable_elimination(
        model.potentials, list(small_candidates), total=model.total
    )
    marginal_query = _worst_approximated(
        small_candidates, answers, estimates, epsilon, sigma, data.domain
    )

    t1 = time.time()
    logging.info('[AIM] Found worst-approximated candidate in %.2fs', t1 - t0)
    logging.info(
        '[AIM] Round %d, Budget used: %.4f, Measuring: %s, Candidates: %d',
        t,
        (rho - rho_remaining) / rho,
        marginal_query,
        len(small_candidates)
    )

    ######################################################################
    # Measure the marginal query privately using the Gaussian mechanism. #
    ######################################################################
    measurement = common.measure_marginals_with_noise(
        data, [marginal_query], sigma
    )[0]
    measurements.append(measurement)
    old_estimate = model.project(marginal_query).datavector()

    #####################################################
    # Estimate the data distribution using Private-PGM. #
    #####################################################
    t2 = time.time()
    callback_fn = mbi.callbacks.default(measurements)
    measured_cliques = list(set(m.clique for m in measurements))
    warm_start = model.potentials.expand(measured_cliques)
    model = mbi.estimation.mirror_descent(
        data.domain,
        measurements,
        potentials=warm_start,
        iters=pgm_iters,
        callback_fn=callback_fn,
    )
    t3 = time.time()
    logging.info('[AIM] Mirror descent took %.2fs', t3 - t2)

    new_estimate = model.project(marginal_query).datavector()

    ##########################################
    # Anneal epsilon and sigma if necessary. #
    ##########################################
    threshold = sigma * np.sqrt(2 / np.pi) * data.domain.size(marginal_query)
    if np.linalg.norm(new_estimate - old_estimate) <= threshold:
      # No useful information at this noise level, so increase budget per round.
      rho_per_round *= 2
      logging.info('[AIM] Reducing sigma: %.1f', sigma)

  return model
