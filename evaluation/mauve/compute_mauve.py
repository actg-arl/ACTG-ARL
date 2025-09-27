# Author: Krishna Pillutla
# License: GPLv3
# 
# Modifications:
#   - Simplified the code logic to compute MAUVE on loaded embeddings.

import argparse
import logging
import math
import time
from types import SimpleNamespace
import faiss
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import auc as compute_area_under_curve
from sklearn.preprocessing import normalize
import time

# --- Setup Standard Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def compute_mauve(
    p_features=None,
    q_features=None,
    num_buckets='auto',
    pca_max_data=-1,
    kmeans_explained_var=0.9,
    kmeans_num_redo=5,
    kmeans_max_iter=500,
    divergence_curve_discretization_size=25,
    mauve_scaling_factor=5,
    verbose=False,
    seed=25,
):
  """Compute the MAUVE score between two text generations P and Q.

  P is specified as ``p_features``. Same with Q.

  :param ``p_features``: ``numpy.ndarray`` of shape (n, d), where n is the
  number of generations.
  :param ``q_features``: ``numpy.ndarray`` of shape (n, d), where n is the
  number of generations.
  :param ``num_buckets``: the size of the histogram to quantize P and Q.
  Options: ``'auto'`` (default, which is n/10) or an integer.
  :param ``pca_max_data``: the number data points to use for PCA. If `-1`, use
  all the data. Default -1.
  :param ``kmeans_explained_var``: amount of variance of the data to keep in
  dimensionality reduction by PCA. Default 0.9.
  :param ``kmeans_num_redo``: number of times to redo k-means clustering (the
  best objective is kept). Default 5.
      Try reducing this to 1 in order to reduce running time.
  :param ``kmeans_max_iter``: maximum number of k-means iterations. Default 500.
      Try reducing this to 100 in order to reduce running time.
  :param ``divergence_curve_discretization_size``: Number of points to consider
  on the divergence curve. Default 25.
      Larger values do not offer much of a difference.
  :param ``mauve_scaling_factor``: The constant``c`` from the paper. Default 5.
      See `Best Practices <index.html#best-practices-for-mauve>`_ for details.
  :param ``verbose``: If True, print running time updates.
  :param ``seed``: random seed to initialize k-means cluster assignments.

  :return: an object with fields p_hist, q_hist, divergence_curve and mauve.

  * ``out.mauve`` is a number between 0 and 1, the MAUVE score. Higher values
  means P is closer to Q.
  * ``out.frontier_integral``, a number between 0 and 1. Lower values mean that
  P is closer to Q.
  * ``out.p_hist`` is the obtained histogram for P. Same for ``out.q_hist``.
  * ``out.divergence_curve`` contains the points in the divergence curve. It is
  of shape (m, 2), where m is ``divergence_curve_discretization_size``
  """

  if p_features is None:
    raise ValueError('Supply p_features')
  if q_features is None:
    raise ValueError('Supply q_features')
  p_features = get_features_from_input(
      p_features,
  )
  q_features = get_features_from_input(
      q_features,
  )
  if num_buckets == 'auto':
    # heuristic: use num_clusters = num_generations / 10
    num_buckets = max(
        2, int(round(min(p_features.shape[0], q_features.shape[0]) / 10))
    )
  elif not isinstance(num_buckets, int):
    raise ValueError('num_buckets is expected to be an integer or "auto"')

  # Acutal binning
  t1 = time.time()
  p, q = cluster_feats(
      p_features,
      q_features,
      num_clusters=num_buckets,
      norm='l2',
      whiten=False,
      pca_max_data=pca_max_data,
      explained_variance=kmeans_explained_var,
      num_redo=kmeans_num_redo,
      max_iter=kmeans_max_iter,
      seed=seed,
      verbose=verbose,
  )
  t2 = time.time()
  if verbose:
    logging.info(f'total discretization time: {round(t2 - t1, 2)} seconds')

  # Divergence curve and mauve
  mixture_weights = np.linspace(
      1e-6, 1 - 1e-6, divergence_curve_discretization_size
  )
  divergence_curve = get_divergence_curve_for_multinomials(
      p, q, mixture_weights, mauve_scaling_factor
  )
  x, y = divergence_curve.T
  idxs1 = np.argsort(x)
  idxs2 = np.argsort(y)
  mauve_score = 0.5 * (
      compute_area_under_curve(x[idxs1], y[idxs1])
      + compute_area_under_curve(y[idxs2], x[idxs2])
  )
  fi_score = get_fronter_integral(p, q)
  to_return = SimpleNamespace(
      p_hist=p,
      q_hist=q,
      divergence_curve=divergence_curve,
      mauve=mauve_score,
      frontier_integral=fi_score,
      num_buckets=num_buckets,
  )
  return to_return


def get_features_from_input(
    features,
):
  features = np.asarray(features)
  return features


def cluster_feats(
    p,
    q,
    num_clusters,
    norm='none',
    whiten=True,
    pca_max_data=-1,
    explained_variance=0.9,
    num_redo=5,
    max_iter=500,
    seed=0,
    verbose=False,
):
  assert 0 < explained_variance < 1
  if verbose:
    logging.info(f'seed = {seed}')
  assert norm in ['none', 'l2', 'l1', None]
  data1 = np.vstack([q, p])
  if norm in ['l2', 'l1']:
    data1 = normalize(data1, norm=norm, axis=1)
  pca = PCA(n_components=None, whiten=whiten, random_state=seed + 1)
  if pca_max_data < 0 or pca_max_data >= data1.shape[0]:
    pca.fit(data1)
  elif 0 < pca_max_data < data1.shape[0]:
    rng = np.random.RandomState(seed + 5)
    idxs = rng.choice(data1.shape[0], size=pca_max_data, replace=False)
    pca.fit(data1[idxs])
  else:
    raise ValueError(
        f'Invalid argument pca_max_data={pca_max_data} with'
        f' {data1.shape[0]} datapoints'
    )
  s = np.cumsum(pca.explained_variance_ratio_)
  idx = np.argmax(s >= explained_variance)  # last index to consider
  if verbose:
    logging.info(f'performing clustering in lower dimension = {idx}')
  data1 = pca.transform(data1)[:, : idx + 1]
  # Cluster features and obtain the labels for each data point.
  data1 = data1.astype(np.float32)  # Faiss requires float32.
  t1 = time.time()
  kmeans = faiss.Kmeans(
      data1.shape[1],
      num_clusters,
      niter=max_iter,
      verbose=verbose,
      nredo=num_redo,
      update_index=True,
      seed=seed + 2,
  )
  kmeans.train(data1)
  _, labels = kmeans.index.search(data1, 1)
  labels = labels.reshape(-1)
  t2 = time.time()
  if verbose:
    logging.info(f'kmeans time: {round(t2 - t1, 2)} s')

  q_labels = labels[: len(q)]
  p_labels = labels[len(q) :]

  q_bins = np.histogram(
      q_labels, bins=num_clusters, range=[0, num_clusters], density=True
  )[0]
  p_bins = np.histogram(
      p_labels, bins=num_clusters, range=[0, num_clusters], density=True
  )[0]
  return p_bins / p_bins.sum(), q_bins / q_bins.sum()


def kl_multinomial(p, q):
  assert p.shape == q.shape
  if np.logical_and(p != 0, q == 0).any():
    return np.inf
  else:
    idxs = np.logical_and(p != 0, q != 0)
    return np.sum(p[idxs] * np.log(p[idxs] / q[idxs]))


def get_divergence_curve_for_multinomials(
    p, q, mixture_weights, scaling_factor
):
  divergence_curve = [[0, np.inf]]  # extreme point
  for w in np.sort(mixture_weights):
    r = w * p + (1 - w) * q
    divergence_curve.append([kl_multinomial(q, r), kl_multinomial(p, r)])
  divergence_curve.append([np.inf, 0])  # other extreme point
  return np.exp(-scaling_factor * np.asarray(divergence_curve))


def get_fronter_integral(p, q, scaling_factor=2):
  total = 0.0
  for p1, q1 in zip(p, q):
    if p1 == 0 and q1 == 0:
      pass
    elif p1 == 0:
      total += q1 / 4
    elif q1 == 0:
      total += p1 / 4
    elif abs(p1 - q1) > 1e-8:
      t1 = p1 + q1
      t2 = p1 * q1 * (math.log(p1) - math.log(q1)) / (p1 - q1)
      total += 0.25 * t1 - 0.5 * t2
    # else: contribution is 0
  return total * scaling_factor

def main():
  # --- Parse Arguments ---
  parser = argparse.ArgumentParser(
      description='Compute MAUVE score between two sets of features or texts.'
  )
  parser.add_argument(
      '--p_feats_path',
      type=str,
      required=True,
      help='Path to the numpy file containing p features.',
  )
  parser.add_argument(
      '--q_feats_path',
      type=str,
      required=True,
      help='Path to the numpy file containing q features.',
  )
  parser.add_argument(
      '--num_buckets',
      type=int,
      default=-1,
      help='Number of buckets for histogram. Default is "auto".',
  )
  args = parser.parse_args()

  # --- Load Features ---
  p_feats = np.load(args.p_feats_path)
  q_feats = np.load(args.q_feats_path)
  logging.info(f'p_feats shape: {p_feats.shape}')
  logging.info(f'q_feats shape: {q_feats.shape}')

  # --- Compute MAUVE ---
  t = time.time()
  result = compute_mauve(
      p_feats,
      q_feats,
      num_buckets='auto' if args.num_buckets == -1 else args.num_buckets,
  )

  # --- Log Results ---
  logging.info(f'MAUVE: {result.mauve}')
  logging.info(f'Time taken: {time.time() - t}')
  

if __name__ == '__main__':
  main()
