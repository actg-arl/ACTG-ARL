"""Utilities for representing attribute domains for categorical+numerical data.

A categorical attribute can be characterized by a finite set of possible
values. A numerical attribute can be characterized by a range of possible
values. In typical differential privacy scenarios, it is assumed/required that
the domain is known in advance. This module provides dataclasses for
representing this information.

In this library we provide special treatment for out-of-domain values.  There
are two ways that we support handling out-of-domain values: (1) directly model
them or (2) map them to an element of the domain.

Values from categorical attributes that are not in the list of possible values
are mapped to some value in the list (default is the first element).
For safety, it's best practice to add a `None` value as the first entry in the
list of possible values for categorical attributes if you are unsure if there
are out-of-domain values.  If you know there are not any out-of-domain values,
then you can (and should) ignore this advice, so that downstream mechanisms
don't generate out-of-domain values when none should exist.

Similarly, values from numerical attributes that are outside of the range
[min_value, max_value] (or not even a numerical type) are clipped to this range
if `clip_to_range=True`. Otherewise, they are mapped to a special out-of-domain
value.  As before we recommend setting clip_to_range=False to directly model
out-of-domain values.  If you are sure none should exist, you can (and should)
ignore this advice, so that downstream mechanisms don't generate out-of-domain
values when none should exist.
"""

import functools
import math
from typing import TypeAlias, Mapping

import attr
import pandas as pd
import yaml


CategoricalValue: TypeAlias = None | bool | int | str | pd.Interval


@attr.define(frozen=True)
class CategoricalAttribute:
  """Dataclass for storing metadata about a categorical attribute.

  We require knowing the set of possible values in advance, and these must be
  specified via the `possible_values` attribute. If a value does not exist
  in this list, it will be mapped to possible_values[out_of_domain_index].
  If your data does not contain out-of-domain values, this will be a no-op.
  If you are unsure, we recommend you include a `None` or other special
  value as the first entry of the `possible_values` list.

  Attributes:
    possible_values: A list of possible values for this attribute.
    out_of_domain_index: The index into possible_values that out-of-domain
      values should be mapped to.
  """

  possible_values: list[CategoricalValue] = attr.field(converter=list)
  # TODO: b/372948651 - Consider refactoring in terms of out_of_domain_value.
  out_of_domain_index: int = attr.field(default=0)

  def __attrs_post_init__(self):
    if self.size == 0:
      raise ValueError('Possible values must not be empty.')
    if self.out_of_domain_index < 0 or self.out_of_domain_index >= self.size:
      raise ValueError(
          f'out_of_domain_index must be in [0, {self.size-1}], got'
          f' {self.out_of_domain_index}.'
      )

  @functools.cached_property
  def size(self) -> int:
    """Returns the number of possible values."""
    return len(self.possible_values)


@attr.define(frozen=True)
class NumericalAttribute:
  """Dataclass for storing metadata about a numerical attribute.

  We require knowing the range of possible values in advance, and these must be
  specified via the `min_value` and `max_value` attributes. If
  `out_of_domain_allowed` is True, out-of-domain values will be mapped to
  `None`. If it is False, out-of-domain values will be clipped to
  (min_value, max_value) if they are a numeric type, and to min_value otherwise.

  Attributes:
    min_value: The minimum possible value for this attribute (inclusive).
    max_value: The maximum possible value for this attribute (inclusive).
    clip_to_range: Specifies how out-of-domain values should be treated.
      If True, out-of-domain values will be clipped to the range
      [min_value, max_value] if they are a numeric type, and to min_value
      otherwise.  If False, out-of-domain values will be grouped together and
      treated as a single special out-of-domain value.
    dtype: The dtype of the data (either 'int' or 'float').
  """

  min_value: float = attr.field(converter=float)
  max_value: float = attr.field(converter=float)
  clip_to_range: bool = attr.field(default=True)
  dtype: str = attr.field(default='float')

  @min_value.validator  # pytype: disable=attribute-error
  def _validate_min_max(self, *_):
    if self.min_value > self.max_value:
      raise ValueError(
          f'min_value ({self.min_value}) must be less than or equal to'
          f' max_value ({self.max_value})'
      )

  @dtype.validator  # pytype: disable=attribute-error
  def _validate_dtype(self, *_):
    if self.dtype not in ['int', 'float']:
      raise ValueError(
          f'dtype must be either "int" or "float", got {self.dtype}.'
      )

  @property
  def exclusive_min_value(self) -> float:
    """Returns the exclusive minimum value for this attribute."""
    if self.dtype == 'int':
      return self.min_value - 1
    return math.nextafter(self.min_value, -math.inf)


AttributeType = CategoricalAttribute | NumericalAttribute
