from dataclasses import dataclass, field
from typing import Optional
from trl import PPOConfig


@dataclass
class PPOPTXConfig(PPOConfig):
  """Configuration class for PPOPTXTrainer.

  This class extends the base TRL PPOConfig with additional parameters
  for the pre-training mixed (PTX) objective.

  Attributes:
    ptx_coef: The coefficient for the PTX loss.
    ptx_coef_initial: The initial coefficient for the PTX loss.
    ptx_coef_final: The final coefficient for the PTX loss.
    block_size: The block size for the PTX pre-training data.
  """

  ptx_coef: Optional[float] = field(
      default=0.1,
      metadata={'help': 'Coefficient for the PTX loss'},
  )
  ptx_coef_initial: Optional[float] = field(
      default=0.1,
      metadata={'help': 'Initial coefficient for the PTX loss'},
  )
  ptx_coef_final: Optional[float] = field(
      default=0.1,
      metadata={'help': 'Final coefficient for the PTX loss'},
  )
  block_size: Optional[int] = field(
      default=512,
      metadata={'help': 'Block size for the PTX pretraining data'},
  )
