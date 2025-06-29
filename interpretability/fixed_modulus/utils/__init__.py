# PRNGS/torch/utils/__init__.py
# This file can be modified in further.

from .datasets import LCG, TLCG, PCG_RS, PCG_RR, PCG_XS, PCG_XSH_RR, PRNGsDataset
from .gpt2 import GPT, GPTConfig
from .prng_data import find_as, find_coprimes, lcg, lcg_vectorized, truncated_lcg, pcg_rs, pcg_rr, pcg_xs, pcg_xsh_rr, rotate_left, rotate_right
from .schedules import linear_warmup_cosine

# If you want to restrict what's available when someone does `from utils import *`,
# you can define __all__:
__all__ = [
    # From .datasets
    "LCG",
    "TLCG",
    "PCG_RS",
    "PCG_RR",
    "PCG_XS",
    "PCG_XSH_RR",
    "PRNGsDataset",
    
    # From .gpt2
    "GPT",
    "GPTConfig",
    
    # From .prng_data
    "find_as",
    "find_coprimes",
    "lcg",
    "lcg_vectorized",
    "truncated_lcg",
    "pcg_rs",
    "pcg_rr",
    "pcg_xs",
    "pcg_xsh_rr",
    "rotate_left",
    "rotate_right",
    
    # From .schedules
    "linear_warmup_cosine"
]