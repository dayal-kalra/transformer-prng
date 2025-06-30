import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from .prng_data import *

class LCG(Dataset):
    """
    Dataset class for Linear Congruential Generator (LCG) sequences.
    LCG generates pseudo-random numbers using the recurrence relation:
    X_{n+1} = (a * X_n + c) mod p
    """

    def __init__(
            self,
            p = 512,                          # Modulus for LCG
            length = 33,                      # Length of each sequence
            a_list = [],                      # List of multiplier values
            c_list = [],                      # List of increment values
            num_examples = None,              # Number of examples to generate (optional)
            rng = np.random.default_rng(97)   # Random number generator with fixed seed
    ):
        """
        Initialize the LCG dataset.
        Creates sequences for all combinations of multipliers (a) and increments (c).
        """
        self.p = p
        self.sequences = []
        # Create all possible combinations of a and c values
        a_mesh, c_mesh = np.meshgrid(a_list, c_list)
        a_flat = a_mesh.flatten()
        c_flat = c_mesh.flatten()
        ac = np.vstack((a_flat, c_flat)).T
        # Generate sequences for each (a,c) pair
        for a,c in ac:
            self.sequences.append(lcg(p=self.p, length=length, a=a, c=c, 
                                   num_examples=num_examples, rng=rng))
        self.sequences = torch.cat(self.sequences, dim=0)

    def __len__(self):
        """Return the total number of sequences in the dataset."""
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        """
        Return a single sequence pair (input, target) at given index.
        Input: sequence[:-1], Target: sequence[1:]
        """
        x, y = self.sequences[idx][:-1], self.sequences[idx][1:]
        return x,y

######## Base b Datasets ########

class BaseBLCG(Dataset):
    """
    Dataset class for Base-b Linear Congruential Generator sequences.
    Similar to LCG but represents numbers in a different base system.
    """

    def __init__(
            self,
            base = 64,                        # Base of the number system
            digits = 2,                       # Number of digits in each number
            length = 65,                      # Length of each sequence
            a_list = [],                      # List of multiplier values
            c_list = [],                      # List of increment values
            p = None,                         # Modulus (optional)
            num_examples = None,              # Number of examples to generate (optional)
            rng = np.random.default_rng(97)   # Random number generator with fixed seed
    ):
        """
        Initialize the Base-b LCG dataset.
        Creates sequences for all combinations of multipliers (a) and increments (c).
        """
        self.sequences = []
        # Create all possible combinations of a and c values
        a_mesh, c_mesh = np.meshgrid(a_list, c_list)
        a_flat = a_mesh.flatten()
        c_flat = c_mesh.flatten()
        ac = np.vstack((a_flat, c_flat)).T
        # Generate sequences for each (a,c) pair
        for a,c in ac:
            self.sequences.append(base_b_lcg(base=base, digits=digits, p=p, length=length,
                                           a=a, c=c, rng=rng, num_examples=num_examples))
        self.sequences = torch.cat(self.sequences, dim=0)

    def __len__(self):
        """Return the total number of sequences in the dataset."""
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        """
        Return a single sequence pair (input, target) at given index.
        Input: sequence[:-1], Target: sequence[1:]
        """
        x, y = self.sequences[idx][:-1], self.sequences[idx][1:]
        return x,y
    
