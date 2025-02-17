import torch
from torch.utils.data import Dataset
import numpy as np
from .prng_data import lcg, lcg_vectorized

class LCG(Dataset):
    """
    A dataset class for generating sequences using the Linear Congruential Generator (LCG) algorithm.

    Args:
        p (int): The modulus value for the LCG algorithm.
        length (int): The length of each generated sequence.
        a_list (list): A list of values for the multiplier parameter 'a' in the LCG algorithm.
        c_list (list): A list of values for the increment parameter 'c' in the LCG algorithm.
        num_examples (int, optional): The number of examples to generate for each combination of 'a' and 'c'. If None, all possible examples will be generated.
        seed (int): The seed value for the random number generator.

    Attributes:
        p (int): The modulus value for the LCG algorithm.
        sequences (Tensor): A tensor containing the generated sequences.

    Methods:
        __len__(): Returns the number of sequences in the dataset.
        __getitem__(idx): Returns the input-output pair at the specified index.

    """

    def __init__(
            self,
            p = 512,
            length = 33,
            a_list = [],
            c_list = [],
            num_examples = None,
            seed = 0
    ):
        self.p = p
        self.sequences = []
        a_mesh, c_mesh = np.meshgrid(a_list, c_list)
        a_flat = a_mesh.flatten()
        c_flat = c_mesh.flatten()
        ac = np.vstack((a_flat, c_flat)).T
        for a, c in ac:
            self.sequences.append(lcg(p=self.p, length=length, a=a, c=c, num_examples=num_examples))
        self.sequences = torch.cat(self.sequences, dim=0)

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        x, y = self.sequences[idx][:-1], self.sequences[idx][1:]
        return x,y
    

class LCGVectorized(Dataset):
    """
    A vectorized for of LCG class
    """

    def __init__(
            self,
            p = 512,
            length = 33,
            a_list = [],
            c_list = [],
            num_examples = None,
            seed = 0
    ):
        self.p = p
        self.sequences = lcg_vectorized(p=self.p, length=length, a_list=a_list, c_list=c_list, num_examples=num_examples)

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        x, y = self.sequences[idx][:-1], self.sequences[idx][1:]
        return x,y

class PRNGsDataset(Dataset):
    """
    A Dataset for PRNGs; Returns tokens[:-1] as the input and tokens[1:] as the output
    """

    def __init__(
            self,
            sequences,
    ):
        self.sequences = sequences

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        x, y = self.sequences[idx][:-1], self.sequences[idx][1:]
        return x,y
