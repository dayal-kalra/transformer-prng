import torch
from torch.utils.data import Dataset
import numpy as np
from .prng_data import lcg, lcg_vectorized, truncated_lcg, pcg_rs, pcg_rr, pcg_xs, pcg_xsh_rr

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
        x = self.sequences[idx][:-1].clone().detach().to(dtype=torch.long)
        y = self.sequences[idx][1:].clone().detach().to(dtype=torch.long)
        return x, y


class TLCG(Dataset):
    """
    TLCG dataset class.

    Args:
        p (int): The modulus value for the truncated linear congruential generator.
        length (int): The length of each sequence in the dataset.
        bits_to_drop (int): The number of bits to drop from the generated sequence.
        a_list (list): List of values for the 'a' parameter of the generator.
        c_list (list): List of values for the 'c' parameter of the generator.
        num_examples (int, optional): The number of examples to generate. If None, generate all possible examples.

    Attributes:
        sequences (Tensor): Tensor containing the generated sequences.

    Methods:
        __len__(): Returns the number of sequences in the dataset.
        __getitem__(idx): Returns the input and target sequences at the given index.

    """

    def __init__(
            self,
            p = 512,
            length = 33,
            bits_to_drop = 8,
            a_list = [],
            c_list = [],
            num_examples = None,

    ):

        self.sequences = []
        a_mesh, c_mesh = np.meshgrid(a_list, c_list)
        a_flat = a_mesh.flatten()
        c_flat = c_mesh.flatten()
        ac = np.vstack((a_flat, c_flat)).T
        for a, c in ac:
            self.sequences.append(truncated_lcg(p=p, length=length, a=a, c=c, bits_to_drop=bits_to_drop, num_examples=num_examples))
        self.sequences = torch.cat(self.sequences, dim=0)

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        x, y = self.sequences[idx][:-1], self.sequences[idx][1:]
        return x,y
    

class PCG_RS(Dataset):
    """
    Random shift
    Permuted LCG sequences for single p
    """

    def __init__(
            self,
            p = 16384,
            length = 65,
            top_bits = 2,
            bits_to_keep = 8,
            min_shift = 0,
            a_list = [],
            c_list = [],
            num_examples = None,

    ):

        self.sequences = []
        a_mesh, c_mesh = np.meshgrid(a_list, c_list)
        a_flat = a_mesh.flatten()
        c_flat = c_mesh.flatten()
        ac = np.vstack((a_flat, c_flat)).T
        for a, c in ac:
            self.sequences.append(pcg_rs(p=p,length=length, a=a, c=c, top_bits=top_bits, bits_to_keep=bits_to_keep,min_shift=min_shift, num_examples=num_examples))
        self.sequences = torch.cat(self.sequences, dim=0)

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        x, y = self.sequences[idx][:-1], self.sequences[idx][1:]
        return x,y
    


class PCG_RR(Dataset):
    """
    Random rotation
    Permuted LCG sequences for single p
    """

    def __init__(
            self,
            p = 2 ** 16,
            length = 513,
            control_bits = 3,
            bits_to_keep = 8,
            a_list = [],
            c_list = [],
            num_examples = None,

    ):

        self.sequences = []
        a_mesh, c_mesh = np.meshgrid(a_list, c_list)
        a_flat = a_mesh.flatten()
        c_flat = c_mesh.flatten()
        ac = np.vstack((a_flat, c_flat)).T
        for a, c in ac:
            self.sequences.append(pcg_rr(p=p, length=length, a=a, c=c, control_bits=control_bits, bits_to_keep=bits_to_keep, num_examples=num_examples))
        self.sequences = torch.cat(self.sequences, dim=0)

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        x, y = self.sequences[idx][:-1], self.sequences[idx][1:]
        return x,y
    


class PCG_XS(Dataset):


    def __init__(
            self,
            p = 2 ** 16,
            length = 513,
            control_bits = 3,
            bits_to_keep = 8,
            a_list = [],
            c_list = [],
            num_examples = None,

    ):

        self.sequences = []
        a_mesh, c_mesh = np.meshgrid(a_list, c_list)
        a_flat = a_mesh.flatten()
        c_flat = c_mesh.flatten()
        ac = np.vstack((a_flat, c_flat)).T
        for a, c in ac:
            self.sequences.append(pcg_xs(p=p, length=length, a=a, c=c, control_bits=control_bits, bits_to_keep=bits_to_keep, num_examples=num_examples))
        self.sequences = torch.cat(self.sequences, dim=0)

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        x, y = self.sequences[idx][:-1], self.sequences[idx][1:]
        return x,y
    


class PCG_XSH_RR(Dataset):
    """
    Parameters:
    - p (int): The modulus value for the generator.
    - length (int): The length of each sequence.
    - control_bits (int): The number of control bits to use.
    - bits_to_keep (int): The number of bits to keep from the generated values.
    - a_list (list): A list of values for the 'a' parameter.
    - c_list (list): A list of values for the 'c' parameter.
    - num_examples (int): The number of examples to generate.

    Attributes:
    - sequences (Tensor): A tensor containing the generated sequences.

    Methods:
    - __len__(): Returns the number of sequences in the dataset.
    - __getitem__(idx): Returns the input and target sequences at the given index.

    Example usage:
    dataset = PCG_XSH_RR(p=2**16, length=513, control_bits=3, bits_to_keep=8, a_list=[1, 2, 3], c_list=[4, 5, 6])
    print(len(dataset))
    x, y = dataset[0]
    print(x, y)
    """


    def __init__(
            self,
            p = 2 ** 16,
            length = 513,
            control_bits = 3,
            bits_to_keep = 8,
            a_list = [],
            c_list = [],
            num_examples = None,

    ):

        self.sequences = []
        a_mesh, c_mesh = np.meshgrid(a_list, c_list)
        a_flat = a_mesh.flatten()
        c_flat = c_mesh.flatten()
        ac = np.vstack((a_flat, c_flat)).T
        for a, c in ac:
            self.sequences.append(pcg_xsh_rr(p=p, length=length, a=a, c=c, control_bits=control_bits, bits_to_keep=bits_to_keep, num_examples=num_examples))
        self.sequences = torch.cat(self.sequences, dim=0)

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        x, y = self.sequences[idx][:-1], self.sequences[idx][1:]
        return x,y