import torch
import sympy
import numpy as np

### The values of a's and c's are 

def find_as(p, limit: int | None = None) -> list[int]:
    """
    Finds a list of numbers 'a' that maximize the period for a Linear Congruential Generator (LCG) with a given prime modulus 'p'.
    This requires the following conditions to be satisified:
        1. a-1 is divisible by all prime factors of p
        2. a-1 is divisible by 4 if p is divisible by 4. 
    Parameters:
        p (int): The number for which to find the list of 'a' values.
        limit (int, optional): The upper limit for the 'a' values. If not provided, it defaults to 'p'.
    Returns:
        list: A list of 'a' values satisfying the given conditions.
    """
    if limit is None:
        limit = p
    factors = sympy.primefactors(p)
    if p % 4 == 0: # if p is divisble by 4
        factors.append(4) # add 4 because we want a-1 to be divisible by 4
        factors.remove(2) # remove 2, otherwise lcm will have 2*4 and this will exclude some numbers
    lcm = 1
    for factor in set(factors):
        lcm *= factor 
    result = []
    for k in range(1, limit // lcm):
        a = k*lcm + 1
        result.append(a)
    return result


def find_coprimes(m, low: int = 3) -> list[int]:
    """
    Finds all the coprimes of a given number <==> find c such that gcd(c, p) == 1

    Parameters:
    - m (int): The number to find coprimes for.
    - low (int): The lower bound for the range of numbers to check for coprimes. Default is 3.

    Returns:
    - list: A list of coprimes of the given number.
    """
    coprimes = [n for n in range(low, m) if np.gcd(n, m) == 1]
    return coprimes



def lcg(p: int = 512, length: int = 8, a: int = 45, c: int = 123, num_examples: int | None = None) -> torch.Tensor:
    """
    Generates a random sequence of integers using the Linear Congruential Generator (LCG) algorithm.

    Args:
        p (int): The modulus value. Defaults to 512.
        length (int): The length of each sequence. Defaults to 8.
        a (int): The multiplier value. Defaults to 45.
        c (int): The increment value. Defaults to 123.
        num_examples (int): The number of examples to generate. If not provided, it will generate as many examples as possible.

    Returns:
        torch.Tensor: A tensor containing the generated sequences of random integers.
    """
    if num_examples is not None:
        n = min(p // length, num_examples)
    else:
        n = p // length

    array = torch.zeros(n * length, dtype=torch.int64)
    array[0] = torch.randint(0, p, (1,))  # Initial seed

    for i in range(1, n * length):
        array[i] = (a*array[i - 1] + c) % p

    reshaped_array = array.reshape(n, length)
    return reshaped_array



def lcg_vectorized(p: int = 512, length: int = 8, a_list: list = [45], c_list: list = [123], num_examples: int | None = None, chunk_size: int | None = None) -> torch.Tensor:
    """
    Vectorized version of lcg function
    """
    # Create mesh grid and flatten
    a_mesh, c_mesh = np.meshgrid(a_list, c_list)
    a_flat = torch.tensor(a_mesh.flatten(), dtype=torch.int64)
    c_flat = torch.tensor(c_mesh.flatten(), dtype=torch.int64)
    

    if num_examples is not None:
        n = min(p // length, num_examples)
    else:
        n = p // length
    
    # Generate initial seeds outside of vmap
    num_sequences = len(a_flat)
    initial_seeds = torch.randint(0, p, (num_sequences,), dtype=torch.int64)

    def single_lcg(a, c, seed):
        
        @torch.compile
        def next_value(prev):
            return (a*prev + c) % p
        
        sequence = [seed]
        for _ in range(n * length - 1):
            sequence.append(next_value(sequence[-1]))
        
        return torch.stack(sequence).reshape(n, length)

    # Vectorize over a, c, and initial seeds
    results = torch.vmap(single_lcg, chunk_size = chunk_size)(a_flat, c_flat, initial_seeds)
    
    # Reshape to combine all sequences
    return results.reshape(-1, length)