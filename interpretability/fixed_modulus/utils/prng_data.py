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
        n = max(min(p // length, num_examples), 1)
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
        n = max(min(p // length, num_examples), 1)
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



def truncated_lcg(p: int = 512, length: int = 8, a: int = 45, c: int = 123, bits_to_drop: int = 8, num_examples: int | None = None) -> torch.Tensor:
    """
    Generates truncated linear congruential generator (LCG) sequences.
    Args:
        p (int): The modulus value for the LCG sequence. Default is 512.
        length (int): The length of each example in the sequence. Default is 8.
        a (int): The multiplier value for the LCG sequence. Default is 45.
        c (int): The increment value for the LCG sequence. Default is 123.
        bits_to_drop (int): The number of bits to drop from each generated value. Default is 8.
        num_examples (int): The number of examples to generate. If None, it generates as many examples as possible. Default is None.
    Returns:
        torch.Tensor: A tensor containing the generated LCG sequence, reshaped into a 2D tensor of shape (num_examples, length).
    """
    
    if num_examples is not None:
        n = min(p // length, num_examples)
    else:
        n = p // length

    array = torch.zeros(n * length, dtype=torch.int64)
    x = np.random.randint(0, p)
    bit_length = np.ceil(np.log2(p)).astype(int)
    assert bit_length > bits_to_drop, "Dropping more bits than possible"
    for i in range(n * length):
        x = (a*x + c) % p
        truncated_x = (x >> bits_to_drop)
        array[i] = truncated_x

    reshaped_array = array.reshape(n, length)
    return reshaped_array



def pcg_rs(p: int = 2**16, length: int = 64, a: int = 45, c: int = 123, top_bits: int = 2, bits_to_keep: int = 8, min_shift: int = 0, num_examples: int | None = None) -> torch.Tensor:
    """
    Generates a random array of integers using the Permuted Congruential Generator - Random Shift (PCG-RS) algorithm.
    Args:
        p (int): The modulus value. Defaults to 2**16.
        length (int): The length of the generated array. Defaults to 64.
        a (int): The multiplier value. Defaults to 45.
        c (int): The increment value. Defaults to 123.
        top_bits (int): The number of top bits to shift. Defaults to 2.
        bits_to_keep (int): The number of bits to keep after shifting. Defaults to 8.
        min_shift (int): The minimum shift value. Defaults to 0.
        num_examples (int): The number of examples to generate. If None, it is calculated based on p and length. Defaults to None.
    Returns:
        torch.Tensor: A reshaped array of random integers with shape (n, length), where n is the number of examples.
    """
    
    if num_examples is not None:
        n = min(p // length, num_examples)
    else:
        n = p // length

    array = torch.zeros(n * length, dtype=torch.int64)
    x = np.random.randint(0, p)
    bit_length = np.ceil(np.log2(p)).astype(int)
    mask = (1 << bits_to_keep) - 1
    assert bit_length > (2**top_bits - 1) + bits_to_keep, "need more bits"
    top_shift = bit_length - top_bits
    for i in range(n * length):
        x = (a*x + c) % p
        shift = (x >> top_shift) + min_shift
        permuted_x = (x >> shift) & mask
        array[i] = permuted_x

    reshaped_array = array.reshape(n, length)
    return reshaped_array


def rotate_right(n, bits: int, width: int = 32):
    """Rotate the bits of n to the right by the specified number of bits."""
    return ((n >> bits) & (2**width - 1)) | (n << (width - bits) & (2**width - 1))

def rotate_left(n, bits: int, width: int = 32):
    """Rotate the bits of n to the left by the specified number of bits."""
    return ((n << bits) & (2**width - 1)) | (n >> (width - bits))



def pcg_rr(p: int = 2**16, length: int = 64, a: int = 45, c: int = 123, control_bits: int = 3, bits_to_keep: int = 8, num_examples: int | None = None) -> torch.Tensor:
    """
    Generates a random array of integers using the PCG-Random-Rotation algorithm.
    Args:
        p (int): The modulus value. Default is 2**16.
        length (int): The length of the output array. Default is 64.
        a (int): The multiplier value. Default is 45.
        c (int): The increment value. Default is 123.
        control_bits (int): The number of control bits used for rotation. Default is 3.
        bits_to_keep (int): The number of bits to keep after rotation. Default is 8.
        num_examples (int): The number of examples to generate. If None, it is calculated based on p and length.
    Returns:
        torch.Tensor: A tensor of shape (n, length) containing the generated random integers.
    """
    
    if num_examples is not None:
        n = min(p // length, num_examples)
    else:
        n = p // length

    array = torch.zeros(n * length, dtype=torch.int64)
    x = np.random.randint(0, p)
    bit_length = np.ceil(np.log2(p)).astype(int)
    mask = (1 << bits_to_keep) - 1
    assert bit_length > (2**control_bits - 1) + bits_to_keep, "cant do the permutation, need more bits"
    top_shift = bit_length - control_bits
    keep_shift = bit_length - control_bits - bits_to_keep
    for i in range(n * length):
        x = (a*x + c) % p
        rotation = (x >> top_shift)
        bits_kept = (x >> keep_shift) & mask
        rotated_x = rotate_right(bits_kept, rotation, width=bits_to_keep)
        array[i] = rotated_x

    reshaped_array = array.reshape(n, length)
    return reshaped_array



def pcg_xs(p: int=2**16,length: int=64, a: int=49, c: int=123, control_bits = 3, bits_to_keep = 8, num_examples: int | None = None) -> torch.Tensor:
    """
    Generates a pseudo-random number generator (PRNG) data using the PCG-XS algorithm.
    Args:
        p (int): The modulus value for the PRNG. Default is 2**16.
        length (int): The length of each PRNG sequence. Default is 64.
        a (int): The multiplier value for the PRNG. Default is 49.
        c (int): The increment value for the PRNG. Default is 123.
        control_bits (int): The number of bits used for control. Default is 3.
        bits_to_keep (int): The number of bits to keep in the PRNG output. Default is 8.
        num_examples (int): The number of PRNG sequences to generate. If None, it will be calculated based on p and length.
    Returns:
        torch.Tensor: A tensor containing the generated PRNG data with shape (num_examples, length).
    """
    
    if num_examples is not None:
        n = min(p // length, num_examples)
    else:
        n = p // length

    array = torch.zeros(n * length, dtype=torch.int64)
    x = np.random.randint(0, p)
    bit_length = np.ceil(np.log2(p)).astype(int)
    assert bit_length > (2**control_bits - 1) + bits_to_keep, "cant do the permutation, need more bits"
    top_shift = bit_length - control_bits
    mask = (1 << top_shift) - 1
    for i in range(n * length):
        x = (a*x + c) % p
        control = (x >> top_shift)
        targets = x & mask
        shift = control + 3
        t1 = targets ^ (targets >> shift)
        t2 = t1 >> (top_shift - bits_to_keep)
        padded_control = control << (bits_to_keep - control)
        array[i] = t2 ^ padded_control

    reshaped_array = array.reshape(n, length)
    return reshaped_array


def pcg_xsh_rr(p: int = 2**16, length: int = 64, a: int = 49, c: int = 123, control_bits: int = 3, bits_to_keep: int = 8, num_examples: int | None = 512) -> torch.Tensor:
    """
    Generates a pseudo-random number generator (PRNG) array using the PCG-XSH-RR algorithm.
    Args:
        p (int): The modulus value for the PRNG. Default is 2**16.
        length (int): The length of each PRNG array. Default is 64.
        a (int): The multiplier value for the PRNG. Default is 49.
        c (int): The increment value for the PRNG. Default is 123.
        control_bits (int): The number of control bits for the PRNG. Default is 3.
        bits_to_keep (int): The number of bits to keep for the PRNG. Default is 8.
        num_examples (int): The number of PRNG arrays to generate. Default is 512.
    Returns:
        torch.Tensor: A tensor containing the generated PRNG arrays.
    Raises:
        AssertionError: If the bit length of p is not greater than control_bits + bits_to_keep.
    """
    
    if num_examples is not None:
        n = min(p // length, num_examples)
    else:
        n = p // length

    array = torch.zeros(n * length, dtype=torch.int64)
    x = np.random.randint(0, p)
    bit_length = p.bit_length() - 1 ## p has to be 2 ** k
    assert bit_length > control_bits + bits_to_keep, "cant do the permutation, need more bits"
    shift = bits_to_keep - control_bits
    xor_shift = (bits_to_keep + control_bits)//2
    ratotation_control = bit_length - control_bits
    mask = (1 << bits_to_keep) - 1

    for i in range(n*length):
        x = (a*x + c) % p
        target = ((x ^ (x >> xor_shift)) >> shift) & mask
        rotation = (x >> ratotation_control) 
        array[i] = rotate_right(target, rotation, width=bits_to_keep)

    reshaped_array = array.reshape(n, length)
    return reshaped_array