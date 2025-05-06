
import torch
from sympy import primefactors, gcd
from itertools import *
import numpy as np

def find_as(p, limit=None, rng=np.random.default_rng(97), num=2000):
    """Find multiplicative factors for LCG that maintain period length.
    
    Args:
        p (int): Modulus for LCG
        limit (int, optional): Upper bound for search. Defaults to p.
        rng (numpy.random.Generator): Random number generator. Defaults to seeded RNG.
        num (int): Number of factors to find. Defaults to 2000.
    
    Returns:
        list: Valid multiplicative factors for LCG (without duplicates)
    """
    if limit is None:
        limit = p
    factors = primefactors(p)

    # Handle special case for modulus divisible by 4
    if p % 4 == 0:
        factors.append(4)
        factors.remove(2)
    
    # Calculate least common multiple of prime factors
    lcm = 1
    for factor in set(factors):
        lcm *= factor
    
    if limit//lcm <= 1:
        return []
    
    # Adjust num to available range
    available_nums = limit // lcm - 1  # -1 because range starts from 1
    if num > available_nums:
        # print(f"Requested {num} a values for p={p} but only {available_nums} are available")
        num = available_nums

    # Generate unique random multipliers
    k_list = rng.choice(range(1, limit // lcm), size=num, replace=False)

    # Calculate valid factors
    result = [k * lcm + 1 for k in k_list]
    return result


def find_coprimes(p, rng=np.random.default_rng(97), num=2000, low=3, search=int(2e4)):
    """Find numbers coprime to p.
    
    Args:
        p (int): Number to find coprimes for
        rng (numpy.random.Generator): Random number generator. Defaults to seeded RNG.
        num (int): Number of coprimes to find. Defaults to 2000.
        low (int): Lower bound for search. Defaults to 3.
        search (int): Upper bound for search. Defaults to 20000.
    
    Returns:
        list: Numbers coprime to m (without duplicates)
    """
    assert num < search, "search range is too small"
    search = min(p-low, search)
    possible_numbers = rng.integers(low=low, high=p, size=search)
    num = min(len(possible_numbers), num)
    coprimes = set()  # Changed to set to ensure uniqueness
    
    for i in possible_numbers:
        if gcd(i, p) == 1 and i not in coprimes:  # Check if not already in set
            coprimes.add(i)
            if len(coprimes) == num:
                break
            
    if len(coprimes) < num:
        print("not enough coprimes found for ", p) 
    return list(coprimes)  # Convert back to list before returning


def lcg(p: int = 512, length: int = 8, a: int = 45, c: int = 123, 
        rng=np.random.default_rng(97), num_examples: int = None):
    """Generate sequences using Linear Congruential Generator (LCG).
    
    Args:
        p (int): Modulus for LCG. Defaults to 512.
        length (int): Length of each sequence. Defaults to 8.
        a (int): Multiplier for LCG. Defaults to 45.
        c (int): Increment for LCG. Defaults to 123.
        rng (numpy.random.Generator): Random number generator. Defaults to seeded RNG.
        num_examples (int, optional): Number of sequences to generate.
    
    Returns:
        torch.Tensor: Generated sequences
    """
    if num_examples:
        n = min(p // length, num_examples)
    else:
        n = p // length

    array = torch.zeros(n * length, dtype=torch.int64)
    array[0] = rng.integers(low=0, high=p)  # Initial seed

    for i in range(1, n * length):
        array[i] = (a * array[i - 1] + c) % p

    reshaped_array = array.reshape(n, length)
    return reshaped_array


def decimal_to_base_b_reverse(n, b, length):
    """Convert decimal number to base-b representation in reverse order.
    
    Args:
        n (int): Decimal number to convert
        b (int): Base to convert to
        length (int): Desired length of output
    
    Returns:
        list: Digits in base-b representation (reversed)
    """
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    while len(digits) < length:
        digits.append(0)
    return digits


def base_b_lcg(base: int=16, digits: int=2, p: int=None, length: int=8, 
               a: int=45, c: int=123, rng=np.random.default_rng(97), num_examples: int=None):
    """Generate sequences using base-b Linear Congruential Generator.
    
    Args:
        base (int): Base for number representation. Defaults to 16.
        digits (int): Number of digits in base-b representation. Defaults to 2.
        p (int, optional): Modulus for LCG. Defaults to base^digits.
        length (int): Length of each sequence. Defaults to 8.
        a (int): Multiplier for LCG. Defaults to 45.
        c (int): Increment for LCG. Defaults to 123.
        rng (numpy.random.Generator): Random number generator. Defaults to seeded RNG.
        num_examples (int, optional): Number of sequences to generate.
    
    Returns:
        torch.Tensor: Generated sequences in base-b representation
    """
    if p is None:
        p = base ** digits
    
    if num_examples:
        n = max(min(p // length, num_examples), 1)
    else:
        n = max(p // length, 1)

    array = np.zeros(digits * n * length)
    x = int(rng.integers(low=0, high=p))
    a = int(a)
    c = int(c)  
    
    for i in range(0, digits * n * length, digits):
        x = (a * x + c) % p
        array[i:i+digits] = decimal_to_base_b_reverse(x, base, digits)

    reshaped_array = array.reshape(n, digits * length)
    return torch.tensor(reshaped_array, dtype=torch.int64)

