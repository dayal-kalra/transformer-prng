import torch
import numpy as np


def first_repeat_index(sequence):
    seen = set()
    first_repeat_index = len(sequence)
    for i, s in enumerate(sequence):
        if s in seen:
            first_repeat_index = i
            break
        seen.add(s)
            
    return first_repeat_index

def find_period_vmap(dataset):
    """ 
    Performs the birthday test based on the repeats 
    dataset: (num_examples, context_length)
    """
    repeat_indices = torch.vmap(first_repeat_index)(dataset)
    return repeat_indices
        


