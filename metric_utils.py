"""Compute phoneme/char edition errors."""

from collections import Counter
from typing import List

import Levenshtein as Lev


def compute_token_errors(refs: List[str], hyps: List[str]) -> Counter:
    """Compute the character error rate and return the edit distances.

    Args:
        hyps (List[str]): List of the hypotheses sequences (text)
        refs (List[str]): List of the reference sequences (text)
    """

    total_errors = {'insert': 0, 'replace': 0, 'delete': 0, 'token_errors': 0}

    for ref, hyp in zip(refs, hyps):
        seq_errors = Counter({'insert': 0, 'replace': 0, 'delete': 0, 'token_errors': 0})
        seq_errors.update(([e[0] for e in Lev.editops(ref, hyp)]))
        seq_errors['token_errors'] = sum(seq_errors.values())

        for editop, _ in total_errors.items():
            total_errors[editop] +=  seq_errors[editop]/len(ref)
    return total_errors

def compute_avg_token_errors(refs: List[str], hyps: List[str]) -> Counter:
    """Compute the character error rate and return the edit distances.

    Args:
        hyps (List[str]): List of the hypotheses sequences (text)
        refs (List[str]): List of the reference sequences (text)
    """

    total_errors = Counter({'insert': 0, 'replace': 0, 'delete': 0, 'token_errors': 0})
    nb_samples = 0

    for ref, hyp in zip(refs, hyps):
        total_errors.update(([e[0] for e in Lev.editops(ref, hyp)]))
        nb_samples += len(ref)

    total_errors['token_errors'] = sum(total_errors.values())
    for editop, _ in total_errors.items():
        total_errors[editop] /=  nb_samples
    return total_errors
