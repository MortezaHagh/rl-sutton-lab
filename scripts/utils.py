import numpy as np


def argmax_random_tie(values: np.ndarray) -> int:
    values = np.asarray(values)
    max_value = np.max(values)
    candidates = np.flatnonzero(values == max_value)
    try:
        return np.random.choice(candidates)
    except ValueError:
        print("Values:", values)
        print("Max value:", max_value)
        print("Candidates:", candidates)
        raise
