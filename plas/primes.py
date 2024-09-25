import numpy as np
import os

def get_primes_up_to(n:int):
    """Return the first n primes, taken from a precomputed list of primes in primes.txt.

    Args:
        n (int): Number of primes to return.

    Raises:
        ValueError: If n is greater than the number of precomputed primes.

    Returns:
        np.ndarray: Numpy array of int32 with the first n primes.
    """
    primes = []
    with open(os.path.join(os.path.dirname(__file__), "primes.txt"), "r") as f:
        max_number_of_primes = int(f.readline())
        if n > max_number_of_primes:
            raise ValueError(f"Only {max_number_of_primes} primes are precomputed. Requested {n}.")
        for i in range(n):
            p = int(f.readline())
            if p > n:
                break
            primes.append(int(f.readline()))
    return np.array(primes, dtype=np.int64)
    