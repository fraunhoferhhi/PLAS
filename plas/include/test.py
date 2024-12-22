def multiplicative_inverse(generator, n):
    n0, n1 = n, generator
    b = [0, 1, 0]
    c = 0
    i = 2
    while n1 > 1:
        c = n0 // n1
        r = n0 - c * n1
        n0, n1 = n1, r
        b[i % 3] = b[(i + 1) % 3] - c * b[(i + 2) % 3]
        i += 1
    return b[(i + 2) % 3]

import unittest

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

class TestMultiplicativeInverse(unittest.TestCase):
    def test_systematically(self):
        for n in range(2, 10000):
            for g in range(1, n):
                if gcd(g, n) != 1:
                    continue
                if multiplicative_inverse(g, n) * g % n != 1:
                    print(f"failed at {n} {g}")
                    print(multiplicative_inverse(g, n))
                    return

if __name__ == '__main__':
    unittest.main()