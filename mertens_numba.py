import time
import numpy as np
from numba import njit

@njit
def mertens_numba(X):
    if X < 1:
        return 0
    # Collect distinct values floor(X / i)
    V_desc = []
    i = 1
    while i <= X:
        v = X // i
        V_desc.append(v)
        i = X // v + 1
    V_asc = V_desc[::-1]
    n = len(V_asc)

    # Integer square root
    S = int(X ** 0.5)
    while (S + 1) * (S + 1) <= X:
        S += 1
    while S * S > X:
        S -= 1

    M = np.zeros(n, dtype=np.int64)
    for i in range(n):
        v = V_asc[i]
        if v == 1:
            M[i] = 1
            continue
        total = 1
        k = 2
        while k <= v:
            q = v // k
            next_k = v // q + 1
            idx = q - 1 if q <= S else n - (X // q)
            total -= (next_k - k) * M[idx]
            k = next_k
        M[i] = total
    return M[-1]

if __name__ == "__main__":
    X = 10**10
    print(f"Computing M({X}) with Numba...")
    # warm up
    mertens_numba(100)
    start = time.perf_counter()
    result = mertens_numba(X)
    elapsed = time.perf_counter() - start
    print(f"M({X}) = {result}")
    print(f"Time: {elapsed:.3f}s")
    print(f"|M|/sqrt(X) = {abs(result) / (X ** 0.5):.6f}")