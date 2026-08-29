import numpy as np
from numba import njit, types
from numba.typed import Dict
import time

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
    # Build typed dict for index lookup
    idx_map = Dict.empty(key_type=types.int64, value_type=types.int64)
    for idx, v in enumerate(V_asc):
        idx_map[v] = idx
    M = np.zeros(len(V_asc), dtype=np.int64)
    for i, v in enumerate(V_asc):
        if v == 1:
            M[i] = 1
            continue
        total = 1
        k = 2
        while k <= v:
            q = v // k
            next_k = v // q + 1
            total -= (next_k - k) * M[idx_map[q]]
            k = next_k
        M[i] = total
    return M[-1]

# Run it
X = 10**12
print(f"Computing M({X}) with Numba...")
start = time.perf_counter()
result = mertens_numba(X)
elapsed = time.perf_counter() - start
print(f"M({X}) = {result}")
print(f"Time: {elapsed:.2f}s")
print(f"|M|/sqrt(X) = {abs(result) / (X ** 0.5):.6f}")