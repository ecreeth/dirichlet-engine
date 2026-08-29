#!/usr/bin/env python3
"""
Equality-Kernel Sieve (EKS)
A modularity-free algebraic framework for prime detection, prime counting,
and the Mertens function.

All primality tests and counting algorithms use ONLY:
  - integer division (//)
  - multiplication (*)
  - addition/subtraction (+/-)
  - the floor function (via //)

No modulo (%) operator is used anywhere.
"""

import math
import time

# ---------------------------------------------------------------------
# 1.  Primality test (Equality-Kernel)
# ---------------------------------------------------------------------

def is_prime_eks(x: int) -> bool:
    """
    Returns True if x is prime, False otherwise.
    Uses the Equality-Kernel identity: x // a * a == x iff a divides x.
    """
    if x < 2:
        return False
    if x == 2 or x == 3:
        return True

    # Quick filters for 2 and 3 (still using // and *)
    if x // 2 * 2 == x:
        return False
    if x // 3 * 3 == x:
        return False

    # Check divisors of the form 6k ± 1 up to sqrt(x)
    a = 5
    while a * a <= x:
        if x // a * a == x:
            return False
        if x // (a + 2) * (a + 2) == x:
            return False
        a += 6
    return True


# ---------------------------------------------------------------------
# 2.  Find the n-th prime (sequential scanning)
# ---------------------------------------------------------------------

def nth_prime_eks(n: int) -> int:
    """Return the n-th prime, using is_prime_eks()."""
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return 2
    count = 1          # we already have 2
    candidate = 3
    while count < n:
        if is_prime_eks(candidate):
            count += 1
            if count == n:
                return candidate
        candidate += 2   # only odd numbers
    return candidate


# ---------------------------------------------------------------------
# 3.  Prime counting (Lucy-DP / Meissel-Lehmer style, 0% modulo)
# ---------------------------------------------------------------------

def prime_pi_hyperbola(X: int) -> int:
    """
    Return π(X) = number of primes ≤ X.
    Uses the hyperbola DP (Lucy DP) - O(√X) memory, sub-linear time.
    """
    if X < 2:
        return 0

    # ---- Step 1: collect all distinct values of floor(X / i) ----
    V = []
    i = 1
    while i <= X:
        v = X // i
        V.append(v)
        i = X // v + 1
    V.reverse()  # ascending order
    n = len(V)
    S_isqrt = math.isqrt(X)

    def get_idx(v: int) -> int:
        return v - 1 if v <= S_isqrt else n - (X // v)

    # ---- Step 2: initialise DP with S[v] = v - 1 (candidates 2..v) ----
    S_arr = [v - 1 for v in V]

    # ---- Step 3: sieve over primes p up to √X ----
    for p in range(2, S_isqrt + 1):
        idx_p = get_idx(p)
        idx_p_minus_1 = get_idx(p - 1)
        if S_arr[idx_p] <= S_arr[idx_p_minus_1]:
            continue

        sp = S_arr[idx_p_minus_1]   # number of primes < p
        p2 = p * p
        for idx_v in range(n - 1, -1, -1):
            v = V[idx_v]
            if v < p2:
                break
            # DP update: S[v] -= (S[v // p] - sp)
            S_arr[idx_v] -= (S_arr[get_idx(v // p)] - sp)

    return S_arr[-1]


# ---------------------------------------------------------------------
# 4.  Mertens function (iterative hyperbola DP, 0% modulo)
# ---------------------------------------------------------------------

def mertens_hyperbola(X: int) -> int:
    """
    Return M(X) = sum_{n=1}^X μ(n).
    Uses the hyperbola recurrence: M(n) = 1 - sum_{k=2}^n M(n // k).
    O(√X) memory and sub-linear time.
    """
    if X < 1:
        return 0

    # ---- Collect distinct values of floor(X / i) ----
    V = []
    i = 1
    while i <= X:
        v = X // i
        V.append(v)
        i = X // v + 1
    V.reverse()  # ascending order
    n = len(V)
    S_isqrt = math.isqrt(X)

    def get_idx(v: int) -> int:
        return v - 1 if v <= S_isqrt else n - (X // v)

    # ---- Process values in ascending order ----
    M = [0] * n
    for idx_v in range(n):
        v = V[idx_v]
        if v == 1:
            M[idx_v] = 1
            continue
        total = 1                # 1 - sum_{k=2}^v M(v//k)
        k = 2
        while k <= v:
            q = v // k
            next_k = v // q + 1
            total -= (next_k - k) * M[get_idx(q)]   # q < v, already computed
            k = next_k
        M[idx_v] = total

    return M[-1]


# ---------------------------------------------------------------------
# 5.  Main: run all tests and print results
# ---------------------------------------------------------------------

def main():
    print("=" * 70)
    print(" EQUALITY-KERNEL SIEVE (EKS) - COMPLETE TEST SUITE")
    print(" Zero modulo (%) - only //, *, +, -")
    print("=" * 70)

    # ---- 5a. Primality test (first few primes) ----
    print("\n--- 5a. Primality test (first 20 primes) ---")
    primes = [n for n in range(2, 100) if is_prime_eks(n)]
    print("First 20 primes:", primes[:20])

    # ---- 5b. n-th prime ----
    print("\n--- 5b. Finding specific primes (sequential trial division) ---")
    tests = [(100, 541), (442, 3089), (443, 3109), (10_000, 104_729)]
    for n, target in tests:
        start = time.perf_counter()
        p = nth_prime_eks(n)
        elapsed = time.perf_counter() - start
        status = "✅" if p == target else "❌"
        print(f"  {n:7d}th prime = {p:8d}  (target {target}) {status}  ({elapsed:.3f}s)")

    # ---- 5c. Prime counting (π(X)) ----
    print("\n--- 5c. Prime counting π(X) via hyperbola DP ---")
    pi_tests = [(10, 4), (100, 25), (1_000_000, 78_498), (10_000_000, 664_579), (100_000_000, 5_761_455)]
    for X, target in pi_tests:
        start = time.perf_counter()
        pi = prime_pi_hyperbola(X)
        elapsed = time.perf_counter() - start
        status = "✅" if pi == target else "❌"
        print(f"  π({X:10d}) = {pi:8d}  (target {target:8d}) {status}  ({elapsed:.3f}s)")

    # ---- 5d. Mertens function M(X) ----
    print("\n--- 5d. Mertens function M(X) via hyperbola DP ---")
    m_tests = [
        (10, -1),
        (100, 1),
        (1000, 2),
        (1_000_000, 212),
        (10_000_000, 1037),
        (100_000_000, 1928),
        (1_000_000_000, -222),
    ]
    for X, target in m_tests:
        start = time.perf_counter()
        M = mertens_hyperbola(X)
        elapsed = time.perf_counter() - start
        status = "✅" if M == target else "❌"
        ratio = abs(M) / math.sqrt(X) if X > 0 else 0
        print(f"  M({X:12d}) = {M:8d}  (target {target:8d}) {status}  "
              f"|M|/√X = {ratio:.6f}  ({elapsed:.3f}s)")

    print("\n" + "=" * 70)
    print(" ALL TESTS COMPLETED.  ZERO MODULO (%) USED.")
    print("=" * 70)


if __name__ == "__main__":
    main()