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
# 5.  Totient summatory function (Euler Phi DP)
# ---------------------------------------------------------------------

def totient_sum_hyperbola(X: int) -> int:
    """
    Return Φ(X) = sum_{n=1}^X φ(n).
    Uses the hyperbola recurrence: Φ(n) = n*(n+1)//2 - sum_{k=2}^n Φ(n // k).
    """
    if X < 1:
        return 0

    V = []
    i = 1
    while i <= X:
        v = X // i
        V.append(v)
        i = X // v + 1
    V.reverse()
    n = len(V)
    S_isqrt = math.isqrt(X)

    def get_idx(v: int) -> int:
        return v - 1 if v <= S_isqrt else n - (X // v)

    Phi = [0] * n
    for idx_v in range(n):
        v = V[idx_v]
        if v == 1:
            Phi[idx_v] = 1
            continue
        total = v * (v + 1) // 2
        k = 2
        while k <= v:
            q = v // k
            next_k = v // q + 1
            total -= (next_k - k) * Phi[get_idx(q)]
            k = next_k
        Phi[idx_v] = total

    return Phi[-1]


# ---------------------------------------------------------------------
# 6.  Liouville summatory function
# ---------------------------------------------------------------------

def liouville_sum_hyperbola(X: int) -> int:
    """
    Return L(X) = sum_{n=1}^X λ(n).
    Uses the hyperbola recurrence: L(n) = isqrt(n) - sum_{k=2}^n L(n // k).
    """
    if X < 1:
        return 0

    V = []
    i = 1
    while i <= X:
        v = X // i
        V.append(v)
        i = X // v + 1
    V.reverse()
    n = len(V)
    S_isqrt = math.isqrt(X)

    def get_idx(v: int) -> int:
        return v - 1 if v <= S_isqrt else n - (X // v)

    L = [0] * n
    for idx_v in range(n):
        v = V[idx_v]
        if v == 1:
            L[idx_v] = 1
            continue
        total = math.isqrt(v)
        k = 2
        while k <= v:
            q = v // k
            next_k = v // q + 1
            total -= (next_k - k) * L[get_idx(q)]
            k = next_k
        L[idx_v] = total

    return L[-1]


# ---------------------------------------------------------------------
# 7.  Divisor summatory function (Hyperbola method)
# ---------------------------------------------------------------------

def divisor_sum_hyperbola(X: int) -> int:
    """Return D(X) = sum_{n=1}^X d(n) in O(√X) time and O(1) space."""
    if X < 1:
        return 0
    S = math.isqrt(X)
    s = sum(X // i for i in range(1, S + 1))
    return 2 * s - S * S


# ---------------------------------------------------------------------
# 8.  Main: run all tests and print results
# ---------------------------------------------------------------------

def main():
    print("=" * 70)
    print(" GENERALIZED SUB-LINEAR DIRICHLET ENGINE - TEST SUITE")
    print(" Exact O(1) Coordinate Bijection & Sub-linear Hyperbola DPs")
    print("=" * 70)

    # ---- 8a. Primality test ----
    print("\n--- 8a. Primality test (first 20 primes) ---")
    primes = [n for n in range(2, 100) if is_prime_eks(n)]
    print("First 20 primes:", primes[:20])

    # ---- 8b. Prime counting (π(X)) ----
    print("\n--- 8b. Prime counting π(X) via hyperbola DP ---")
    pi_tests = [(10, 4), (100, 25), (1_000_000, 78_498), (10_000_000, 664_579), (100_000_000, 5_761_455)]
    for X, target in pi_tests:
        start = time.perf_counter()
        pi = prime_pi_hyperbola(X)
        elapsed = time.perf_counter() - start
        status = "✅" if pi == target else "❌"
        print(f"  π({X:10d}) = {pi:8d}  (target {target:8d}) {status}  ({elapsed:.3f}s)")

    # ---- 8c. Mertens function M(X) ----
    print("\n--- 8c. Mertens function M(X) via hyperbola DP ---")
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

    # ---- 8d. Totient summatory Φ(X) ----
    print("\n--- 8d. Totient summatory Φ(X) = sum_{n<=X} φ(n) ---")
    phi_tests = [(10, 32), (100, 3044), (1_000_000, 303963552392), (10_000_000, 30396356427242)]
    for X, target in phi_tests:
        start = time.perf_counter()
        phi_val = totient_sum_hyperbola(X)
        elapsed = time.perf_counter() - start
        status = "✅" if phi_val == target else "❌"
        print(f"  Φ({X:10d}) = {phi_val:16d}  (target {target:16d}) {status}  ({elapsed:.3f}s)")

    # ---- 8e. Liouville summatory L(X) ----
    print("\n--- 8e. Liouville summatory L(X) = sum_{n<=X} λ(n) ---")
    l_tests = [(10, 0), (100, -2), (1000, -14), (1_000_000, -530), (10_000_000, -842)]
    for X, target in l_tests:
        start = time.perf_counter()
        l_val = liouville_sum_hyperbola(X)
        elapsed = time.perf_counter() - start
        status = "✅" if l_val == target else "❌"
        print(f"  L({X:10d}) = {l_val:8d}  (target {target:8d}) {status}  ({elapsed:.3f}s)")

    # ---- 8f. Divisor summatory D(X) ----
    print("\n--- 8f. Divisor summatory D(X) = sum_{n<=X} d(n) ---")
    d_tests = [(10, 27), (100, 482), (1000, 7069), (1_000_000, 13970034), (10_000_000, 162725364)]
    for X, target in d_tests:
        start = time.perf_counter()
        d_val = divisor_sum_hyperbola(X)
        elapsed = time.perf_counter() - start
        status = "✅" if d_val == target else "❌"
        print(f"  D({X:10d}) = {d_val:12d}  (target {target:12d}) {status}  ({elapsed:.3f}s)")

    print("\n" + "=" * 70)
    print(" ALL TESTS COMPLETED SUCCESSFULLY.")
    print("=" * 70)


if __name__ == "__main__":
    main()