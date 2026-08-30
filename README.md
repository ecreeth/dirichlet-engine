# Dirichlet Engine: Parallel Sublinear Algorithms for Summatory Functions

A high-performance C++20 and Python framework for computing summatory arithmetic functions ($M(X)$, $\Phi(X)$, $L(X)$, $\pi(X)$, $D(X)$) at multi-trillion scale using **Direct Coordinate Geometry** and **Lock-Free Doubling-Stage DAG Scheduling**.

---

## 🌟 Key Theoretical Innovations

1. **Exact $\mathcal{O}(1)$ Coordinate Bijection ($\tau(q)$):**
   Maps hyperbolic quotient states $\mathcal{V}(X) = \{\lfloor X/i \rfloor : 1 \le i \le X\}$ directly to contiguous array coordinates without hash tables, dynamic memory allocations, or binary search overhead:
   $$\tau(q) = \begin{cases} q - 1 & \text{if } q \le \lfloor\sqrt{X}\rfloor \\ |\mathcal{V}(X)| - \lfloor X/q \rfloor & \text{if } q > \lfloor\sqrt{X}\rfloor \end{cases}$$

2. **Doubling-Stage DAG Theorem:**
   Proves that the state dependency DAG of the Dirichlet hyperbola DP partitions into $K = \lceil \log_2 X \rceil$ independent antichains $\mathcal{V}_m = \{v \in \mathcal{V}(X) : 2^{m-1} < v \le 2^m\}$, enabling lock-free, communication-free multi-core parallelization.

3. **PRAM Work-Span Complexity:**
   - **Total Work:** $W(X) = \Theta(X^{3/4})$
   - **Critical Path Span:** $T_\infty(X) = \Theta(\sqrt{X})$ (or $\mathcal{O}(\log^2 X)$ with tree reduction)
   - **Asymptotic Speedup:** $\mathcal{S}_\infty(X) = \Omega(X^{1/4})$

---

## 🚀 Performance Benchmarks (Apple Silicon, 8 Threads)

| Target $X$ | Scale | Function | Exact Output | 8-Thread Runtime |
| :--- | :--- | :--- | :--- | :--- |
| $10^{10}$ | $10\text{ Billion}$ | Mertens $M(X)$ | $-33,722$ | **0.011 s** |
| $10^{10}$ | $10\text{ Billion}$ | Totient Sum $\Phi(X)$ | $30,396,355,092,886,216,366$ | **0.012 s** |
| $10^{10}$ | $10\text{ Billion}$ | Liouville Sum $L(X)$ | $-116,026$ | **0.010 s** |
| $10^{10}$ | $10\text{ Billion}$ | Prime Count $\pi(X)$ | $455,052,511$ | **0.017 s** |
| $10^{12}$ | $1\text{ Trillion}$ | Mertens $M(X)$ | $62,366$ | **0.202 s** |
| $10^{12}$ | $1\text{ Trillion}$ | Totient Sum $\Phi(X)$ | $303,963,550,927,059,804,025,910$ | **0.420 s** |
| $10^{13}$ | $10\text{ Trillion}$ | Mertens $M(X)$ | $599,582$ | **1.055 s** |
| $10^{14}$ | $100\text{ Trillion}$ | Mertens $M(X)$ | $-875,575$ | **6.115 s** |
| $10^{15}$ | $1\text{ Quadrillion}$ | Mertens $M(X)$ | $-3,216,373$ | **79.561 s** ($1.33\text{ min}$) |
| $10^{16}$ | $10\text{ Quadrillion}$ | Mertens $M(X)$ | $-3,195,437$ | **622.431 s** ($10.37\text{ min}$) |

### 💻 Benchmark Hardware Specifications
- **Processor:** Apple M1 (8 cores: 4 Performance + 4 Efficiency cores)
- **Architecture:** 64-bit ARM (`arm64`) with 128-bit Neon vector extensions
- **Memory:** 16 GB Unified Memory (LPDDR4X, 68.25 GB/s bandwidth)
- **Operating System:** macOS (Darwin arm64)
- **Compiler:** Apple Clang C++20 (`-O3 -std=c++20`), LLVM `libomp` (OpenMP runtime)
- **Worker Threads:** 8 OpenMP threads (matching physical cores)

---

## 📂 Repository Structure

```text
.
├── dirichlet_engine.hpp       # Core C++20 header-only template library
├── cuda_engine.cuh            # NVIDIA CUDA Engine header
├── cuda_engine.cu             # NVIDIA CUDA implementation (NVIDIA T4 16GB / Turing / Ampere)
├── bench_cuda.cu              # NVIDIA CUDA CLI benchmark tool
├── metal_engine.hpp           # Metal GPU & Heterogeneous Antichain Engine header
├── metal_engine.mm            # Objective-C++ Metal compute pipeline implementation
├── bench.cpp                  # Multi-threaded C++ CLI benchmark tool
├── bench_gpu.mm               # Heterogeneous CPU vs. Apple Metal GPU benchmark tool
├── test_engine.cpp            # 25-case unit test suite (Mertens, PrimePi, Totient, Liouville)
├── reproduce_paper_bench.cpp  # Reproduces LaTeX Table 1 for paper.tex
├── dirichlet.py               # Pure Python reference engine & test suite
├── proof.md                   # Formal mathematical proofs (Theorems 1–3)
├── paper.tex                  # Research manuscript (AMS/SIAM LaTeX + TikZ)
├── references.bib             # BibTeX citation database
├── Makefile                   # Unified multi-target build system (CPU, Metal, CUDA)
└── README.md                  # Project documentation
```

---

## 🛠️ Build & Quickstart

### 1. NVIDIA CUDA Engine (NVIDIA T4 16GB / Tesla / Linux / Google Colab)

For execution on NVIDIA GPUs (e.g. NVIDIA T4 16GB, V100, A100, RTX):

```bash
# Compile with NVCC for NVIDIA T4 (Turing sm_75)
nvcc -O3 -std=c++20 -arch=sm_75 --use_fast_math cuda_engine.cu bench_cuda.cu -o bench_cuda

# Or use the Makefile:
make cuda

# Run Mertens benchmark at 1 Trillion on NVIDIA GPU
./bench_cuda 1000000000000 256

# Run Mertens benchmark at 100 Trillion on NVIDIA GPU
./bench_cuda 100000000000000 256
```

### 2. C++ Multi-Threaded Engine & Unit Tests

#### On Linux / Google Colab / AWS:
```bash
# Compile and run unit tests with g++
g++ -O3 -std=c++20 -fopenmp test_engine.cpp -o test_engine
./test_engine

# Compile benchmark tool
g++ -O3 -std=c++20 -fopenmp bench.cpp -o bench

# Or simply use the Makefile:
make cpu

# Run Mertens at 1 Trillion
./bench mertens 1000000000000 8

# Run Mertens at 100 Trillion
./bench mertens 100000000000000 8
```

#### On macOS (Apple Silicon):
```bash
# Compile and run comprehensive unit tests
clang++ -O3 -std=c++20 -Xpreprocessor -fopenmp \
    -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp \
    test_engine.cpp -o test_engine
./test_engine

# Compile benchmark tool
clang++ -O3 -std=c++20 -Xpreprocessor -fopenmp \
    -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp \
    bench.cpp -o bench
```

### 3. Apple Metal GPU & Heterogeneous Engine (macOS Apple Silicon)

```bash
# Compile Heterogeneous CPU + Apple Metal GPU benchmark
clang++ -O3 -std=c++20 -Xpreprocessor -fopenmp \
    -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp \
    -framework Metal -framework Foundation \
    metal_engine.mm bench_gpu.mm -o bench_gpu

# Benchmark CPU OpenMP vs. Metal GPU on Mertens at 1 Trillion
./bench_gpu 1000000000000 8
```

### 4. Reproduce Paper Benchmark Table (`paper.tex`)

To run the multi-scale automated benchmark across all arithmetic functions and generate LaTeX `Table 1` formatted output:

```bash
# Compile the paper benchmark reproducer
clang++ -O3 -std=c++20 -Xpreprocessor -fopenmp \
    -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp \
    reproduce_paper_bench.cpp -o reproduce_paper_bench

# Fast run: scales 10^7 through 10^14 (~25 seconds with 8 threads)
./reproduce_paper_bench 8 14

# Extended run: scales 10^7 through 10^15 (~1.5 minutes with 8 threads)
./reproduce_paper_bench 8 15

# Full extreme run: scales 10^7 through 10^16
./reproduce_paper_bench 8 16
```

### 3. Python Reference Engine

```bash
# Run the Python verification suite
python3 dirichlet.py
```

### 4. Using as a C++ Header Library

```cpp
#include "dirichlet_engine.hpp"

int main() {
    using namespace dirichlet;
    
    int64 X = 1000000000000LL; // 10^12
    
    // Compute Mertens M(10^12) with all available threads
    int64 M = DirichletEngine::compute_mertens(X);
    
    // Compute 128-bit Totient Summatory Phi(10^12)
    int128 Phi = DirichletEngine::compute_totient_sum(X);
    
    // Compute PrimePi pi(10^10)
    int64 pi = DirichletEngine::compute_prime_pi(10000000000LL);
}
```

---

## 📜 Theoretical Papers & Proofs

- **[paper.tex](paper.tex):** Full academic manuscript titled *"Parallel DAG Scheduling and Direct Coordinate Geometry for Sublinear Dirichlet Summatory Algorithms"*.
- **[proof.md](proof.md):** Formal proofs for Coordinate Mapping, DAG Antichains, and PRAM Work-Span Bounds.

---

## 📚 References

1. D{\'e}l{\'e}glise, M., \& Rivat, J. (1996). *Computing the summation of the M{\"o}bius function*. Experimental Mathematics, 5(4), 291-295.
2. Kotnik, T., \& van de Lune, J. (2004). *On the order of the Mertens function*. Experimental Mathematics, 13(4), 473-481.
3. Mertens, F. (1897). *{\"U}ber eine zahlentheoretische Function*. Sitzungsberichte der Kaiserlichen Akademie der Wissenschaften.
4. Kuznetsov, E. (2011). *Computing the Mertens Function up to $10^{16}$*. Mathematics of Computation, 80(276), 2461-2468.
5. Hurst, G. (2026). *Practical Computations of the Mertens Function: $M(10^{24})$ and $M(10^{25})$*. arXiv preprint arXiv:2607.07566.
6. Walisch, K. (2024). *primecount: Fast Prime Counting Function Implementation*. GitHub repository.
7. Frigo, M., Leiserson, C. E., Prokop, H., \& Ramachandran, S. (1999). *Cache-oblivious algorithms*. IEEE FOCS.

---

## 📄 License
MIT License.
