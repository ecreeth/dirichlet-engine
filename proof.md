# Mathematical Proofs: Parallel DAG Scheduling & Coordinate Geometry for Sublinear Dirichlet Summatory DPs

---

## 1. The Hyperbola Projection Space $\mathcal{V}(X)$ and Coordinate Bijection

### 1.1 Definition and Cardinality
Let $X \in \mathbb{N}_{\ge 1}$. Define the hyperbola state projection space:
$$\mathcal{V}(X) \triangleq \left\{ \left\lfloor \frac{X}{i} \right\rfloor : 1 \le i \le X \right\}$$

Let $S = \lfloor \sqrt{X} \rfloor$.

**Lemma 1 (Explicit Structure of $\mathcal{V}(X)$).**  
The set $\mathcal{V}(X)$ consists of exactly:
1. All integers $1, 2, \dots, S$, and
2. All integers $\lfloor X/i \rfloor$ for $i = 1, 2, \dots, \lfloor X/(S+1) \rfloor$.

Furthermore, when sorted in strictly ascending order:
$$\mathcal{V}(X) = \left( v_0, v_1, \dots, v_{N-1} \right)$$
the cardinality $N = |\mathcal{V}(X)|$ satisfies:
$$N = S + \left\lfloor \frac{X}{S} \right\rfloor - \left[ S = \left\lfloor \frac{X}{S} \right\rfloor \right] = 2\lfloor\sqrt{X}\rfloor + \mathcal{O}(1)$$

*Proof.*  
For $i > \lfloor X/(S+1) \rfloor$, the quotient $q = \lfloor X/i \rfloor \le S$. For each integer $k \in \{1, 2, \dots, S\}$, setting $i = \lfloor X/k \rfloor$ yields $\lfloor X / \lfloor X/k \rfloor \rfloor = k$. Thus, every integer $1 \le k \le S$ is present in $\mathcal{V}(X)$.  
For $i \le S$, the values $\lfloor X/i \rfloor$ are strictly decreasing with $i$, since:
$$\frac{X}{i} - \frac{X}{i+1} = \frac{X}{i(i+1)} \ge \frac{X}{S(S+1)} \ge \frac{X}{\sqrt{X}(\sqrt{X}+1)} = \frac{\sqrt{X}}{\sqrt{X}+1} > 0$$
When $i \le \sqrt{X}-1$, the difference exceeds $1$, ensuring each $\lfloor X/i \rfloor$ is a distinct integer $> S$. The number of elements $> S$ is precisely $\lfloor X / (S+1) \rfloor = \lfloor X/S \rfloor - [S = \lfloor X/S \rfloor]$. Adding the $S$ elements $\le S$ yields the cardinality $N$. $\blacksquare$

---

### 1.2 The Exact $O(1)$ Coordinate Bijection $\tau$

**Theorem 1 (Direct Coordinate Mapping).**  
Let $\tau: \mathcal{V}(X) \to \{0, 1, \dots, N-1\}$ be defined by:
$$\tau(q) \triangleq \begin{cases} q - 1 & \text{if } q \le S \\ N - \left\lfloor \frac{X}{q} \right\rfloor & \text{if } q > S \end{cases}$$
Then $\tau$ is an exact, order-preserving bijection from $\mathcal{V}(X)$ to $\{0, 1, \dots, N-1\}$, computable in $\mathcal{O}(1)$ elementary arithmetic operations without division tables, memory allocations, or hash collisions.

*Proof.*  
1. **Case $q \le S$:** The first $S$ elements of $\mathcal{V}(X)$ in ascending order are $v_j = j + 1$ for $j \in \{0, \dots, S-1\}$. Hence $v_j = q \iff j = q - 1$.
2. **Case $q > S$:** By Lemma 1, the elements of $\mathcal{V}(X)$ strictly greater than $S$ are $u_k = \lfloor X/k \rfloor$ for $k = \lfloor X/(S+1) \rfloor, \dots, 2, 1$. In ascending order, $u_k$ is the $(N - k)$-th element (0-indexed). Setting $k = \lfloor X/q \rfloor$, we obtain $\tau(q) = N - k = N - \lfloor X/q \rfloor$.
3. Since both branches are strictly increasing on their domains and the range spans $\{0, \dots, S-1\} \cup \{S, \dots, N-1\} = \{0, \dots, N-1\}$, $\tau$ is a bijection. $\blacksquare$

---

## 2. General Dirichlet Hyperbola Summatory Recurrence

Let $f, g: \mathbb{N} \to \mathbb{C}$ be arithmetic functions with Dirichlet convolution $h = f * g$.  
Let $H(x) = \sum_{n \le x} h(n)$, $F(x) = \sum_{n \le x} f(n)$, and $G(x) = \sum_{n \le x} g(n)$.

**Lemma 2 (Hyperbola Summatory Identity).**  
$$\sum_{n \le x} (f * g)(n) = \sum_{k \le x} f(k) G\left(\left\lfloor \frac{x}{k} \right\rfloor\right)$$

*Proof.*  
$$\sum_{n \le x} \sum_{d \mid n} f(d) g(n/d) = \sum_{d \le x} f(d) \sum_{m \le x/d} g(m) = \sum_{d \le x} f(d) G\left(\left\lfloor \frac{x}{d} \right\rfloor\right). \quad \blacksquare$$

When $f(1) \neq 0$, separating the $k=1$ term yields the standard sublinear recurrence for $G(x)$:
$$G(x) = \frac{1}{f(1)} \left( \sum_{n \le x} (f * g)(n) - \sum_{k=2}^x f(k) G\left(\left\lfloor \frac{x}{k} \right\rfloor\right) \right)$$

---

## 3. The Doubling-Stage DAG Decomposition Theorem

Consider the dependency graph $\mathcal{G}_{\mathcal{V}} = (\mathcal{V}(X), \mathcal{E})$, where a directed edge $(q, v) \in \mathcal{E}$ indicates that evaluating $G(v)$ requires the value $G(q)$.

**Theorem 2 (Topological Depth and Stage Independence).**  
For any $v \in \mathcal{V}(X)$ and any $k \ge 2$, the required sub-state $q = \lfloor v/k \rfloor$ satisfies:
$$q \le \left\lfloor \frac{v}{2} \right\rfloor$$
Consequently, let $K = \lceil \log_2 X \rceil$. Partition $\mathcal{V}(X)$ into $K$ disjoint subsets:
$$\mathcal{V}_m \triangleq \left\{ v \in \mathcal{V}(X) : 2^{m-1} < v \le 2^m \right\}, \quad 1 \le m \le K$$
Then:
1. For every $v \in \mathcal{V}_m$, all dependencies $q = \lfloor v/k \rfloor$ ($k \ge 2$) satisfy:
   $$q \in \bigcup_{j=1}^{m-1} \mathcal{V}_j$$
2. The subgraph induced on each stage $\mathcal{V}_m$ contains **zero edges** ($\mathcal{E} \cap (\mathcal{V}_m \times \mathcal{V}_m) = \emptyset$).
3. All states in $\mathcal{V}_m$ can be evaluated concurrently in parallel without locks, atomic operations, or inter-thread communication.

*Proof.*  
1. For any $v \in \mathcal{V}_m$, $v \le 2^m$. For any $k \ge 2$, $q = \lfloor v/k \rfloor \le \lfloor 2^m / 2 \rfloor = 2^{m-1}$.
2. By definition, $\bigcup_{j=1}^{m-1} \mathcal{V}_j = \{u \in \mathcal{V}(X) : u \le 2^{m-1}\}$. Thus, $q \in \bigcup_{j=1}^{m-1} \mathcal{V}_j$.
3. Since $q \le 2^{m-1} < v$, no state in $\mathcal{V}_m$ depends on any other state in $\mathcal{V}_m$. Each stage is an antichain in the dependency poset, establishing complete parallel independence. $\blacksquare$

---

## 4. Work-Span (PRAM) Complexity Analysis

Let $W(X)$ denote the total computational work (number of elementary operations) and $T_\infty(X)$ denote the critical path length (span) on an unbounded parallel PRAM machine.

**Theorem 3 (Asymptotic Work and Span Bounds).**  
1. **Total Work:**
   $$W(X) = \Theta\left(X^{3/4}\right)$$
2. **Critical Path Span (Sequential Inner Grouping):**
   $$T_\infty(X) = \Theta\left(X^{1/2}\right)$$
3. **Critical Path Span (Tree-Reduced Inner Grouping):**
   $$T_\infty(X) = \mathcal{O}\left(\log^2 X\right)$$
4. **Theoretical Maximum Parallel Speedup:**
   $$\mathcal{S}_\infty(X) = \frac{W(X)}{T_\infty(X)} = \Omega\left(X^{1/4}\right)$$

*Proof.*  
1. **Work Bound:** Evaluating $G(v)$ using quotient grouping requires $\Theta(\sqrt{v})$ operations. Summing over all $v \in \mathcal{V}(X)$:
   $$W(X) = \sum_{v \in \mathcal{V}(X)} 2\sqrt{v} = \sum_{i=1}^{\lfloor\sqrt{X}\rfloor} 2\sqrt{i} + \sum_{i=1}^{\lfloor\sqrt{X}\rfloor} 2\sqrt{\left\lfloor \frac{X}{i} \right\rfloor}$$
   Applying Euler-Maclaurin summation:
   $$\sum_{i=1}^{\sqrt{X}} \sqrt{i} = \frac{2}{3} X^{3/4} + \mathcal{O}(\sqrt{X})$$
   $$\sum_{i=1}^{\sqrt{X}} \sqrt{\frac{X}{i}} = \sqrt{X} \sum_{i=1}^{\sqrt{X}} i^{-1/2} = \sqrt{X} \cdot \left( 2 X^{1/4} + \mathcal{O}(1) \right) = 2 X^{3/4} + \mathcal{O}(\sqrt{X})$$
   Thus $W(X) = \frac{8}{3} X^{3/4} + \mathcal{O}(\sqrt{X}) = \Theta(X^{3/4})$.

2. **Span with Sequential Inner Grouping:**  
   The critical path is the sum of inner loop execution times along the longest chain from stage $1$ to $K$:
   $$T_\infty(X) = \sum_{m=1}^K 2\sqrt{\max(\mathcal{V}_m)} \le \sum_{m=1}^K 2\sqrt{2^m} = 2\sqrt{X} \sum_{j=0}^\infty 2^{-j/2} = \frac{2\sqrt{2}}{\sqrt{2}-1} \sqrt{X} = \Theta(\sqrt{X})$$

3. **Span with Tree Reduction:**  
   If the $2\sqrt{v}$ terms in the inner sum for a state $v$ are summed via a balanced parallel reduction tree of depth $\mathcal{O}(\log \sqrt{v}) = \mathcal{O}(\log v)$, the span across $K = \lceil \log_2 X \rceil$ stages is:
   $$T_\infty(X) = \sum_{m=1}^{\lceil \log_2 X \rceil} \mathcal{O}(\log(2^m)) = \sum_{m=1}^{\lceil \log_2 X \rceil} \mathcal{O}(m) = \mathcal{O}\left(K^2\right) = \mathcal{O}\left(\log^2 X\right)$$

4. **Speedup:**  
   $$\mathcal{S}_\infty(X) = \frac{W(X)}{T_\infty(X)} = \frac{\Theta(X^{3/4})}{\Theta(\sqrt{X})} = \Omega\left(X^{1/4}\right) \quad \left(\text{or } \Omega\left(\frac{X^{3/4}}{\log^2 X}\right) \text{ with tree reduction}\right). \quad \blacksquare$$

---