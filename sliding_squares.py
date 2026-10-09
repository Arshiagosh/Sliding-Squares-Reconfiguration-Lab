"""
sliding_squares.py - a small sandbox for sliding-squares modular robot reconfiguration.

What it gives you (all on the integer grid, unit squares):
  * the sliding-squares move set (slide + convex transition) with the
    connectivity rule of the literature: the configuration minus the moving
    module must stay edge-connected;
  * articulation points (cut modules) via Tarjan's algorithm;
  * two lower bounds:
        - moves  >= EMD_L1(start, target) / 2        (each move shifts one module by L1 <= 2)
        - steps  >= bottleneck matching value         (parallel unit-speed motion)
  * a deliberately naive "greedy potential descent" compaction heuristic,
    to show WHY naive local strategies get stuck (the same lesson as local
    minima of potential fields);
  * plotting.

Run:   python sliding_squares.py           (prints a demo, saves demo PNGs)
Needs: numpy, scipy, matplotlib  (all in Anaconda)
"""

from __future__ import annotations

from collections import deque
import random

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

Cell = tuple[int, int]
DIRS: list[Cell] = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def add(a: Cell, b: Cell) -> Cell:
    return (a[0] + b[0], a[1] + b[1])


# --------------------------------------------------------------------------
# Connectivity
# --------------------------------------------------------------------------
def neighbours(c: Cell, cells: set[Cell]) -> list[Cell]:
    return [add(c, d) for d in DIRS if add(c, d) in cells]


def is_connected(cells: set[Cell]) -> bool:
    """Edge-connectivity (4-neighbourhood) of a set of grid cells, by BFS."""
    if not cells:
        return True
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        c = q.popleft()
        for n in neighbours(c, cells):
            if n not in seen:
                seen.add(n)
                q.append(n)
    return len(seen) == len(cells)


def articulation_points(cells: set[Cell]) -> set[Cell]:
    """Cut modules: removing one disconnects the configuration (iterative Tarjan)."""
    if len(cells) <= 2:
        return set()
    disc: dict[Cell, int] = {}
    low: dict[Cell, int] = {}
    parent: dict[Cell, Cell | None] = {}
    cut: set[Cell] = set()
    t = 0
    for root in cells:
        if root in disc:
            continue
        parent[root] = None
        disc[root] = low[root] = t
        t += 1
        root_children = 0
        stack = [(root, iter(neighbours(root, cells)))]
        while stack:
            v, it = stack[-1]
            advanced = False
            for w in it:
                if w not in disc:
                    parent[w] = v
                    disc[w] = low[w] = t
                    t += 1
                    if v == root:
                        root_children += 1
                    stack.append((w, iter(neighbours(w, cells))))
                    advanced = True
                    break
                elif w != parent[v]:
                    low[v] = min(low[v], disc[w])
            if not advanced:
                stack.pop()
                if stack:
                    u = stack[-1][0]
                    low[u] = min(low[u], low[v])
                    if parent[u] is not None and low[v] >= disc[u]:
                        cut.add(u)
        if root_children > 1:
            cut.add(root)
    return cut


# --------------------------------------------------------------------------
# Sliding-squares moves
# --------------------------------------------------------------------------
def valid_moves(cells: set[Cell]) -> list[tuple[Cell, Cell, str]]:
    """All (from, to, kind) moves of the sequential sliding-square model.

    slide:              c+e and c+d+e occupied, c+d empty            -> c moves to c+d
    convex transition:  c+e occupied, c+d and c+d+e empty            -> c moves to c+d+e
    plus: cells - {c} must stay connected (c is not a cut module).
    """
    cut = articulation_points(cells)
    moves = []
    for c in cells:
        if c in cut:
            continue
        for d in DIRS:
            cd = add(c, d)
            if cd in cells:
                continue
            for e in DIRS:
                if e[0] * d[0] + e[1] * d[1] != 0:  # e must be perpendicular to d
                    continue
                ce, cde = add(c, e), add(cd, e)
                if ce in cells and cde in cells:
                    moves.append((c, cd, "slide"))
                elif ce in cells and cde not in cells:
                    moves.append((c, cde, "convex"))
    # remove duplicates (same from/to reachable by two supports)
    return sorted(set(moves))


def apply_move(cells: set[Cell], move: tuple[Cell, Cell, str]) -> set[Cell]:
    a, b, _ = move
    new = set(cells)
    new.remove(a)
    new.add(b)
    assert is_connected(new), "move broke connectivity"
    return new


# --------------------------------------------------------------------------
# Measures and lower bounds
# --------------------------------------------------------------------------
def potential(cells: set[Cell]) -> int:
    """Sum of coordinates (the input-sensitive measure of the compaction papers)."""
    return sum(x + y for x, y in cells)


def l1_cost_matrix(A: list[Cell], B: list[Cell]) -> np.ndarray:
    A_, B_ = np.array(A), np.array(B)
    return np.abs(A_[:, None, :] - B_[None, :, :]).sum(axis=2)


def emd_lower_bound_moves(start: set[Cell], target: set[Cell]) -> int:
    """Unlabeled lower bound on #sequential moves: min-cost L1 matching / 2."""
    A, B = sorted(start), sorted(target)
    C = l1_cost_matrix(A, B)
    r, c = linear_sum_assignment(C)
    return int(np.ceil(C[r, c].sum() / 2))


def bottleneck_assignment(A: list[Cell], B: list[Cell]) -> tuple[int, list[tuple[Cell, Cell]]]:
    """Minimise the maximum L1 distance over perfect matchings A->B.

    Binary search over the sorted distinct distances; each test is a bipartite
    perfect-matching check. This value is a lower bound on the makespan of ANY
    parallel unit-speed schedule (unlabeled robots).
    """
    C = l1_cost_matrix(A, B)
    values = np.unique(C)
    lo, hi = 0, len(values) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        G = csr_matrix((C <= values[mid]).astype(int))
        match = maximum_bipartite_matching(G, perm_type="column")
        if (match >= 0).all():
            best = (values[mid], match)
            hi = mid - 1
        else:
            lo = mid + 1
    assert best is not None
    thr, match = best
    return int(thr), [(A[i], B[j]) for i, j in enumerate(match)]


# --------------------------------------------------------------------------
# A deliberately naive heuristic (to learn from its failure)
# --------------------------------------------------------------------------
def greedy_compaction(cells: set[Cell], max_moves: int = 10_000, seed: int = 0):
    """Repeatedly make a move that strictly lowers the potential, keeping x,y >= 0.

    Returns (final_cells, moves). Can get STUCK in a local minimum: that is the
    point. Compare with the papers, which add global structure (pillars,
    gathering, canonical shapes) to guarantee progress.
    """
    rng = random.Random(seed)
    history = []
    for _ in range(max_moves):
        candidates = [
            m for m in valid_moves(cells)
            if m[1][0] >= 0 and m[1][1] >= 0 and sum(m[1]) < sum(m[0])
        ]
        if not candidates:
            break
        best = min(sum(m[1]) - sum(m[0]) for m in candidates)
        m = rng.choice([m for m in candidates if sum(m[1]) - sum(m[0]) == best])
        cells = apply_move(cells, m)
        history.append(m)
    return cells, history


def is_compact(cells: set[Cell]) -> bool:
    """'Finished' in the compaction sense: every cell's origin-rectangle is full."""
    return all((i, j) in cells for (x, y) in cells for i in range(x + 1) for j in range(y + 1))


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot(cells: set[Cell], title: str = "", highlight: set[Cell] | None = None, ax=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4, 4))
    highlight = highlight or set()
    for (x, y) in cells:
        ax.add_patch(Rectangle((x, y), 1, 1, facecolor="#d1495b" if (x, y) in highlight else "#30638e",
                               edgecolor="white"))
    xs = [c[0] for c in cells] + [0]
    ys = [c[1] for c in cells] + [0]
    ax.set_xlim(min(xs) - 1, max(xs) + 2)
    ax.set_ylim(min(ys) - 1, max(ys) + 2)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3)
    ax.set_title(title, fontsize=9)
    return ax


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1) A horizontal line of n squares: the classic Omega(n^2) instance.
    n = 9
    line = {(i, 0) for i in range(n)}
    print(f"line of {n}: potential = {potential(line)}, cut modules = {len(articulation_points(line))}")
    block = {(i, j) for i in range(3) for j in range(3)}
    print(f"EMD lower bound on moves, line -> 3x3 block: {emd_lower_bound_moves(line, block)}")

    # 2) A Gamma-shape lifted off the floor: greedy descent stalls in a local minimum.
    gamma = {(0, y) for y in range(5)} | {(x, 4) for x in range(9)}
    final, hist = greedy_compaction(gamma)
    print(f"Gamma: greedy made {len(hist)} moves, potential {potential(gamma)} -> {potential(final)}, "
          f"compact={is_compact(final)}  <- stuck if False")

    # 3) How often does greedy get stuck on random connected shapes?
    rng = random.Random(3)
    stuck = trials = 0
    while trials < 300:
        cells = {(rng.randint(0, 5), rng.randint(0, 5))}
        size = rng.randint(6, 14)
        while len(cells) < size:
            c = rng.choice(sorted(cells))
            nc = add(c, rng.choice(DIRS))
            if nc[0] >= 0 and nc[1] >= 0:
                cells.add(nc)
        if is_compact(cells):
            continue
        trials += 1
        f, _ = greedy_compaction(cells)
        stuck += not is_compact(f)
    print(f"random shapes: greedy stuck in {stuck}/{trials} instances")

    # 4) Barrier formation: 8 robots scattered, target = horizontal wall at y=5.
    robots = rng.sample([(x, y) for x in range(10) for y in range(10) if y != 5], 8)
    wall = [(x, 5) for x in range(1, 9)]
    B, pairs = bottleneck_assignment(robots, wall)
    print(f"barrier: bottleneck matching = {B} grid steps (lower bound on parallel makespan)")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    plot(line, "line: cut modules in red", articulation_points(line), axes[0])
    plot(gamma, "Gamma shape (start)", articulation_points(gamma), axes[1])
    plot(final, f"after greedy: {len(hist)} moves, compact={is_compact(final)}", None, axes[2])
    fig.tight_layout()
    fig.savefig("demo_toolkit.png", dpi=150)
    print("saved demo_toolkit.png")
