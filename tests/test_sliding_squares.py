import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sliding_squares import (articulation_points, bottleneck_assignment, emd_lower_bound_moves,
                             greedy_compaction, is_compact, is_connected, l1_cost_matrix,
                             valid_moves)

GAMMA = {(0, y) for y in range(5)} | {(x, 4) for x in range(9)}


def test_line_has_n_minus_2_cut_modules():
    for n in range(3, 12):
        line = {(i, 0) for i in range(n)}
        assert articulation_points(line) == {(i, 0) for i in range(1, n - 1)}


def test_block_has_no_cut_modules():
    block = {(i, j) for i in range(3) for j in range(3)}
    assert articulation_points(block) == set()


def test_every_valid_move_keeps_the_shape_connected():
    for cells in (GAMMA, {(i, 0) for i in range(6)}, {(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)}):
        for a, b, _ in valid_moves(cells):
            assert is_connected((cells - {a}) | {b})


def test_emd_bound_line_to_block():
    line = {(i, 0) for i in range(9)}
    block = {(i, j) for i in range(3) for j in range(3)}
    assert emd_lower_bound_moves(line, block) == 18


def test_bottleneck_is_at_most_the_min_sum_matching_maximum():
    A = [(0, 0), (3, 7), (8, 2), (5, 9), (1, 4)]
    B = [(x, 5) for x in range(1, 6)]
    value, pairs = bottleneck_assignment(A, B)
    C = l1_cost_matrix(A, B)
    r, c = linear_sum_assignment(C)
    assert value <= C[r, c].max()
    assert max(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b in pairs) == value
    assert sorted(b for _, b in pairs) == sorted(B)


def test_greedy_gets_stuck_on_gamma():
    final, history = greedy_compaction(GAMMA)
    assert len(final) == len(GAMMA)
    assert is_connected(final)
    assert not is_compact(final)
