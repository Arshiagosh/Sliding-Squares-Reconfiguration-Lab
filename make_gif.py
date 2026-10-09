"""
make_gif.py - animate the greedy compaction heuristic on the Gamma shape.

Replays every move of greedy_compaction() one frame at a time, with cut
modules in red, until the descent gets stuck in a local minimum.

Run:   python make_gif.py            (saves media/greedy_gamma.gif)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from sliding_squares import (apply_move, articulation_points, greedy_compaction,
                             is_compact, plot, potential)

gamma = {(0, y) for y in range(5)} | {(x, 4) for x in range(9)}
final, history = greedy_compaction(gamma)

frames = [gamma]
for m in history:
    frames.append(apply_move(frames[-1], m))
frames += [final] * 6  # hold the last frame

fig, ax = plt.subplots(figsize=(5, 4))


def draw(k):
    ax.clear()
    cells = frames[k]
    step = min(k, len(history))
    title = f"greedy descent: move {step}/{len(history)}, potential {potential(cells)}"
    if k >= len(history):
        title += f"\nstuck, compact={is_compact(cells)}"
    plot(cells, title, articulation_points(cells), ax)
    ax.set_xlim(-1, 10)
    ax.set_ylim(-1, 6)


anim = FuncAnimation(fig, draw, frames=len(frames))
anim.save("media/greedy_gamma.gif", writer=PillowWriter(fps=2))
print(f"saved media/greedy_gamma.gif ({len(history)} moves)")
