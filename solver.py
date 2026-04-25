"""A* solver for the classic 8-puzzle.

Conventions:
- Values 0..7 represent puzzle tiles, 8 represents the empty slot.
- Goal state: ``[[0,1,2],[3,4,5],[6,7,8]]``.
- A move corresponds to the tile that slides into the empty slot, which is
  also the only tile the player has to click in the game.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable

GRID = 3
NUM_TILES = GRID * GRID
EMPTY_VALUE = NUM_TILES - 1
GOAL_STATE: tuple[int, ...] = tuple(range(NUM_TILES))

DIRECTION_NAMES = {
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
}


@dataclass(frozen=True)
class Move:
    """A single move from the player's perspective."""

    tile_value: int
    from_row: int
    from_col: int
    to_row: int
    to_col: int
    direction: str

    @property
    def direction_name(self) -> str:
        return DIRECTION_NAMES.get(self.direction, self.direction)

    def describe(self, step_index: int) -> str:
        return (
            f"Step {step_index}: click the tile at "
            f"(row {self.from_row + 1}, col {self.from_col + 1}) – "
            f"it slides {self.direction_name}."
        )


def flatten(state: list[list[int]]) -> tuple[int, ...]:
    return tuple(value for row in state for value in row)


def unflatten(state: Iterable[int]) -> list[list[int]]:
    flat = list(state)
    if len(flat) != NUM_TILES:
        raise ValueError(f"State must have {NUM_TILES} cells")
    return [flat[r * GRID : (r + 1) * GRID] for r in range(GRID)]


def is_solvable(state: list[list[int]]) -> bool:
    """3x3 inversion test: the puzzle is solvable iff inversions are even."""

    flat = [value for value in flatten(state) if value != EMPTY_VALUE]
    inversions = 0
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i] > flat[j]:
                inversions += 1
    return inversions % 2 == 0


def manhattan(state: tuple[int, ...]) -> int:
    distance = 0
    for index, value in enumerate(state):
        if value == EMPTY_VALUE:
            continue
        cur_row, cur_col = divmod(index, GRID)
        goal_row, goal_col = divmod(value, GRID)
        distance += abs(cur_row - goal_row) + abs(cur_col - goal_col)
    return distance


def _neighbors(state: tuple[int, ...]):
    empty_idx = state.index(EMPTY_VALUE)
    er, ec = divmod(empty_idx, GRID)

    deltas = (
        ("up", -1, 0),
        ("down", 1, 0),
        ("left", 0, -1),
        ("right", 0, 1),
    )

    for empty_dir, dr, dc in deltas:
        nr, nc = er + dr, ec + dc
        if not (0 <= nr < GRID and 0 <= nc < GRID):
            continue
        new_idx = nr * GRID + nc
        new_state = list(state)
        new_state[empty_idx], new_state[new_idx] = new_state[new_idx], new_state[empty_idx]

        tile_dir = {
            "up": "down",
            "down": "up",
            "left": "right",
            "right": "left",
        }[empty_dir]

        move = Move(
            tile_value=state[new_idx],
            from_row=nr,
            from_col=nc,
            to_row=er,
            to_col=ec,
            direction=tile_dir,
        )
        yield move, tuple(new_state)


def solve(state: list[list[int]]) -> list[Move]:
    """Solve the puzzle and return the optimal sequence of moves.

    Raises ``ValueError`` if the state is not solvable.
    """

    start = flatten(state)
    if start == GOAL_STATE:
        return []
    if not is_solvable(state):
        raise ValueError("Puzzle state is not solvable.")

    open_heap: list[tuple[int, int, tuple[int, ...]]] = []
    g_score: dict[tuple[int, ...], int] = {start: 0}
    came_from: dict[tuple[int, ...], tuple[tuple[int, ...], Move]] = {}

    counter = 0
    heapq.heappush(open_heap, (manhattan(start), counter, start))

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current == GOAL_STATE:
            return _reconstruct(came_from, current)

        current_g = g_score[current]
        for move, neighbor in _neighbors(current):
            tentative = current_g + 1
            if tentative < g_score.get(neighbor, 10**9):
                g_score[neighbor] = tentative
                came_from[neighbor] = (current, move)
                f_score = tentative + manhattan(neighbor)
                counter += 1
                heapq.heappush(open_heap, (f_score, counter, neighbor))

    raise RuntimeError("A* search failed to find a path (should not happen).")


def _reconstruct(
    came_from: dict[tuple[int, ...], tuple[tuple[int, ...], Move]],
    end: tuple[int, ...],
) -> list[Move]:
    moves: list[Move] = []
    current = end
    while current in came_from:
        prev, move = came_from[current]
        moves.append(move)
        current = prev
    moves.reverse()
    return moves


def apply_move(state: list[list[int]], move: Move) -> list[list[int]]:
    """Apply a move to the state and return a new copy."""

    new = [row.copy() for row in state]
    new[move.to_row][move.to_col] = move.tile_value
    new[move.from_row][move.from_col] = EMPTY_VALUE
    return new
