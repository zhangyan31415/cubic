from typing import List, Optional

from .core.solver import RubikSolver


def move_to_cn(move: str) -> str:
    if not move:
        return ""
    face_map = {'U': '上', 'D': '下', 'L': '左', 'R': '右', 'F': '前', 'B': '后'}
    face = face_map.get(move[0], move[0])
    if len(move) > 1:
        if move[1] == "'":
            return f"{face}面逆时针90°"
        elif move[1] == '2':
            return f"{face}面180°"
    return f"{face}面顺时针90°"


class Solver:
    """Friendly wrapper around RubikSolver."""

    def __init__(self):
        self._solver = RubikSolver()

    def solve(self, state_string: str) -> Optional[List[str]]:
        return self._solver.solve(state_string)

    def next_move(self) -> Optional[str]:
        return self._solver.get_next_move()

    def advance(self):
        self._solver.advance_move()

    def remaining(self) -> List[str]:
        return self._solver.get_remaining_moves()

    def is_solved(self) -> bool:
        return self._solver.is_solved()

    def reset(self):
        self._solver.reset()

