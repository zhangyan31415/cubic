from typing import List, Optional, Tuple, Dict
import numpy as np

from .core.state import StateManager


FACE_ORDER = ['U', 'R', 'F', 'D', 'L', 'B']


class CubeState:
    """High-level cube state wrapper around StateManager with validation utilities."""

    def __init__(self):
        self.sm = StateManager()

    def reset(self):
        self.sm.reset()

    def update_face_by_label(self, face_label: str, labels_9: List[str], confidence: float = 1.0) -> bool:
        if face_label not in FACE_ORDER:
            return False
        idx = FACE_ORDER.index(face_label)
        return self.sm.update_face(idx, labels_9, confidence)

    def completeness(self) -> float:
        return self.sm.get_completeness()

    def face_completeness(self) -> List[float]:
        return self.sm.get_face_completeness()

    def build_state_string(self) -> Optional[str]:
        return self.sm.build_state_string()

    def is_counts_valid(self, state_string: str) -> Tuple[bool, Dict[str, int]]:
        return self.sm.check_counts(state_string)

    def is_solvable(self, state_string: str) -> bool:
        return self.sm.check_solvability(state_string)

    def get_stable_state(self) -> Optional[str]:
        return self.sm.get_stable_state()

