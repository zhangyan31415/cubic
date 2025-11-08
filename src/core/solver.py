"""
求解器模块：Kociemba两阶段求解
"""
import kociemba
from typing import List, Optional

class RubikSolver:
    """魔方求解器"""
    
    def __init__(self):
        self.current_solution = []
        self.current_state = None
    
    def solve(self, state_string: str) -> Optional[List[str]]:
        """
        求解魔方状态
        Args:
            state_string: 54字符的状态串
        Returns:
            移动序列列表，如 ['R', "U'", 'F2', ...]
        """
        try:
            solution = kociemba.solve(state_string)
            moves = solution.split()
            self.current_solution = moves
            self.current_state = state_string
            return moves
        except Exception as e:
            print(f"求解失败: {e}")
            return None
    
    def get_next_move(self) -> Optional[str]:
        """获取下一步移动"""
        if self.current_solution:
            return self.current_solution[0]
        return None
    
    def advance_move(self):
        """移动到下一步"""
        if self.current_solution:
            self.current_solution.pop(0)
    
    def get_remaining_moves(self) -> List[str]:
        """获取剩余移动"""
        return self.current_solution.copy()
    
    def is_solved(self) -> bool:
        """检查是否已解决"""
        return len(self.current_solution) == 0
    
    def reset(self):
        """重置求解器"""
        self.current_solution = []
        self.current_state = None

