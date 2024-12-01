from enum import Enum
import sympy as sp


class VecShape(Enum):
    Col = 0
    Row = 1

    def __str__(self) -> str:
        return "Col" if self == VecShape.Col else "Row"

    def func_name(self, func_name: str, i) -> str:
        if self == VecShape.Col:
            return f"{func_name}_{i}_0"
        else:
            return f"{func_name}_0_{i}"


def get_vecshape(mat: sp.Matrix):
    r, c = mat.shape
    if r == 1 and c == 1:
        raise ValueError("Found (1, 1) matrix, please use scalar instead.")
    elif r == 1:
        return VecShape.Row
    elif c == 1:
        return VecShape.Col
    else:
        raise ValueError(f"Expects vector of shape (1, n) or (n, 1), found {mat.shape}")
