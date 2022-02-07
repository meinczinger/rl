from rubik.common.src.enums import Color, Direction, SideType
from collections import deque


class Side:
    def __init__(self, side: SideType, dim: int, colors: list = [Color.RED] * 4):
        self._side = side
        self._dim = dim
        self._colors = deque(colors)

    def __copy__(self):
        return Side(self._side, self._dim, self._colors)

    def rotate(self, direction: Direction):
        if direction == Direction.CLOCKWISE:
            self._colors.appendleft(self._colors.pop())
        else:
            self._colors.append(self._colors.popleft())

    def colors(self) -> list:
        return [f for f in self._colors]

    def side(self) -> SideType:
        return self._side
    
    def setside(self, colors: list):
        self._side = deque(colors)
