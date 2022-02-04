from rubik.common.src.enums import Color, Direction
from collections import deque


class Face:
    def __init__(self, dim: int, colors: list = [Color.RED] * 4):
        self._dim = dim
        self._face = deque(colors)

    def __copy__(self):
        return Face(self._dim, self._face)

    def rotate(self, direction: Direction):
        if direction == Direction.CLOCKWISE:
            self._face.appendleft(self._face.pop())
        else:
            self._face.append(self._face.popleft())

    def face(self) -> list:
        return [f for f in self._face]
