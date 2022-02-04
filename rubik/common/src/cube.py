from rubik.common.src.enums import Color
from rubik.common.src.face import Face


class Cube:
    def __init__(self, dim: int):
        self._dim = dim
        self._faces = [Face(self._dim, [c] * 4) for c in Color]

    def faces(self) -> list:
        return self._faces
