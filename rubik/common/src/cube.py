from ctypes.wintypes import tagMSG
from rubik.common.src.enums import Color, Direction, SideType, Position
from rubik.common.src.side import Side
from copy import copy, deepcopy


class Cube:
    def __init__(self, dim: int):
        self._dim = dim
        self._sides = {s: Side(s, self._dim, [c] * 4) for s, c in zip(SideType, Color)}
        self._dependencies = {
            SideType.TOP: {
                "impacts": [
                    SideType.LEFT,
                    SideType.FRONT,
                    SideType.RIGHT,
                    SideType.BACK,
                ],
                "changes_position": [Position.LEFT_UP, Position.RIGHT_UP],
            },
            SideType.LEFT: {
                "impacts": [
                    SideType.TOP,
                    SideType.FRONT,
                    SideType.BACK,
                    SideType.BOTTOM,
                ],
                "changes_position": [Position.LEFT_UP, Position.RIGHT_BOTTOM],
            },
        }

    def sides(self) -> dict:
        return self._sides

    def rotate(self, side: SideType, dir: Direction):
        colors = [self.sides()[s].colors() for s in self._dependencies[side]["impacts"]]

        cp = self._dependencies[side]["changes_position"]

        target_colors = deepcopy(colors)
        for i in range(len(colors)):
            for cp1 in cp:
                if dir == Direction.CLOCKWISE:
                    target_colors[i][cp1.value] = colors[(i - 1) % 4][cp1.value]
                else:
                    target_colors[i][cp1.value] = colors[(i + 1) % 4][cp1.value]
