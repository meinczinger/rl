from rubik.common.src.enums import Color, Direction, SideType
from rubik.common.src.side import Side
import unittest
from copy import copy


class TestFace(unittest.TestCase):
    def test_default(self):
        f = Side(SideType.TOP, 2)
        self.assertListEqual(f.colors(), [Color.RED] * 4)

    def test_rotation(self):
        f = Side(SideType.TOP, 2, [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE])
        # Check initial permutation
        self.assertListEqual(
            f.colors(), [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
        )
        f_rotated = copy(f)
        f_rotated.rotate(Direction.CLOCKWISE)
        # Check rotated state
        self.assertListEqual(
            f_rotated.colors(), [Color.WHITE, Color.RED, Color.BLUE, Color.ORANGE]
        )
        # Now rotate back
        f_rotated.rotate(Direction.COUNTER_CLOCKWISE)
        self.assertListEqual(f_rotated.colors(), f.colors())

    def test_cycle(self):
        f = Side(SideType.TOP, 2, [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE])
        f_rotated = copy(f)
        # rotate 4 times
        for _ in range(4):
            f_rotated.rotate(Direction.CLOCKWISE)
        self.assertListEqual(f.colors(), f_rotated.colors())


if __name__ == "__main__":
    unittest.main()
