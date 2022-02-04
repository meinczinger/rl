from rubik.common.src.enums import Color, Direction
from rubik.common.src.face import Face
import unittest
from copy import copy


class TestFace(unittest.TestCase):
    def test_default(self):
        f = Face(2)
        self.assertListEqual(f.face(), [Color.RED] * 4)

    def test_rotation(self):
        f = Face(2, [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE])
        # Check initial permutation
        self.assertListEqual(
            f.face(), [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE]
        )
        f_rotated = copy(f)
        f_rotated.rotate(Direction.CLOCKWISE)
        # Check rotated state
        self.assertListEqual(
            f_rotated.face(), [Color.WHITE, Color.RED, Color.BLUE, Color.ORANGE]
        )
        # Now rotate back
        f_rotated.rotate(Direction.COUNTER_CLOCKWISE)
        self.assertListEqual(f_rotated.face(), f.face())

    def test_cycle(self):
        f = Face(2, [Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE])
        f_rotated = copy(f)
        # rotate 4 times
        for _ in range(4):
            f_rotated.rotate(Direction.CLOCKWISE)
        self.assertListEqual(f.face(), f_rotated.face())


if __name__ == "__main__":
    unittest.main()
