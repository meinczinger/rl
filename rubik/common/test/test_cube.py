from rubik.common.src.side import Side
from rubik.common.src.cube import Cube
from rubik.common.src.enums import Color, Direction, SideType
import unittest


class CubeTest(unittest.TestCase):
    def test_default(self):
        cube = Cube(2)
        fc = cube.sides()[SideType.TOP]
        # First face should be red
        self.assertListEqual(
            cube.sides()[SideType.TOP].colors(),
            [Color.RED, Color.RED, Color.RED, Color.RED],
        )

    def test_rotate(self):
        cube = Cube(2)
        cube.rotate(SideType.TOP, Direction.CLOCKWISE)


if __name__ == "__main__":
    unittest.main()
