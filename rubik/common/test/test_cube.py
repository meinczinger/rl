from rubik.common.src.face import Face
from rubik.common.src.cube import Cube
from rubik.common.src.enums import Color, Direction
import unittest


class CubeTest(unittest.TestCase):
    def test_default(self):
        cube = Cube(2)
        fc = cube.faces()[0]
        # First face should be red
        self.assertListEqual(
            cube.faces()[0].face(), [Color.RED, Color.RED, Color.RED, Color.RED]
        )


if __name__ == "__main__":
    unittest.main()
