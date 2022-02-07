from enum import Enum


class Color(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2
    YELLOW = 3
    WHITE = 4
    ORANGE = 5


class Direction(Enum):
    CLOCKWISE = 1
    COUNTER_CLOCKWISE = -1


class SideType(Enum):
    TOP = 0
    LEFT = 1
    FRONT = 2
    RIGHT = 3
    BACK = 4
    BOTTOM = 5


class Position(Enum):
    LEFT_UP = 0
    RIGHT_UP = 1
    RIGHT_BOTTOM = 2
    LEFT_BOTTOM = 3
