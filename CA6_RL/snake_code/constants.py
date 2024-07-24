from enum import Enum

SNAKE_1_Q_TABLE = "./snake_code/s1_qtble.npy"
SNAKE_2_Q_TABLE = "./snake_code/s2_qtble.npy"

STATE_SIZE = 2 ** 16

WIDTH = 500
HEIGHT = 500

ROWS = 20

GRID_SIZE = 20
ACTIONS_SIZE = 4

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
