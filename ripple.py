__author__ = "Soumyakanti Das"

"""
    CSCI-630 Lab 1
    Author: Soumyakanti Das

    This program reads text file representations of a ripple puzzle and 
    solves it using brute force and constraint-satisfaction search.

"""

import turtle as t
from collections import deque
from time import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file", help="path to puzzle file")
parser.add_argument("-b", "--brute", help="Run brute force solver",
                    action="store_true")
parser.add_argument("-nf", "--not_fewest", help="Don't select successor with "
                                                "min remaining values",
                    action="store_false")
parser.add_argument("-ncp", "--no_constraint_prop", help="Do not Use constraint"
                                                         " propagation",
                    action="store_false")
parser.add_argument("-nv", "--no_visualization", help="Don't visualize final "
                                                      "result",
                    action="store_false")
args = parser.parse_args()


class Board:
    """
    Represents the ripple puzzle.
    """
    __slots__ = "cells", "shape", "regions", "count"

    def __init__(self, shape):
        """
        Initializes variables. Contains shape, cells and regions.

        :param shape: shape of the board.
        """
        self.shape = shape
        # Dictionary to store cells. cells[(i, j)] = Cell
        self.cells = dict()
        # Dictionary to store regions. regions[region] = True
        self.regions = dict()
        self.count = 0

    def is_full(self):
        return self.count == self.shape[0]*self.shape[1]


class Cell:
    """
    Represents a cell on the board.

    """
    __slots__ = "value", "region", "final", "legal_values", "pos"

    def __init__(self, value=0):
        """
        Initializes cell variables.

        :param value: value of the cell.
        """
        # position on board. (i, j)
        self.pos = None
        self.value = value
        # True if value is fixed from input.
        self.final = False
        # The region to which the cell belongs
        self.region = None
        # Remaining legal values.
        self.legal_values = deque()

    def __str__(self):
        """
        Overrides the str method for printing.

        :return: str
        """
        s = "value: {}, pos: {}, region: {}, legal-vals: {}"\
            .format(self.value, self.pos, self.region._id, self.legal_values)
        return s


class Region:
    """
    Represents a region on the board.
    """
    __slots__ = "_id", "members"

    # class variable to assign ids to the region.
    count = 0

    def __init__(self):
        """
        Initializes variables.
        """
        self._id = -1
        Region.count += 1
        # Dictionary of member cells. members[Cell] = (i, j)
        self.members = dict()

    def __str__(self):
        """
        Overrides str method for printing.

        :return: str
        """
        s = "id: {}, members: {}".format(self._id, list(self.members.values()))
        return s


def make_board(raw_board, board_shape):
    """
    This function takes in a string representation of the puzzle, along with
    shape of the puzzle and makes a Board object.

    :param raw_board: List of string representation of the board
    :param board_shape: shape of the board
    :return: Board
    """
    board = Board(board_shape)
    for i in range(0, 2*board_shape[0]-1):
        for j in range(1, 2*board_shape[1]):
            # Every alternate line contains numbers.
            if i % 2 == 1:
                continue

            if raw_board[i][j] not in ["|", " "]:
                _make_board(raw_board, board, board_shape, (i, j))
            else:
                continue

    return board


def _make_board(raw_board, board, shape, current, region=None):
    """
    Helper recursive function to create the Board.

    :param raw_board: string representation of the board.
    :param board: Board object
    :param shape: shape of the board
    :param current: current position in the raw_board (i, j)
    :param region: current region
    :return: None
    """
    i, j = current
    # if cell not created
    if not board.cells.get((i // 2, j // 2)):
        c = Cell()
        # If input contains value, fill it and mark as final
        if raw_board[i][j] != ".":
            c.value = int(raw_board[i][j])
            c.final = True
            board.count += 1

        # If region not allotted, create and allot one.
        if region is None:
            region = Region()
            region._id = Region.count

        # Update for current cell
        c.region = region
        c.pos = (i//2, j//2)
        region.members[c] = (i//2, j//2)
        board.cells[(i // 2, j // 2)] = c
        board.regions[region] = True

        # Call recursively to create cells to the north and south
        if i - 2 >= 0 and raw_board[i - 1][j] == " ":
            _make_board(raw_board, board, shape, (i-2, j), c.region)
        if i + 2 < 2*shape[0]-1 and raw_board[i+1][j] == " ":
            _make_board(raw_board, board, shape, (i+2, j), c.region)

        # Call recursively to create cells to the east and west
        if j-2 >= 0 and raw_board[i][j-1] == " ":
            _make_board(raw_board, board, shape, (i, j-2), c.region)
        if j+2 < 2*shape[1] and raw_board[i][j+1] == " ":
            _make_board(raw_board, board, shape, (i, j+2), c.region)


def read_puzzle(file):
    """
    Reads a puzzle from the given file.

    :param file: path to the puzzle file.
    :return: Board
    """
    board_shape = 0, 0
    raw_board = []
    with open(file) as f:
        for i, line in enumerate(f):
            line = line.strip("\n")
            if i == 0:
                board_shape = tuple(map(int, line.split(" ")))
                continue
            elif i in [1, 2*board_shape[0]+1]:
                continue
            line = line + " "*((2*board_shape[1]+1) - len(line))
            raw_board.append(list(line))

    return make_board(raw_board, board_shape)


def is_num_placement_legal(board, num, pos):
    """
    Given a board, number and position, checks if the number can be placed in
    the position legally.

    :param board: Board object
    :param num: number to place
    :param pos: position to place at
    :return: boolean
    """
    cell = board.cells[pos]
    region = cell.region
    size_region = len(region.members)

    # if cell.value == 0:
    #     return False

    # Final cells are always legal
    if cell.final:
        return True

    # region constraint
    if num > size_region or num < 1:
        return False
    # If any neighbour has the same value, return false
    for neighbour in region.members.values():
        if neighbour != pos and board.cells[neighbour].value == num:
            return False

    # row constraint
    for i in range(pos[0]-num, pos[0]+num+1):
        if not (0 <= i < board.shape[0]):
            if i < pos[0]:
                continue
            else:
                break

        if i != pos[0] and board.cells[(i, pos[1])].value == num:
            return False

    # col constraint
    for i in range(pos[1] - num, pos[1] + num + 1):
        if not (0 <= i < board.shape[1]):
            if i < pos[1]:
                continue
            else:
                break

        if i != pos[1] and board.cells[(pos[0], i)].value == num:
            return False

    return True


def is_goal(board):
    """
    Checks if every value on the board is legal.

    :param board: Board
    :return: boolean
    """
    if not board.is_full():
        return False

    result = True
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            pos = (i, j)
            num = board.cells[(i, j)].value
            result = result and is_num_placement_legal(board, num, pos)
            if not result:
                return False

    return result


def next_state(board, state):
    """
    Returns next state for the brute force solver.

    :param board: Board
    :param state: current state
    :return: next state -> (i, j)
    """
    if state[0] >= board.shape[0]:
        return None

    next_i, next_j = state
    # wrap around to the next row
    next_j = (next_j + 1) % board.shape[1]

    if next_j < state[1]:
        next_i += 1

    result = (next_i, next_j)
    if result[0] >= board.shape[0]:
        return None

    if board.cells[result].final:
        result = next_state(board, result)

    return result


def successors(board, state):
    """
    Returns a list of successors for the brute force solver.

    :param board: Board
    :param state: current state
    :return: list(successors)
    """
    s = []
    cell = board.cells[state]
    region = cell.region
    size_region = len(region.members)

    for num in range(1, size_region+1):
        if is_num_placement_legal(board, num, state):
            s.append((state, num, next_state(board, state)))

    return list(reversed(s))


def initialize_legal_values(board):
    """
    This method initializes all legal values to a cell ( 1 to region size)

    :param board: Board
    :return: None
    """
    final_cells = []
    for region in board.regions:
        size = len(region.members)
        for cell in region.members:
            if not cell.final:
                cell.legal_values = deque([x for x in range(1, size+1)])
            else:
                final_cells.append(cell.pos)

    # For final cells, do forward checking and remove its value from
    # neighboring cells
    for pos in final_cells:
        forward_checking(board, pos, board.cells[pos].value, {})


def successors_mrv(board, visited, few=True):
    """
    Returns successor for mrv solver.

    :param few: when true, it returns cell with minimum remaining value
    :param board: Board
    :param visited: visited dict
    :return: next state
    """
    min_cell = None

    ls = []
    for cell in board.cells.values():
        if cell.pos not in visited and not cell.final:
            ls.append(cell)

    if ls:
        if few:
            # Minimum based on legal values remaining
            min_cell = min(ls, key=lambda c: len(c.legal_values))
        else:
            min_cell = ls[0]

    return min_cell


def forward_checking(board, pos, num, visited):
    """
    For given board, number, position and visited dict, does removes num from
    neighboring cells.

    :param board: Board
    :param pos: board position
    :param num: num at pos
    :param visited: visited dict
    :return: result and all operations performed
    """
    operations = []
    region = board.cells[pos].region
    result = True

    # region constraint
    for neighbour in region.members.values():
        if not visited.get(neighbour, False) and \
                num in board.cells[neighbour].legal_values:
            board.cells[neighbour].legal_values.remove(num)
            if board.cells[neighbour].value == 0 and \
                    len(board.cells[neighbour].legal_values) == 0:
                result = False
            operations.append((neighbour, num))

    # row constraint
    for i in range(pos[0] - num, pos[0] + num + 1):
        if not (0 <= i < board.shape[0]):
            if i < pos[0]:
                continue
            else:
                break

        if not visited.get((i, pos[1]), False) and \
                num in board.cells[(i, pos[1])].legal_values:
            board.cells[(i, pos[1])].legal_values.remove(num)
            if board.cells[(i, pos[1])].value == 0 and \
                    len(board.cells[(i, pos[1])].legal_values) == 0:
                result = False
            operations.append(((i, pos[1]), num))

    # col constraint
    for i in range(pos[1] - num, pos[1] + num + 1):
        if not (0 <= i < board.shape[1]):
            if i < pos[1]:
                continue
            else:
                break

        if not visited.get((pos[0], i), False) and \
                num in board.cells[(pos[0], i)].legal_values:
            board.cells[(pos[0], i)].legal_values.remove(num)
            if board.cells[(pos[0], i)].value == 0 and \
                    len(board.cells[(pos[0], i)].legal_values) == 0:
                result = False
            operations.append(((pos[0], i), num))

    return result, operations


def dfs_mrv(board, calls=0):
    """
    This function solves the board using mrv.

    :param calls: number of calls to the function
    :param board: Board
    :return: boolean.
    """
    initialize_legal_values(board)

    return _dfs_mrv(board, calls)


def _dfs_mrv(board, calls, visited={}, found=False):
    """
    This function solves the puzzle recursively using backtracking with mrv and
    forward checking.

    :param calls: number of calls to the function
    :param board: Board
    :param visited: visited dict
    :param found: result
    :return: boolean. True if solution found, false otherwise
    """
    # check if goal state is reached
    if is_goal(board):
        for cell in board.cells.values():
            cell.legal_values = deque()
        return True, calls

    # get next successor
    state = successors_mrv(board, visited, few=args.not_fewest)

    # if no successor remaining, return False
    if not state:
        return False, calls

    legal_values = list(reversed(state.legal_values))

    # For all legal values, do forward checking and constraint propagation`
    for val in legal_values:
        if val not in state.legal_values:
            continue
        else:
            state.value = val
            board.count += 1

        # Forward checking results the operations performed during the process
        res, ops = forward_checking(board, state.pos, state.value, visited)
        # Apply constraint propagation. It also returns the operations performed
        if args.no_constraint_prop:
            ops = constraint_prop(board, state.pos, visited, ops)

        # Add the operations performed to the visited dictionary
        visited[state.pos] = ops

        # Call the function recursively with incremented number of calls
        found, calls = _dfs_mrv(board, calls+1, visited)

        # If goal not found or no remaining values for a cell, backtrack
        if not found:
            for pos, num in ops:
                board.cells[pos].legal_values.appendleft(num)
                board.cells[pos].value = 0
                if pos in visited:
                    del visited[pos]
                    board.count -= 1

    return found, calls


def constraint_prop(board, pos, visited, ops):
    """
    This function implements constraint propagation, i.e., it finds cells
    with only one remaining value in a region and assigns the value.

    :param board: Board
    :param pos: position of cell
    :param visited: visited dict
    :param ops: operations performed
    :return: list of operations performed
    """
    region = board.cells[pos].region

    for neighbour in region.members.values():
        cell = board.cells[neighbour]
        # Find cells in the region with only one remaining legal value
        if neighbour != pos and neighbour not in visited and \
                len(cell.legal_values) == 1:
            # Assign the value
            cell.value = cell.legal_values[0]
            board.count += 1
            # Do forward checking
            res, op = forward_checking(board, neighbour, cell.value, visited)
            # And extend the operations list
            ops.extend(op)
            visited[neighbour] = op
    return ops


def brute_dfs_iter(board, calls=0, state=(0, 0)):
    """
    This function implements the brute force solver iteratively.

    :param board: Board
    :param calls: number of calls
    :param state: starting state
    :return: (boolean, calls)
    """
    # Get the successors of the state
    stack = successors(board, state)

    while stack:
        calls += 1
        curr_state, num, _next = stack.pop()
        board.cells[curr_state].value = num
        board.count += 1

        # If goal state reached, return
        if is_goal(board):
            return True, calls

        # Extend stack with successors of the state
        s = successors(board, _next)
        stack.extend(s)

        # If no successor returned, backtrack with clean_up function
        if len(s) == 0:
            other_state = stack[-1][0] if stack else (0, 0)
            clean_up(board, curr_state, other_state)

    return False, calls


def clean_up(board, start, end):
    """
    Helps with resetting part of the board when wrong values were selected.

    :param board: Board
    :param start: starting state
    :param end: ending state
    :return: None
    """
    while start != end:
        if not board.cells[end].final:
            board.cells[end].value = 0
            board.count -= 1
        next_i, next_j = end
        next_j = (next_j + 1) % board.shape[1]

        if next_j < end[1]:
            next_i += 1

        end = next_i, next_j
    if not board.cells[end].final:
        board.cells[end].value = 0
        board.count -= 1


def draw_regions(board, cell_size):
    """
    Draws regions(BOLD) on the turtle board

    :param board: Board object
    :param cell_size: size of cell
    :return: None
    """
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            region_id = board.cells[(i, j)].region._id
            # top
            if i-1 >= 0 and board.cells[(i-1, j)].region._id != region_id:
                t.down()
                t.forward(cell_size)
                t.right(90)
            else:
                t.up()
                t.forward(cell_size)
                t.right(90)

            # right
            if j+1 < board.shape[1] and board.cells[(i, j+1)].region._id != \
                    region_id:
                t.down()
                t.forward(cell_size)
                t.right(90)
            else:
                t.up()
                t.forward(cell_size)
                t.right(90)

            # bottom
            if i+1 < board.shape[0] and board.cells[(i+1, j)].region._id != \
                    region_id:
                t.down()
                t.forward(cell_size)
                t.right(90)
            else:
                t.up()
                t.forward(cell_size)
                t.right(90)

            # left
            if j-1 >= 0 and board.cells[(i, j-1)].region._id != region_id:
                t.down()
                t.forward(cell_size)
                t.right(90)
            else:
                t.up()
                t.forward(cell_size)
                t.right(90)

            t.up()
            t.forward(cell_size)

        t.backward(board.shape[1] * cell_size)
        t.right(90)
        t.forward(cell_size)
        t.left(90)

    t.left(90)
    t.up()
    t.forward(board.shape[0] * cell_size)
    t.right(90)
    #########
    t.forward(cell_size/2)
    t.right(90)
    t.forward(cell_size * 0.9)
    t.left(90)
    fill_numbers(board, cell_size)


def fill_numbers(board, cell_size):
    """
    Fills result values in corresponding cells. Numbers given as input are
    underlined and bold

    :param board: Board
    :param cell_size: size of the cell
    :return: None
    """
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            val = board.cells[(i, j)].value
            if board.cells[(i, j)].final:
                t.write(val, align="center",
                        font=("Arial", int(cell_size / board.shape[0] * 2.25),
                              "normal", "bold", "underline"))
            elif val > 0:
                t.write(val, align="center",
                        font=("Arial", int(cell_size / board.shape[0] * 2.25),
                              "normal"))
            t.forward(cell_size)
        t.backward(cell_size * board.shape[1])
        t.right(90)
        t.forward(cell_size)
        t.left(90)


def display_board_turtle(board):
    """
    Draws the cells and calls other helper methods to draw regions and fill
    numbers.

    :param board: Board
    :return: None
    """
    cell_size = 100
    board_x, board_y = board.shape[1] * cell_size, board.shape[0] * cell_size
    screen_y = board_y + 200
    screen_x = board_x + 200

    t.setworldcoordinates(-screen_x/2, -screen_y/2, screen_x/2, screen_y/2)
    t.tracer(0, 0)
    t.up()
    t.color("gray")
    t.left(90)
    t.forward(board_y/2)
    t.right(90)
    t.backward(board_x/2)

    # Draw horizontal lines
    for i in range(board.shape[0]+1):
        t.down()
        t.forward(board_x)
        t.up()
        t.backward(board_x)
        if i == board.shape[0]:
            continue
        t.right(90)
        t.forward(cell_size)
        t.left(90)

    t.left(90)
    # Draw vertical lines
    for i in range(board.shape[1]+1):
        t.down()
        t.forward(board_y)
        t.up()
        t.backward(board_y)
        if i == board.shape[1]:
            continue
        t.right(90)
        t.forward(cell_size)
        t.left(90)

    t.color("black")
    t.pensize(3)

    t.down()
    t.forward(board_y)
    t.left(90)
    t.forward(board_x)
    t.left(90)
    t.forward(board_y)
    t.left(90)
    t.forward(board_x)
    t.left(90)
    t.up()
    t.forward(board_y)
    t.left(90)
    t.forward(board_x)
    t.left(180)

    draw_regions(board, cell_size)


def main():
    board = read_puzzle(args.file)
    t1 = time()
    if args.brute:
        result, calls = brute_dfs_iter(board)
    else:
        result, calls = dfs_mrv(board)
    del_t = time() - t1

    if result:
        print("Solution took {} calls and {} seconds.".format(
            calls, del_t
        ))
    else:
        print("Solution not found with {} calls and {} seconds.".format(
            calls, del_t
        ))
    if args.no_visualization:
        display_board_turtle(board)
        t.update()
        t.mainloop()


if __name__ == '__main__':
    main()
