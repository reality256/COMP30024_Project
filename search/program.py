# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade

import heapq
import math
from .core import (
    CellState, Coord, Direction, Action,
    MoveAction, EatAction, CascadeAction,
    PlayerColor, BOARD_N,
)
from .utils import render_board

# state representation
BoardState = frozenset

def board_to_state(board: dict[Coord, CellState]) -> BoardState:
    return frozenset(
        (coord, cell.color, cell.height)
        for coord, cell in board.items()
    )


def state_to_board(state: BoardState) -> dict[Coord, CellState]:
    return {
        coord: CellState(color, height)
        for coord, color, height in state
    }

# goal test
def is_goal(board: dict[Coord, CellState]) -> bool:
    return not any(cell.color == PlayerColor.BLUE for cell in board.values())

# CASCADE 
def push_stack(
        board: dict[Coord, CellState],
        coord: Coord,
        direction: Direction) -> dict[Coord, CellState]:

    stack = board[coord]
    nr = coord.r + direction.r
    nc = coord.c + direction.c

    new_board = dict(board)
    del new_board[coord]

    if not (0 <= nr < BOARD_N and 0 <= nc < BOARD_N):
        return new_board  # outside of board

    next_coord = Coord(nr, nc)

    if next_coord in new_board:
        new_board = push_stack(new_board, next_coord, direction)  # push consecutive

    new_board[next_coord] = stack
    return new_board


def apply_cascade(
        board: dict[Coord, CellState],
        coord: Coord,
        direction: Direction,
        height: int) -> dict[Coord, CellState]:

    new_board = dict(board)
    del new_board[coord]  # eliminate previous stack

    for i in range(1, height + 1):
        nr = coord.r + direction.r * i
        nc = coord.c + direction.c * i

        if not (0 <= nr < BOARD_N and 0 <= nc < BOARD_N):
            continue  # outside of board

        target = Coord(nr, nc)

        if target in new_board:
            new_board = push_stack(new_board, target, direction)

        new_board[target] = CellState(PlayerColor.RED, 1)

    return new_board


# Actions
def get_possible_actions(board: dict[Coord, CellState]) -> list[tuple[Action, dict]]:

    results = []

    for coord, cell in list(board.items()):
        if cell.color != PlayerColor.RED:
            continue

        for direction in Direction:

            try:
                dest = coord + direction
            except ValueError:
                continue  

            dest_cell = board.get(dest)

            #MOVE 
            if dest_cell is None:
                new_board = dict(board)
                del new_board[coord]
                new_board[dest] = cell
                results.append((MoveAction(coord, direction), new_board))

            elif dest_cell.color == PlayerColor.RED:
                #Merge when alley
                new_board = dict(board)
                del new_board[coord]
                merged_height = cell.height + dest_cell.height
                new_board[dest] = CellState(PlayerColor.RED, merged_height)
                results.append((MoveAction(coord, direction), new_board))

            # EAT
            if dest_cell is not None and dest_cell.color == PlayerColor.BLUE:
                if cell.height >= dest_cell.height:
                    new_board = dict(board)
                    del new_board[coord]
                    #maintain height after EAT
                    new_board[dest] = CellState(PlayerColor.RED, cell.height)
                    results.append((EatAction(coord, direction), new_board))

            #CASCADE
            if cell.height >= 2:
                new_board = apply_cascade(board, coord, direction, cell.height)
                results.append((CascadeAction(coord, direction), new_board))

    return results

#heuristic function

def heuristic(board: dict[Coord, CellState]) -> int:

    blues = [
        (coord, cell.height)
        for coord, cell in board.items()
        if cell.color == PlayerColor.BLUE
    ]
    reds = [
        (coord, cell.height)
        for coord, cell in board.items()
        if cell.color == PlayerColor.RED
    ]

    if not blues:
        return 0      

    if not reds:
        return 10000  

    per_blue_costs = []
    for blue_coord, _ in blues:
        min_cost = min(
            math.ceil(
                (abs(red_coord.r - blue_coord.r) + abs(red_coord.c - blue_coord.c))
                / red_height
            ) if (abs(red_coord.r - blue_coord.r) + abs(red_coord.c - blue_coord.c)) > 0
            else 1
            for red_coord, red_height in reds
        )
        per_blue_costs.append(min_cost)

    return max(per_blue_costs)

#A* search
def astar(initial_board: dict[Coord, CellState]) -> list[Action] | None:

    if is_goal(initial_board):
        return []

    counter = 0
    h0 = heuristic(initial_board)
    open_set: list = [(h0, counter, 0, initial_board, [])]

    visited: set[BoardState] = set()
    visited.add(board_to_state(initial_board))

    while open_set:
        f, _, g, current_board, actions = heapq.heappop(open_set)

        if is_goal(current_board):
            return actions

        for action, next_board in get_possible_actions(current_board):
            state = board_to_state(next_board)

            if state in visited:
                continue
            visited.add(state)

            new_g = g + 1           
            new_h = heuristic(next_board)
            new_f = new_g + new_h

            counter += 1
            heapq.heappush(
                open_set,
                (new_f, counter, new_g, next_board, actions + [action])
            )

    return None  



def search(
    board: dict[Coord, CellState]) -> list[Action] | None:
    # print(render_board(board, ansi=True))

    return astar(board)