# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade

from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction
from .utils import render_board

BOARD_N = 8
BoardState = frozenset


def board_to_state(board: dict[Coord, CellState]) -> BoardState:
    return frozenset((coord, cell.color, cell.height) for coord, cell in board.items())


def is_goal(board: dict[Coord, CellState]) -> bool:
    return not any(
        cell.color is not None and cell.color.name == "BLUE"
        for cell in board.values()
    )


def push_stack(
    board: dict[Coord, CellState],
    coord: Coord,
    direction: Direction,
) -> dict[Coord, CellState]:
    if coord not in board:
        return board

    stack_cell = board[coord]
    target_r = coord.r + direction.r
    target_c = coord.c + direction.c

    if not (0 <= target_r < BOARD_N and 0 <= target_c < BOARD_N):
        # Pushing off board removes the stack
        new_board = dict(board)
        del new_board[coord]
        return new_board

    target = Coord(target_r, target_c)
    new_board = dict(board)
    del new_board[coord]

    if target in new_board:
        new_board = push_stack(new_board, target, direction)

    new_board[target] = stack_cell
    return new_board


def apply_cascade(
    board: dict[Coord, CellState],
    coord: Coord,
    direction: Direction,
    height: int,
    source_color,
) -> dict[Coord, CellState]:
    new_board = dict(board)
    if coord in new_board:
        del new_board[coord]

    for step in range(1, height + 1):
        target_r = coord.r + direction.r * step
        target_c = coord.c + direction.c * step

        if not (0 <= target_r < BOARD_N and 0 <= target_c < BOARD_N):
            continue

        target = Coord(target_r, target_c)
        if target in new_board:
            new_board = push_stack(new_board, target, direction)

        new_board[target] = CellState(source_color, 1)

    return new_board


def get_possible_actions(board: dict[Coord, CellState]) -> list[tuple[Action, dict[Coord, CellState]]]:
    #Get all legal actions from the current board state.
    possible = []

    for coord, cell in list(board.items()):
        if cell.color is None or cell.color.name != "RED":
            continue

        for direction in Direction:
            try:
                dest = coord + direction
            except ValueError:
                continue

            if not (0 <= dest.r < BOARD_N and 0 <= dest.c < BOARD_N):
                continue

            dest_cell = board.get(dest)

            #Move
            if dest_cell is None:
                new_board = dict(board)
                del new_board[coord]
                new_board[dest] = cell
                possible.append((MoveAction(coord, direction), new_board))
            elif dest_cell.color is not None and dest_cell.color.name == "RED":
                new_board = dict(board)
                del new_board[coord]
                merged_height = cell.height + dest_cell.height
                new_board[dest] = CellState(cell.color, merged_height)
                possible.append((MoveAction(coord, direction), new_board))

            #Eat
            if (
                dest_cell is not None
                and dest_cell.color is not None
                and dest_cell.color.name == "BLUE"
                and cell.height >= dest_cell.height
            ):
                new_board = dict(board)
                del new_board[coord]
                new_board[dest] = CellState(cell.color, cell.height)
                possible.append((EatAction(coord, direction), new_board))

            #Cascade
            if cell.height >= 2:
                new_board = apply_cascade(board, coord, direction, cell.height, cell.color)
                possible.append((CascadeAction(coord, direction), new_board))

    return possible


def depth_limited_dfs(initial_board: dict[Coord, CellState], max_depth: int) -> list[Action] | None:
    #Depth-limited DFS with visited-state pruning.
    if is_goal(initial_board):
        return []

    stack = [(initial_board, [])]
    visited: set[BoardState] = {board_to_state(initial_board)}

    while stack:
        current_board, actions = stack.pop()

        if len(actions) > max_depth:
            continue

        if is_goal(current_board):
            return actions

        for action, next_board in get_possible_actions(current_board):
            state = board_to_state(next_board)
            if state in visited:
                continue
            visited.add(state)
            stack.append((next_board, actions + [action]))

    return None


def ids(initial_board: dict[Coord, CellState], max_depth: int = 200) -> list[Action] | None:
    for depth in range(max_depth + 1):
        result = depth_limited_dfs(initial_board, depth)
        if result is not None:
            return result
    return None


def search(
    board: dict[Coord, CellState]
) -> list[Action] | None:
    print(render_board(board, ansi=True))
    return ids(board)