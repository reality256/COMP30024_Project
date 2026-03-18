# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade

from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction, PlayerColor, BOARD_N #included PlayerColor, BOARD_N
from .utils import render_board

# Make Board Hashable
BoardState = frozenset 
def board_to_state(board: dict[Coord, CellState]) -> BoardState:
    return frozenset((coord, cell.color, cell.height) for coord, cell in board.items())
def state_to_board(state:BoardState) -> dict[Coord, CellState]:
    return { coord: CellState(color, height) for coord, color, height in state}

#Define Goal
def is_goal(board: dict[Coord, CellState]) -> bool:
    return not any(cell.color == PlayerColor.BLUE for cell in board.values())

#Actions
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
        #Move
        if dest_cell is None:
            new_board = dict(board)
            del new_board[coord]
            new_board[dest] = cell
            results.append((MoveAction(coord, direction), new_board))

        elif dest_cell.color == PlayerColor.RED:
            new_board = dict(board)
            del new_board[coord]
            merged_height = cell.height + dest_cell.height
            new_board[dest] = CellState(PlayerColor.RED, merged_height)
            results.append((MoveAction(coord, direction), new_board))
        #EAT
        if dest_cell is not None and dest_cell.color == PlayerColor.BLUE:
            if cell.height >= dest_cell.height:
                new_board = dict(board)
                del new_board[coord]
                new_board[dest] = CellState(PlayerColor.RED, cell.height)
                results.append((EatAction(coord, direction), new_board))
        #CASCADE
        if cell.height >= 2:
            new_board = apply_cascade(board, coord, direction, cell.height)
            results.append((CascadeAction(coord, direction), new_board))
    return results

#Simulating Action Cascade 
def apply_cascade(
        board: dict[Coord, CellState], 
        coord: Coord, 
        direction: Direction, 
        height: int) -> dict[Coord, CellState]:
    new_board = dict(board)
    del new_board[coord]

    for i in range(1, height+1):
        nr = coord.r + direction.r *i
        nc = coord.c + direction.c *i

        if not (0<= nr < BOARD_N and 0 <= nc < BOARD_N):
            continue
        target = Coord(nr,nc)
        if target in new_board:
            new_board = push_stack(new_board, target, direction)

        new_board[target] = CellState(PlayerColor.RED, 1)
    return new_board

def push_stack(
        board: dict[Coord, CellState],
        coord: Coord,
        direction: Direction) -> dict[Coord, CellState]:
    stack = board[coord]
    nr = coord.r + direction.r
    nc = coord.c + direction.c 

    new_board = dict(board)
    del new_board[coord]

    if not (0<= nr < BOARD_N and 0 <= nc < BOARD_N):
            return new_board
    next_coord = Coord(nr, nc)

    if next_coord in new_board:
        new_board = push_stack(new_board, next_coord, direction)
    new_board[next_coord] = stack
    return new_board

#Search Method
def bfs(initial_board: dict[Coord, CellState]) -> list[Action] | None:
    from collections import deque
    if is_goal(initial_board):
        return []
    queue = deque([(initial_board, [])])
    visited: set[BoardState] = set()
    visited.add(board_to_state(initial_board))

    while queue:
        current_board, actions = queue.popleft()
        for action, next_board in get_possible_actions(current_board):
            state = board_to_state(next_board)

            if state in visited:
                continue
            visited.add(state)

            new_actions = actions + [action]
            if is_goal(next_board):
                return new_actions
            queue.append((next_board, new_actions))
    return None

def search(
    board: dict[Coord, CellState]
) -> list[Action] | None:
    """
    This is the entry point for your submission. You should modify this
    function to solve the search problem discussed in the Part A specification.
    See `core.py` for information on the types being used here.

    Parameters:
        `board`: a dictionary representing the initial board state, mapping
            coordinates to `CellState` instances (each with a `.color` and
            `.height` attribute).

    Returns:
        A list of actions (MoveAction, EatAction, or CascadeAction), or `None`
        if no solution is possible.
    """

    # The render_board() function is handy for debugging. It will print out a
    # board state in a human-readable format. If your terminal supports ANSI
    # codes, set the `ansi` flag to True to print a colour-coded version!
    print(render_board(board, ansi=True))

    # Do some impressive AI stuff here to find the solution...
    # ...
    # ... (your solution goes here!)
    # ...

    # Here we're returning "hardcoded" actions as an example of the expected
    # output format. Of course, you should instead return the result of your
    # search algorithm. Remember: if no solution is possible for a given input,
    # return `None` instead of a list.
    return bfs(board)
