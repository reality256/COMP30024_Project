import random
from referee.game import PlayerColor, Coord, Direction, Action, CARDINAL_DIRECTIONS, \
    PlaceAction, MoveAction, EatAction, CascadeAction
from referee.game.board import Board, GamePhase
from referee.game.constants import BOARD_N


def get_legal_actions(board: Board, color: PlayerColor) -> list[Action]:
    actions = []
    opponent = color.opponent

    if board.phase == GamePhase.PLACEMENT:
        for r in range(BOARD_N):
            for c in range(BOARD_N):
                coord = Coord(r, c)
                try:
                    board.apply_action(PlaceAction(coord))
                    board.undo_action()
                    actions.append(PlaceAction(coord))
                except Exception:
                    pass
        return actions

    # Play
    for r in range(BOARD_N):
        for c in range(BOARD_N):
            coord = Coord(r, c)
            cell = board[coord]
            if cell.color != color:
                continue

            for direction in CARDINAL_DIRECTIONS:
                dest_r = coord.r + direction.r
                dest_c = coord.c + direction.c
                if not (0 <= dest_r < BOARD_N and 0 <= dest_c < BOARD_N):
                    continue
                dest = board[Coord(dest_r, dest_c)]

                # MOVE
                if dest.is_empty or dest.color == color:
                    actions.append(MoveAction(coord, direction))

                # EAT
                if dest.color == opponent and cell.height >= dest.height:
                    actions.append(EatAction(coord, direction))

            # CASCADE
            if cell.height >= 2:
                for direction in CARDINAL_DIRECTIONS:
                    actions.append(CascadeAction(coord, direction))

    return actions


class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        self._color = color
        self._board = Board()

    def action(self, **referee: dict) -> Action:
        actions = get_legal_actions(self._board, self._color)
        if not actions:
            raise Exception("No legal actions available")
        return random.choice(actions)

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        self._board.apply_action(action)