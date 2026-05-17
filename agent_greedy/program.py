from math import inf
from referee.game import PlayerColor, Coord, Direction, CARDINAL_DIRECTIONS, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction
from referee.game.board import Board, GamePhase
from referee.game.constants import BOARD_N
from referee.game.exceptions import IllegalActionException


WIN_SCORE = 1_000_000


class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        self._color = color
        self._board = Board()

        match color:
            case PlayerColor.RED:
                print("Greedy: I am playing as RED")
            case PlayerColor.BLUE:
                print("Greedy: I am playing as BLUE")

    def action(self, **referee: dict) -> Action:
        legal_actions = self._legal_actions()
        if not legal_actions:
            raise RuntimeError("No legal actions found.")

        best_action = legal_actions[0]
        best_score = -inf

        for action in legal_actions:
            self._board.apply_action(action)
            score = self._evaluate()
            self._board.undo_action()

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        self._board.apply_action(action)

    def _evaluate(self) -> float:
        if self._board.game_over:
            if self._board.winner_color == self._color:
                return WIN_SCORE
            if self._board.winner_color == self._color.opponent:
                return -WIN_SCORE
            return 0

        my_tokens = self._count_tokens(self._color)
        opp_tokens = self._count_tokens(self._color.opponent)
        my_stacks = self._count_stacks(self._color)
        opp_stacks = self._count_stacks(self._color.opponent)
        my_eats = self._count_eat_actions(self._color)
        opp_eats = self._count_eat_actions(self._color.opponent)
        placement_score = self._placement_score(self._color) \
            - self._placement_score(self._color.opponent)

        return (
            1000 * (my_tokens - opp_tokens)
            + 100 * (my_stacks - opp_stacks)
            + 50  * (my_eats - opp_eats)
            + placement_score
        )

    def _legal_actions(self) -> list[Action]:
        actions: list[Action] = []
        color = self._board.turn_color
        opponent = color.opponent

        if self._board.phase == GamePhase.PLACEMENT:
            opponent_adjacent = set()
            for coord, cell in self._board._state.items():
                if cell.color == opponent:
                    for direction in CARDINAL_DIRECTIONS:
                        adj = self._try_add(coord, direction)
                        if adj is not None:
                            opponent_adjacent.add(adj)

            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    coord = Coord(r, c)
                    cell = self._board._state[coord]
                    if not cell.is_empty:
                        continue
                    if self._board.turn_count > 0 and coord in opponent_adjacent:
                        continue
                    actions.append(PlaceAction(coord))
            return actions

        for coord, cell in self._board._state.items():
            if cell.color != color:
                continue

            for direction in CARDINAL_DIRECTIONS:
                dest = self._try_add(coord, direction)
                if dest is None:
                    continue
                dest_cell = self._board._state[dest]

                if dest_cell.is_empty or dest_cell.color == color:
                    actions.append(MoveAction(coord, direction))

                if dest_cell.color == opponent and cell.height >= dest_cell.height:
                    actions.append(EatAction(coord, direction))

            if cell.height >= 2:
                for direction in CARDINAL_DIRECTIONS:
                    actions.append(CascadeAction(coord, direction))

        return actions

    def _count_tokens(self, color: PlayerColor) -> int:
        return sum(
            cell.height for cell in self._board._state.values()
            if cell.color == color
        )

    def _count_stacks(self, color: PlayerColor) -> int:
        return sum(
            1 for cell in self._board._state.values()
            if cell.color == color
        )

    def _count_eat_actions(self, color: PlayerColor) -> int:
        total = 0
        opponent = color.opponent
        for coord, cell in self._board._state.items():
            if cell.color != color:
                continue
            for direction in CARDINAL_DIRECTIONS:
                dest = self._try_add(coord, direction)
                if dest is None:
                    continue
                dest_cell = self._board._state[dest]
                if dest_cell.color == opponent and cell.height >= dest_cell.height:
                    total += 1
        return total

    def _placement_score(self, color: PlayerColor) -> int:
        score = 0
        for coord, cell in self._board._state.items():
            if cell.color != color:
                continue
            score -= 5 * self._distance_to_center(coord)
            if coord.r in (0, BOARD_N - 1) or coord.c in (0, BOARD_N - 1):
                score -= 10
        return score

    def _distance_to_center(self, coord: Coord) -> int:
        return min(
            abs(coord.r - r) + abs(coord.c - c)
            for r in (3, 4)
            for c in (3, 4)
        )

    def _try_add(self, coord: Coord, direction: Direction) -> Coord | None:
        try:
            return coord + direction
        except ValueError:
            return None