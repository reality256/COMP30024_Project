# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from math import inf
from referee.game import PlayerColor, Coord, Direction, CARDINAL_DIRECTIONS, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction
from referee.game.board import Board, GamePhase
from referee.game.constants import BOARD_N
from referee.game.exceptions import IllegalActionException


WIN_SCORE = 1_000_000
SEARCH_DEPTH = 4
TIME_BUFFER = 5.0 


class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        self._color = color
        self._board = Board()

        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED (first player)")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")

    def action(self, **referee: dict) -> Action:
        legal_actions = self._legal_actions()
        if not legal_actions:
            raise RuntimeError("No legal actions found.")

        best_action = legal_actions[0]

        # Iterative Deepening
        time_remaining = referee.get("time_remaining", None)
        max_depth = SEARCH_DEPTH

        if time_remaining is not None:
            if time_remaining < 20:
                max_depth = 2
            elif time_remaining < 60:
                max_depth = 3
            else:
                max_depth = SEARCH_DEPTH

        for depth in range(1, max_depth + 1):
            best_score = -inf
            candidate = legal_actions[0]

            for action in self._ordered_actions(legal_actions):
                self._board.apply_action(action)
                score = self._alphabeta(depth - 1, -inf, inf)
                self._board.undo_action()

                if score > best_score:
                    best_score = score
                    candidate = action

            best_action = candidate
            # no further search when able to win
            if best_score >= WIN_SCORE:
                break

        return best_action

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        self._board.apply_action(action)

    def _alphabeta(self, depth: int, alpha: float, beta: float) -> float:
        if depth == 0 or self._board.game_over:
            return self._evaluate()

        legal_actions = self._legal_actions()
        if not legal_actions:
            return self._evaluate()

        if self._board.turn_color == self._color:
            value = -inf
            for action in self._ordered_actions(legal_actions):
                self._board.apply_action(action)
                value = max(value, self._alphabeta(depth - 1, alpha, beta))
                self._board.undo_action()
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = inf
        for action in self._ordered_actions(legal_actions):
            self._board.apply_action(action)
            value = min(value, self._alphabeta(depth - 1, alpha, beta))
            self._board.undo_action()
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

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
        my_cascade = self._count_cascade_power(self._color)
        opp_cascade = self._count_cascade_power(self._color.opponent)
        placement_score = self._placement_score(self._color) \
            - self._placement_score(self._color.opponent)

        return (
            1000 * (my_tokens - opp_tokens)
            + 100  * (my_stacks - opp_stacks)
            + 50   * (my_eats - opp_eats)
            + 30   * (my_cascade - opp_cascade)  # CASCADE included
            + placement_score
        )

    def _legal_actions(self) -> list[Action]:
        actions: list[Action] = []
        color = self._board.turn_color
        opponent = color.opponent

        # Placement phase: excluded cells that are adjacent to opponent stack
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

        # Play phase: check rules without apply or undo
        for coord, cell in self._board._state.items():
            if cell.color != color:
                continue

            for direction in CARDINAL_DIRECTIONS:
                dest = self._try_add(coord, direction)
                if dest is None:
                    continue
                dest_cell = self._board._state[dest]

                # MOVE
                if dest_cell.is_empty or dest_cell.color == color:
                    actions.append(MoveAction(coord, direction))

                # EAT
                if dest_cell.color == opponent and cell.height >= dest_cell.height:
                    actions.append(EatAction(coord, direction))

            # CASCADE
            if cell.height >= 2:
                for direction in CARDINAL_DIRECTIONS:
                    actions.append(CascadeAction(coord, direction))

        return actions

    def _ordered_actions(self, actions: list[Action]) -> list[Action]:
        return sorted(actions, key=self._action_priority, reverse=True)

    def _action_priority(self, action: Action) -> int:
        match action:
            case EatAction(coord, direction):
                target = self._try_add(coord, direction)
                if target is None:
                    return 100
                return 100 + self._board._state[target].height
            case CascadeAction():
                return 50
            case PlaceAction(coord):
                return 20 - self._distance_to_center(coord)
            case MoveAction():
                return 10
            case _:
                return 0

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

    def _count_cascade_power(self, color: PlayerColor) -> int:
        # Calculate pushable enemy stack 
        total = 0
        opponent = color.opponent
        for coord, cell in self._board._state.items():
            if cell.color != color or cell.height < 2:
                continue
            for direction in CARDINAL_DIRECTIONS:
                for i in range(1, cell.height + 1):
                    target_r = coord.r + direction.r * i
                    target_c = coord.c + direction.c * i
                    if not (0 <= target_r < BOARD_N and 0 <= target_c < BOARD_N):
                        break
                    target_cell = self._board._state[Coord(target_r, target_c)]
                    if target_cell.color == opponent:
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