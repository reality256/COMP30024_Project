# agent_weak/program.py
# 팀원 에이전트와 동일하되 탐색 깊이만 낮춘 버전
from math import inf
from time import process_time
from referee.game import PlayerColor, Coord, Direction, CARDINAL_DIRECTIONS, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction
from referee.game.board import Board, GamePhase
from referee.game.constants import BOARD_N, MAX_TURNS, PLACEMENT_TURNS
from referee.game.exceptions import IllegalActionException

WIN_SCORE = 1_000_000
SEARCH_DEPTH = 2  # 낮은 탐색 깊이


class Agent:
    def __init__(self, color: PlayerColor, **referee: dict):
        self._color = color
        self._board = Board()

    def action(self, **referee: dict) -> Action:
        legal_actions = self._legal_actions()
        if not legal_actions:
            raise RuntimeError("No legal actions found.")

        best_action = legal_actions[0]
        best_score = -inf

        for action in self._ordered_actions(legal_actions):
            self._board.apply_action(action)
            score = self._alphabeta(SEARCH_DEPTH - 1, -inf, inf)
            self._board.undo_action()
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        self._board.apply_action(action)

    def _alphabeta(self, depth, alpha, beta):
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

    def _evaluate(self):
        if self._board.game_over:
            if self._board.winner_color == self._color:
                return WIN_SCORE
            if self._board.winner_color == self._color.opponent:
                return -WIN_SCORE
            return 0

        my_tokens = self._board._count_tokens(self._color)
        opp_tokens = self._board._count_tokens(self._color.opponent)
        my_eats = self._count_eat_actions(self._color)
        opp_eats = self._count_eat_actions(self._color.opponent)

        return (
            1000 * (my_tokens - opp_tokens)
            + 70 * (my_eats - opp_eats)
        )

    def _legal_actions(self):
        actions = []
        color = self._board.turn_color
        opponent = color.opponent

        if self._board.phase == GamePhase.PLACEMENT:
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    coord = Coord(r, c)
                    if not self._board._state[coord].is_empty:
                        continue
                    if (self._board._placement_count > 0
                            and self._adjacent_to_color(coord, opponent)):
                        continue
                    actions.append(PlaceAction(coord))
            return actions

        for coord, cell in self._board._state.items():
            if cell.color != color:
                continue
            for direction in CARDINAL_DIRECTIONS:
                dest = self._neighbor(coord, direction)
                if dest is None:
                    continue
                dest_cell = self._board._state[dest]
                if dest_cell.is_empty or dest_cell.color == color:
                    actions.append(MoveAction(coord, direction))
                elif cell.height >= dest_cell.height:
                    actions.append(EatAction(coord, direction))
            if cell.height >= 2:
                for direction in CARDINAL_DIRECTIONS:
                    actions.append(CascadeAction(coord, direction))

        return actions

    def _ordered_actions(self, actions):
        return sorted(actions, key=self._action_priority, reverse=True)

    def _action_priority(self, action):
        match action:
            case EatAction(coord, direction):
                target = self._neighbor(coord, direction)
                if target is None:
                    return 100
                return 400 + 40 * self._board._state[target].height
            case CascadeAction():
                return 180
            case MoveAction():
                return 60
            case PlaceAction(coord):
                return 80 - 8 * self._distance_to_center(coord)
            case _:
                return 0

    def _count_eat_actions(self, color):
        total = 0
        for coord, cell in self._board._state.items():
            if cell.color != color:
                continue
            for direction in CARDINAL_DIRECTIONS:
                dest = self._neighbor(coord, direction)
                if dest is None:
                    continue
                dest_cell = self._board._state[dest]
                if dest_cell.color == color.opponent \
                        and cell.height >= dest_cell.height:
                    total += 1
        return total

    def _adjacent_to_color(self, coord, color):
        for direction in CARDINAL_DIRECTIONS:
            neighbor = self._neighbor(coord, direction)
            if neighbor is not None \
                    and self._board._state[neighbor].color == color:
                return True
        return False

    def _neighbor(self, coord, direction):
        r = coord.r + direction.r
        c = coord.c + direction.c
        if self._board._is_within_bounds(r, c):
            return Coord(r, c)
        return None

    def _distance_to_center(self, coord):
        return min(
            abs(coord.r - r) + abs(coord.c - c)
            for r in (3, 4) for c in (3, 4)
        )