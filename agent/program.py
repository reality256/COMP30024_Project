# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from math import inf
from time import process_time

from referee.game import PlayerColor, Coord, Direction, CARDINAL_DIRECTIONS, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction
from referee.game.board import Board, GamePhase
from referee.game.constants import BOARD_N, MAX_TURNS, PLACEMENT_TURNS
from referee.game.exceptions import IllegalActionException


WIN_SCORE = 1_000_000
DEFAULT_SEARCH_DEPTH = 3
TACTICAL_SEARCH_DEPTH = 4
PLACEMENT_SEARCH_DEPTH = 2

# Keep well below the referee's 180s-per-player game budget. The search is
# iterative, so it can always fall back to the best completed shallower result.
MAX_PLAYER_SECONDS = 170.0
PLAY_TURN_SECONDS = 0.70
PLACEMENT_TURN_SECONDS = 0.25
MIN_TURN_SECONDS = 0.03
TIME_SAFETY_SECONDS = 1.50

# Alpha-beta is strongest when ordering is good, but Cascade has a large
# branching factor. These caps preserve tactical options while keeping the
# per-turn tree small enough for tournament time limits.
ROOT_ACTION_LIMIT = 36
SEARCH_ACTION_LIMIT = 24


class SearchTimeout(Exception):
    """Raised internally when the current iterative-deepening slice expires."""


class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Cascade game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._color = color
        self._board = Board()
        self._time_spent = 0.0

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object.
        """

        legal_actions = self._legal_actions()
        if not legal_actions:
            raise RuntimeError("No legal actions found.")

        start = process_time()
        deadline = start + self._turn_time_budget(referee, len(legal_actions))
        root_actions = self._ordered_root_actions(
            legal_actions,
            ROOT_ACTION_LIMIT,
            deadline,
        )
        best_action = self._fallback_action(root_actions, deadline)

        # Iterative deepening gives a legal move quickly, then improves it while
        # time remains. If a deeper iteration times out, the last full result is
        # still safe to play.
        try:
            for depth in range(
                1,
                self._search_depth_limit(len(root_actions), referee) + 1,
            ):
                best_score = -inf
                depth_best = best_action
                alpha = -inf

                for action in root_actions:
                    self._check_time(deadline)
                    self._board.apply_action(action)
                    try:
                        score = self._alphabeta(depth - 1, alpha, inf, deadline)
                    finally:
                        self._board.undo_action()

                    if score > best_score:
                        best_score = score
                        depth_best = action
                    alpha = max(alpha, best_score)

                best_action = depth_best
        except SearchTimeout:
            pass
        finally:
            self._time_spent += process_time() - start

        return self._safe_action(best_action, legal_actions)

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state.
        """
        self._board.apply_action(action)

    def _alphabeta(
        self,
        depth: int,
        alpha: float,
        beta: float,
        deadline: float,
    ) -> float:
        self._check_time(deadline)

        # End the search loop.
        if depth == 0 or self._board.game_over:
            return self._evaluate()
        
        # Must have legal actions to continue.
        legal_actions = self._legal_actions()
        if not legal_actions:
            return self._evaluate()
        
        ordered_actions = self._ordered_actions(legal_actions, SEARCH_ACTION_LIMIT)

        # The agent's turn, maximizing score.
        if self._board.turn_color == self._color:
            value = -inf
            for action in ordered_actions:
                self._check_time(deadline)
                self._board.apply_action(action)
                try:
                    value = max(
                        value,
                        self._alphabeta(depth - 1, alpha, beta, deadline),
                    )
                finally:
                    self._board.undo_action()

                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        
        # The opponent's turn, minimizing score.
        value = inf
        for action in ordered_actions:
            self._check_time(deadline)
            self._board.apply_action(action)
            try:
                value = min(
                    value,
                    self._alphabeta(depth - 1, alpha, beta, deadline),
                )
            finally:
                self._board.undo_action()

            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def _evaluate(self) -> int:
        if self._board.game_over:
            if self._board.winner_color == self._color:
                return WIN_SCORE - self._board.play_phase_turn_count
            if self._board.winner_color == self._color.opponent:
                return -WIN_SCORE + self._board.play_phase_turn_count
            return 0

        my_tokens = self._board._count_tokens(self._color)
        opp_tokens = self._board._count_tokens(self._color.opponent)
        token_diff = my_tokens - opp_tokens

        my_stacks = self._board._count_stacks(self._color)
        opp_stacks = self._board._count_stacks(self._color.opponent)

        my_eats = self._count_eat_actions(self._color)
        opp_eats = self._count_eat_actions(self._color.opponent)

        placement_score = self._placement_score(self._color) \
            - self._placement_score(self._color.opponent)

        cascade_score = self._cascade_potential(self._color) \
            - self._cascade_potential(self._color.opponent)

        turn_limit_score = self._turn_limit_score(token_diff)
        repetition_score = self._repetition_score(token_diff)
        
        # Material dominates because token count decides elimination and the
        # turn-limit winner. Mobility terms are deliberately smaller so they
        # guide choices without overriding obvious captures.
        return (
            1000 * token_diff
            + 85 * (my_stacks - opp_stacks)
            + 70 * (my_eats - opp_eats)
            + 4 * cascade_score
            + placement_score
            + turn_limit_score
            + repetition_score
        )

    def _legal_actions(self) -> list[Action]:
        actions: list[Action] = []
        color = self._board.turn_color

        # Placement phase: mirror Board._resolve_place_action without using
        # exceptions for every candidate.
        if self._board.phase == GamePhase.PLACEMENT:
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    coord = Coord(r, c)
                    if not self._board._state[coord].is_empty:
                        continue
                    if (
                        self._board._placement_count > 0
                        and self._adjacent_to_color(coord, color.opponent)
                    ):
                        continue
                    actions.append(PlaceAction(coord))
            return actions

        # Play phase: the checks match the referee's MOVE/EAT/CASCADE rules.
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

    def _referee_accepts(self, action: Action) -> bool:
        # Final safety check using the referee implementation.
        try:
            self._board.apply_action(action)
        except IllegalActionException:
            return False
        self._board.undo_action()
        return True

    def _safe_action(self, preferred: Action, legal_actions: list[Action]) -> Action:
        if self._referee_accepts(preferred):
            return preferred

        for action in legal_actions:
            if self._referee_accepts(action):
                return action

        raise RuntimeError("No referee-legal action found.")

    def _ordered_actions(
        self,
        actions: list[Action],
        limit: int | None = None,
    ) -> list[Action]:
        ordered = sorted(actions, key=self._action_priority, reverse=True)
        if limit is None:
            return ordered
        return ordered[:limit]

    def _ordered_root_actions(
        self,
        actions: list[Action],
        limit: int | None = None,
        deadline: float | None = None,
    ) -> list[Action]:
        scored_actions: list[tuple[int, Action]] = []
        for action in actions:
            score = self._action_priority(action)
            if deadline is None or process_time() < deadline:
                score += self._terminal_action_bonus(action)
            scored_actions.append((score, action))

        ordered = [
            action
            for _, action in sorted(
                scored_actions,
                key=lambda item: item[0],
                reverse=True,
            )
        ]
        if limit is None:
            return ordered
        return ordered[:limit]

    def _terminal_action_bonus(self, action: Action) -> int:
        self._board.apply_action(action)
        try:
            if not self._board.game_over:
                return 0

            winner = self._board.winner_color
            if winner == self._color:
                return WIN_SCORE
            if winner == self._color.opponent:
                return -WIN_SCORE

            token_diff = (
                self._board._count_tokens(self._color)
                - self._board._count_tokens(self._color.opponent)
            )
            if token_diff > 0:
                return -WIN_SCORE // 2
            if token_diff < 0:
                return WIN_SCORE // 2
            return 0
        finally:
            self._board.undo_action()

    def _fallback_action(self, actions: list[Action], deadline: float) -> Action:
        best_action = actions[0]
        best_score = -inf

        for action in actions:
            if process_time() >= deadline:
                break

            self._board.apply_action(action)
            try:
                score = self._evaluate()
            finally:
                self._board.undo_action()

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _action_priority(self, action: Action) -> int:
        color = self._board.turn_color

        match action:
            case EatAction(coord, direction):
                target = self._neighbor(coord, direction)
                if target is None:
                    return 100
                return 400 + 40 * self._board._state[target].height
            case CascadeAction(coord, direction):
                return 180 + self._cascade_direction_potential(
                    coord,
                    self._board._state[coord].height,
                    direction,
                    color,
                )
            case PlaceAction(coord):
                # Place stacks closer to the center while avoiding brittle edges.
                return 80 - 8 * self._distance_to_center(coord)
            case MoveAction(coord, direction):
                target = self._neighbor(coord, direction)
                if target is None:
                    return 0
                dest = self._board._state[target]
                if dest.color == color:
                    return 140 + 10 * dest.height
                return 60 - 4 * self._distance_to_center(target)
            case _:
                return 0

    def _count_eat_actions(self, color: PlayerColor) -> int:
        # Count the number of possible eat actions for either player without
        # changing Board.turn_color.
        total = 0
        for coord, cell in self._board._state.items():
            if cell.color != color:
                continue

            for direction in CARDINAL_DIRECTIONS:
                dest = self._neighbor(coord, direction)
                if dest is None:
                    continue

                dest_cell = self._board._state[dest]
                if dest_cell.color == color.opponent and cell.height >= dest_cell.height:
                    total += 1

        return total

    def _cascade_potential(self, color: PlayerColor) -> int:
        # A cheap static estimate of useful cascade options. It avoids full
        # cascade simulation at every leaf, which is the main time saver.
        total = 0
        for coord, cell in self._board._state.items():
            if cell.color != color or cell.height < 2:
                continue

            # Having a cascade available is useful, but tall stacks are also
            # valuable material, so the directional score does most of the work.
            total += 8 * (cell.height - 1)
            best_direction = max(
                self._cascade_direction_potential(
                    coord,
                    cell.height,
                    direction,
                    color,
                )
                for direction in CARDINAL_DIRECTIONS
            )
            total += best_direction

        return total

    def _cascade_direction_potential(
        self,
        coord: Coord,
        height: int,
        direction: Direction,
        color: PlayerColor,
    ) -> int:
        score = 0
        distance_to_edge = self._distance_to_edge(coord, direction)
        lost_tokens = max(0, height - distance_to_edge)
        score -= 25 * lost_tokens

        for step in range(1, min(height, distance_to_edge) + 1):
            target = Coord(
                coord.r + direction.r * step,
                coord.c + direction.c * step,
            )
            target_cell = self._board._state[target]
            if target_cell.is_empty:
                score += 4
                continue

            push_score = self._push_chain_score(target, direction, color)
            if target_cell.color == color.opponent:
                score += 35 + 12 * target_cell.height + push_score
            else:
                score -= 10 * target_cell.height
                score += push_score

        return score

    def _push_chain_score(
        self,
        start: Coord,
        direction: Direction,
        color: PlayerColor,
    ) -> int:
        # If a cascade pushes a consecutive line of stacks off board, the last
        # stack in that line is eliminated. Reward enemy losses, punish our own.
        coord = start
        last_cell = None

        while coord is not None:
            cell = self._board._state[coord]
            if cell.is_empty:
                return 8

            last_cell = cell
            coord = self._neighbor(coord, direction)

        if last_cell is None:
            return 0
        if last_cell.color == color.opponent:
            return 90 + 30 * last_cell.height
        return -80 - 25 * last_cell.height

    def _placement_score(self, color: PlayerColor) -> int:
        # Closer to the center is better; edge stacks are easier to push off.
        score = 0
        for coord, cell in self._board._state.items():
            if cell.color != color:
                continue

            score -= 5 * self._distance_to_center(coord)
            if coord.r in (0, BOARD_N - 1) or coord.c in (0, BOARD_N - 1):
                score -= 10

        return score

    def _turn_time_budget(self, referee: dict, action_count: int) -> float:
        remaining = self._remaining_time(referee)
        if remaining <= TIME_SAFETY_SECONDS + MIN_TURN_SECONDS:
            return MIN_TURN_SECONDS

        base_budget = (
            PLACEMENT_TURN_SECONDS
            if self._board.phase == GamePhase.PLACEMENT
            else PLAY_TURN_SECONDS
        )
        usable_time = max(MIN_TURN_SECONDS, remaining - TIME_SAFETY_SECONDS)
        fair_share = usable_time / self._estimated_future_action_calls()

        if fair_share < base_budget:
            budget = fair_share * 1.20
        else:
            budget = base_budget + 0.45 * (fair_share - base_budget)

        cap = 0.45 if self._board.phase == GamePhase.PLACEMENT else 1.15
        if self._board.phase == GamePhase.PLAY:
            if action_count <= 10:
                cap = 1.60
            if self._remaining_play_turns() <= 24:
                cap = max(cap, 1.80)

        return max(
            MIN_TURN_SECONDS,
            min(cap, budget, usable_time),
        )

    def _search_depth_limit(self, action_count: int, referee: dict) -> int:
        remaining = self._remaining_time(referee)
        if remaining <= 2.0:
            return 1
        if remaining <= 6.0:
            return 2

        if self._board.phase == GamePhase.PLACEMENT:
            return PLACEMENT_SEARCH_DEPTH
        if self._remaining_play_turns() <= 20 and action_count <= 18:
            return TACTICAL_SEARCH_DEPTH
        if action_count <= 10:
            return TACTICAL_SEARCH_DEPTH + (1 if remaining > 20.0 else 0)
        return DEFAULT_SEARCH_DEPTH

    def _check_time(self, deadline: float):
        if process_time() >= deadline:
            raise SearchTimeout

    def _remaining_time(self, referee: dict) -> float:
        time_remaining = referee.get("time_remaining")
        if time_remaining is not None:
            return max(0.0, float(time_remaining))
        return max(0.0, MAX_PLAYER_SECONDS - self._time_spent)

    def _estimated_future_action_calls(self) -> int:
        if self._board.phase == GamePhase.PLACEMENT:
            placement_turns = PLACEMENT_TURNS - self._board.turn_count
            play_turns = MAX_TURNS
            return max(1, (placement_turns + 1) // 2 + (play_turns + 1) // 2)

        return max(1, (self._remaining_play_turns() + 1) // 2)

    def _remaining_play_turns(self) -> int:
        return max(0, MAX_TURNS - self._board.play_phase_turn_count)

    def _turn_limit_score(self, token_diff: int) -> int:
        if self._board.phase != GamePhase.PLAY:
            return 0

        remaining_turns = self._remaining_play_turns()
        if remaining_turns > 40:
            return 0
        return (40 - remaining_turns) * 20 * token_diff

    def _repetition_score(self, token_diff: int) -> int:
        if self._board.phase != GamePhase.PLAY or token_diff == 0:
            return 0

        current_hash = self._board._board_hash()
        repetitions = self._board._position_history.count(current_hash)
        if repetitions < 2:
            return 0

        pressure = 650 + 250 * abs(token_diff)
        return -pressure if token_diff > 0 else pressure

    def _distance_to_center(self, coord: Coord) -> int:
        # On the 8x8 board, the four centre cells all get the same best score.
        return min(
            abs(coord.r - r) + abs(coord.c - c)
            for r in (3, 4)
            for c in (3, 4)
        )

    def _distance_to_edge(self, coord: Coord, direction: Direction) -> int:
        distance = 0
        current = coord
        while True:
            current = self._neighbor(current, direction)
            if current is None:
                return distance
            distance += 1

    def _adjacent_to_color(self, coord: Coord, color: PlayerColor) -> bool:
        for direction in CARDINAL_DIRECTIONS:
            neighbor = self._neighbor(coord, direction)
            if neighbor is not None and self._board._state[neighbor].color == color:
                return True
        return False

    def _neighbor(self, coord: Coord, direction: Direction) -> Coord | None:
        r = coord.r + direction.r
        c = coord.c + direction.c
        if self._board._is_within_bounds(r, c):
            return Coord(r, c)
        return None
