# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from math import inf
from time import process_time

from referee.game import PlayerColor, Coord, Direction, CARDINAL_DIRECTIONS, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction
from referee.game.board import Board, GamePhase
from referee.game.constants import BOARD_N, MAX_TURNS, PLACEMENT_TURNS
from referee.game.exceptions import IllegalActionException

# [NEW] Agent-side board with incremental Zobrist hashing. The referee Board
# is still instantiated for _referee_accepts (a defensive final legality
# check), but all hot-path operations during search go through MyBoard.
from .myboard import MyBoard


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

# [NEW] Transposition table: cache evaluated positions to avoid re-searching
# the same board state reached via different move orders. Entries also store
# the best action discovered at that node, which seeds move ordering in
# later iterative-deepening passes. The cap keeps memory well under the
# referee's 250 MB per-player limit.
TT_MAX_ENTRIES = 400_000

# [NEW] TT entry flags for alpha-beta bound types.
TT_EXACT = 0   # score is the true minimax value
TT_LOWER = 1   # score is a lower bound (caused a beta-cutoff)
TT_UPPER = 2   # score is an upper bound (no move improved alpha)

# [NEW] Phase-aware evaluation weights. Tuned to address two self-play
# weaknesses: short games ending in threefold repetition (opening fixes it
# by encouraging position-changing merges and cascades) and 300-turn
# draws (endgame fixes it by sharpening the token-diff term so a small
# material lead is converted into a turn-limit win).
_OPENING_WEIGHTS = {
        "token":   1000,  
        "stack":    60,   # merge actions change the position, break repetition
        "eat":     100,
        "cascade":   3,
}
_MIDGAME_WEIGHTS = {
        "token":  1000,   # baseline (the current best-tuned values)
        "stack":    40,
        "eat":     120,
        "cascade":   2,
}
_ENDGAME_WEIGHTS = {
        "token":  1600,   # token diff dominates: turn-limit decides on tokens
        "stack":    25,
        "eat":     130,
        "cascade":   2,
}

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
        # [CHANGED] Search runs on MyBoard (numpy-backed cells + incremental
        # Zobrist hashing). A referee Board is also kept in sync to power the
        # defensive _referee_accepts check at the end of action().
        self._board = MyBoard()
        self._ref_board = Board()
        self._time_spent = 0.0
        # [NEW] Transposition table: maps board_hash -> (depth, flag, score, best_action).
        # Persists across turns: the board hash encodes the full game state,
        # so a hash match always means the same position. Older entries seed
        # iterative-deepening passes and improve move ordering, effectively
        # giving one extra ply for free.
        self._tt: dict[int, tuple[int, int, float, Action | None]] = {}
        # [NEW] Killer moves table: (depth, side_to_move) -> [killer1, killer2].
        # Records non-tactical actions that caused a beta-cutoff at a given
        # ply *for a particular side*. The side-to-move must be part of the
        # key: a move that refutes a MAX-node position is not interchangeable
        # with one that refutes a MIN-node position — promoting the former
        # at the latter would tilt the opponent's ordering in our favour and
        # delay finding their real best reply. These are tried right after
        # the TT best move on subsequent matching nodes. Reset at the start
        # of every action() call because killers from a previous turn are
        # unlikely to still be relevant.
        self._killers: dict[tuple[int, PlayerColor], list[Action]] = {}

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object.
        """
        legal_actions = self._legal_actions()
        if not legal_actions:
            raise RuntimeError("No legal actions found.")

        # [NEW] Clear killer moves at the start of every turn. Killer moves
        # are heuristics keyed by ply depth; they do not generalise across
        # turns because the surrounding board state has changed.
        self._killers.clear()

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

                # [NEW] At each new iteration, try the previous iteration's
                # best move first. This is the cheapest, most effective form
                # of move ordering across deepening passes.
                ordered_iter = self._promote_first(root_actions, best_action)

                for action in ordered_iter:
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
        # [CHANGED] Apply to both our hash-tracked MyBoard and the kept-in-sync
        # referee Board. Apply order matters only for _ref_board (used purely
        # for legality checking later).
        self._board.apply_action(action)
        self._ref_board.apply_action(action)

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

        # [NEW] Transposition table lookup. The hash is read for free from
        # MyBoard (incrementally maintained), so the check itself is O(1).
        board_hash = self._board.board_hash
        tt_best: Action | None = None
        tt_entry = self._tt.get(board_hash)
        if tt_entry is not None:
            cached_depth, cached_flag, cached_score, cached_best = tt_entry
            tt_best = cached_best
            if cached_depth >= depth:
                if cached_flag == TT_EXACT:
                    return cached_score
                if cached_flag == TT_LOWER and cached_score >= beta:
                    return cached_score
                if cached_flag == TT_UPPER and cached_score <= alpha:
                    return cached_score

        ordered_actions = self._ordered_actions(legal_actions, SEARCH_ACTION_LIMIT)
        # [NEW] Promote the TT best move to the front. If the cached best is
        # outside the cap (e.g. SEARCH_ACTION_LIMIT trimmed it), we still try
        # it first because TT memory of "good at this position" outweighs the
        # static priority heuristic.
        if tt_best is not None:
            ordered_actions = self._promote_first(ordered_actions, tt_best)
        # [NEW] Then promote killer moves for this (depth, side-to-move),
        # just behind the TT best move. Killers come from previous
        # beta-cutoffs at this ply *for the same side* and tend to refute
        # many sibling positions in the same neighbourhood. Keying on the
        # side-to-move keeps MAX-node and MIN-node killers from
        # contaminating each other's ordering.
        side_to_move = self._board.turn_color
        ordered_actions = self._promote_killers(ordered_actions, depth, side_to_move)

        original_alpha = alpha
        best_action_here: Action | None = None

        # The agent's turn, maximizing score.
        if self._board.turn_color == self._color:
            value = -inf
            for action in ordered_actions:
                self._check_time(deadline)
                self._board.apply_action(action)
                try:
                    child = self._alphabeta(depth - 1, alpha, beta, deadline)
                finally:
                    self._board.undo_action()

                if child > value:
                    value = child
                    best_action_here = action
                alpha = max(alpha, value)
                if alpha >= beta:
                    # [NEW] Record this beta-cutoff producer as a killer for
                    # the current (depth, side-to-move).
                    self._record_killer(depth, side_to_move, action)
                    break
        else:
            # The opponent's turn, minimizing score.
            value = inf
            for action in ordered_actions:
                self._check_time(deadline)
                self._board.apply_action(action)
                try:
                    child = self._alphabeta(depth - 1, alpha, beta, deadline)
                finally:
                    self._board.undo_action()

                if child < value:
                    value = child
                    best_action_here = action
                beta = min(beta, value)
                if alpha >= beta:
                    # [NEW] Same recording for the minimizing side. The
                    # (depth, side) key keeps these separate from the
                    # maximizing side's killers.
                    self._record_killer(depth, side_to_move, action)
                    break

        # [NEW] Transposition table store with depth-preferred replacement:
        # a shallower entry never overwrites a deeper one, which keeps the
        # most reliable scores around. Also flush the table if it grows too
        # large rather than risk crossing the 250 MB memory cap.
        if len(self._tt) >= TT_MAX_ENTRIES:
            self._tt.clear()

        if value <= original_alpha:
            flag = TT_UPPER   # failed low: true value is at most this
        elif value >= beta:
            flag = TT_LOWER   # failed high: true value is at least this
        else:
            flag = TT_EXACT   # within the window: exact minimax value

        existing = self._tt.get(board_hash)
        if existing is None or existing[0] <= depth:
            self._tt[board_hash] = (depth, flag, value, best_action_here)

        return value
    
    def _threat_score(self, color: PlayerColor) -> int:
        if self._board.phase == GamePhase.PLACEMENT:
            return 0  # placement phase에선 EAT 불가능
    
        score = 0
        for coord, value in self._board.iter_cells():
            if (value > 0) != (color == PlayerColor.RED):
                continue
            my_height = abs(value)
            for direction in CARDINAL_DIRECTIONS:
                neighbor = self._neighbor(coord, direction)
                if neighbor is None:
                    continue
                n_value = self._board.cell_value(neighbor)
                if n_value == 0:
                    continue
                if (n_value > 0) == (color == PlayerColor.RED):
                    continue  # 아군
                if abs(n_value) >= my_height:
                    score += my_height
                    break  # 같은 셀에 중복 페널티 방지
        return score

    def _evaluate(self) -> int:
        # [CHANGE] Add draw penalty
        # Draws score slightly negative so the agent prefers fighting to
        # forced repetition, but the magnitude is bounded so we never trade
        # a clear loss for a draw avoidance. The penalty is intentionally
        # symmetric: an asymmetric token-diff-aware version was tried and
        # caused the agent to under-defend its own tokens in the lead-up
        # to a likely draw, hurting win rate.
        if self._board.game_over:
            if self._board.winner_color == self._color: 
                return WIN_SCORE - self._board.play_phase_turn_count
            if self._board.winner_color == self._color.opponent: 
                return -WIN_SCORE + self._board.play_phase_turn_count
            return -100

        my_tokens = self._board.count_tokens(self._color)
        opp_tokens = self._board.count_tokens(self._color.opponent)
        token_diff = my_tokens - opp_tokens

        my_stacks = self._board.count_stacks(self._color)
        opp_stacks = self._board.count_stacks(self._color.opponent)

        my_eats = self._count_eat_actions(self._color)
        opp_eats = self._count_eat_actions(self._color.opponent)

        placement_score = self._placement_score(self._color) \
            - self._placement_score(self._color.opponent)

        cascade_score = self._cascade_potential(self._color) \
            - self._cascade_potential(self._color.opponent)

        turn_limit_score = self._turn_limit_score(token_diff)
        repetition_score = self._repetition_score(token_diff)
        # placement phase가 아닐 때만 의미 있음
        threat_diff = (
            self._threat_score(self._color.opponent)   # 적이 위협받음 = 우리에게 좋음
            - self._threat_score(self._color)           # 우리가 위협받음 = 나쁨
        )

        # [NEW] Phase-aware weights. Opening favours position-changing moves
        # (merges, cascades) to break early-repetition stalemates; endgame
        # leans on raw token diff because the turn-limit tiebreaker is decided
        # by token count. Midgame keeps the previously tuned baseline.
        weights = self._phase_weights()

        return (
            weights["token"]    * token_diff
            + weights["stack"]  * (my_stacks - opp_stacks)
            + weights["eat"]    * (my_eats - opp_eats)
            + weights["cascade"] * cascade_score
            + 15 * threat_diff  # [NEW] 위협 평가
            + placement_score
            + turn_limit_score
            + repetition_score
        )

    def _phase_weights(self) -> dict[str, int]:
        # [NEW] Returns the evaluation weights for the current game phase.
        # Phase is determined by play-phase turns remaining (out of the 300-turn
        # play limit). Placement phase falls through to the opening weights —
        # _placement_score handles the bulk of that phase's signal.
        if self._board.phase == GamePhase.PLACEMENT:
            return _OPENING_WEIGHTS

        remaining = self._remaining_play_turns()
        if remaining > 200:    # play_turns < 100
            return _OPENING_WEIGHTS
        if remaining > 100:    # 100 <= play_turns < 200
            return _MIDGAME_WEIGHTS
        return _ENDGAME_WEIGHTS    # play_turns >= 200

    def _legal_actions(self) -> list[Action]:
        actions: list[Action] = []
        color = self._board.turn_color

        # Placement phase: mirror Board._resolve_place_action without using
        # exceptions for every candidate.
        if self._board.phase == GamePhase.PLACEMENT:
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    coord = Coord(r, c)
                    # [CHANGED] Use MyBoard's typed accessors instead of
                    # poking referee Board internals.
                    if not self._board.is_empty(coord):
                        continue
                    if (
                        self._board._placement_count > 0
                        and self._adjacent_to_color(coord, color.opponent)
                    ):
                        continue
                    actions.append(PlaceAction(coord))
            return actions

        # Play phase: the checks match the referee's MOVE/EAT/CASCADE rules.
        # [CHANGED] iter_cells yields only non-empty cells (about 8 of 64
        # in a typical mid-game), which is much cheaper than the previous
        # full-grid scan.
        for coord, value in self._board.iter_cells():
            if (value > 0) != (color == PlayerColor.RED):
                continue
            height = abs(value)

            for direction in CARDINAL_DIRECTIONS:
                dest = self._neighbor(coord, direction)
                if dest is None:
                    continue

                dest_value = self._board.cell_value(dest)
                if dest_value == 0 or ((dest_value > 0) == (color == PlayerColor.RED)):
                    actions.append(MoveAction(coord, direction))
                elif height >= abs(dest_value):
                    actions.append(EatAction(coord, direction))

            if height >= 2:
                for direction in CARDINAL_DIRECTIONS:
                    actions.append(CascadeAction(coord, direction))

        return actions

    def _referee_accepts(self, action: Action) -> bool:
        # Final safety check using the referee implementation.
        # [CHANGED] Uses the kept-in-sync _ref_board so MyBoard's search-time
        # state is never disturbed by this probe.
        try:
            self._ref_board.apply_action(action)
        except IllegalActionException:
            return False
        self._ref_board.undo_action()
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

    def _promote_first(
        self,
        actions: list[Action],
        preferred: Action | None,
    ) -> list[Action]:
        # [NEW] Return a list where `preferred` (if present) is moved to the
        # front, otherwise the list is returned unchanged. Used to seed root
        # iterations with the previous best move and inner nodes with the TT
        # best move.
        if preferred is None:
            return actions
        try:
            idx = actions.index(preferred)
        except ValueError:
            return actions
        if idx == 0:
            return actions
        return [preferred] + actions[:idx] + actions[idx + 1:]

    def _promote_killers(
        self,
        actions: list[Action],
        depth: int,
        side: PlayerColor,
    ) -> list[Action]:
        # [NEW] Move killer moves for this (depth, side) cell to the front
        # of the action list, just behind the TT best move (which the caller
        # has already promoted with _promote_first). Killers are slotted in
        # their stored order: killer1 first, killer2 second. Tactical actions
        # are never added to the killer table (see _record_killer), so there
        # is no risk of double-counting with the EAT/CASCADE priority bonus.
        killers = self._killers.get((depth, side))
        if not killers:
            return actions

        promoted: list[Action] = []
        remaining = list(actions)
        for killer in killers:
            try:
                remaining.remove(killer)
            except ValueError:
                continue
            promoted.append(killer)

        if not promoted:
            return actions
        return promoted + remaining

    def _record_killer(self, depth: int, side: PlayerColor, action: Action) -> None:
        # [NEW] Remember an action that caused a beta-cutoff at this depth
        # for this side-to-move. Tactical actions (EAT, CASCADE) are skipped
        # because they already rank highly via _action_priority — killer
        # slots are reserved for quiet moves that turned out to refute the
        # position. Keying on (depth, side) keeps MAX-node and MIN-node
        # killers from contaminating each other's ordering.
        if isinstance(action, (EatAction, CascadeAction)):
            return
        key = (depth, side)
        killers = self._killers.setdefault(key, [])
        if action in killers:
            return
        killers.insert(0, action)
        if len(killers) > 2:
            killers.pop()

    def _terminal_action_bonus(self, action: Action) -> int:
        # The 반복-회피 block that was sketched here was unreachable in the
        # previous version (the early `return 0` above it short-circuited
        # every non-terminal action). It is removed for now until the
        # asymmetric repetition logic is reworked from scratch — adding it
        # back as-is would not have changed search behaviour.
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
                self._board.count_tokens(self._color)
                - self._board.count_tokens(self._color.opponent)
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
                # [CHANGED] Use MyBoard's typed accessor.
                _, target_height = self._board.cell(target)
                return 400 + 40 * target_height
            case CascadeAction(coord, direction):
                _, height = self._board.cell(coord)
                return 180 + self._cascade_direction_potential(
                    coord,
                    height,
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
                dest_color, dest_height = self._board.cell(target)
                if dest_color == color:
                    return 140 + 10 * dest_height
                return 60 - 4 * self._distance_to_center(target)
            case _:
                return 0

    def _count_eat_actions(self, color: PlayerColor) -> int:
        # Count the number of possible eat actions for either player without
        # changing Board.turn_color.
        total = 0
        # [CHANGED] iter_cells skips empties; fewer cells to scan per call.
        for coord, value in self._board.iter_cells():
            if (value > 0) != (color == PlayerColor.RED):
                continue
            height = abs(value)

            for direction in CARDINAL_DIRECTIONS:
                dest = self._neighbor(coord, direction)
                if dest is None:
                    continue

                dest_value = self._board.cell_value(dest)
                if dest_value == 0:
                    continue
                # Opponent stack iff signs differ.
                if (dest_value > 0) != (value > 0) and height >= abs(dest_value):
                    total += 1

        return total

    def _cascade_potential(self, color: PlayerColor) -> int:
        # A cheap static estimate of useful cascade options. It avoids full
        # cascade simulation at every leaf, which is the main time saver.
        total = 0
        for coord, value in self._board.iter_cells():
            if (value > 0) != (color == PlayerColor.RED):
                continue
            height = abs(value)
            if height < 2:
                continue

            # Having a cascade available is useful, but tall stacks are also
            # valuable material, so the directional score does most of the work.
            total += 8 * (height - 1)
            best_direction = max(
                self._cascade_direction_potential(
                    coord,
                    height,
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
            # [CHANGED] Use MyBoard accessors.
            target_value = self._board.cell_value(target)
            if target_value == 0:
                score += 4
                continue

            target_height = abs(target_value)
            target_is_opponent = (target_value > 0) != (color == PlayerColor.RED)
            push_score = self._push_chain_score(target, direction, color)
            if target_is_opponent:
                score += 35 + 12 * target_height + push_score
            else:
                score -= 10 * target_height
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
        last_value = 0

        while coord is not None:
            value = self._board.cell_value(coord)
            if value == 0:
                return 8

            last_value = value
            coord = self._neighbor(coord, direction)

        if last_value == 0:
            return 0
        last_height = abs(last_value)
        last_is_opponent = (last_value > 0) != (color == PlayerColor.RED)
        if last_is_opponent:
            return 90 + 30 * last_height
        return -80 - 25 * last_height

    def _placement_score(self, color: PlayerColor) -> int:
        # Closer to the center is better; edge stacks are easier to push off.
        score = 0
        for coord, value in self._board.iter_cells():
            if (value > 0) != (color == PlayerColor.RED):
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

        # [CHANGED] Read repetition count straight from MyBoard's hash-based
        # position history; no need to recompute the hash here.
        repetitions = self._board.repetition_count()
        if repetitions < 1:
            return 0
        # more pressure for repetitions
        pressure_base = 300 if repetitions == 1 else 650 + 250 * abs(token_diff)
        return -pressure_base if token_diff > 0 else pressure_base

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
            if neighbor is None:
                continue
            # [CHANGED] Use the typed cell accessor instead of poking the
            # referee's _state dict.
            nb_color, _ = self._board.cell(neighbor)
            if nb_color == color:
                return True
        return False

    def _neighbor(self, coord: Coord, direction: Direction) -> Coord | None:
        r = coord.r + direction.r
        c = coord.c + direction.c
        if MyBoard.is_within_bounds(r, c):
            return Coord(r, c)
        return None