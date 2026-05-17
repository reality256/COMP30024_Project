# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

# [NEW] Agent-side board representation. Mirrors the referee's Board where
# necessary (notably the CASCADE push-chain rules) but is stripped down for
# fast minimax search: cells are packed into a flat int array, the Zobrist
# hash is maintained incrementally, and apply / undo never raise — the agent
# only ever feeds it pre-validated actions, so legality checks belong to the
# action generator instead.

import numpy as np

from referee.game import PlayerColor, Coord, Direction, CARDINAL_DIRECTIONS, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction
from referee.game.board import GamePhase
from referee.game.constants import BOARD_N, MAX_TURNS, PLACEMENT_TURNS, \
    INITIAL_STACK_HEIGHT


# Cell encoding: 0 = empty, +h = RED stack of height h, -h = BLUE stack of
# height h. The referee allows merging up to all 12 tokens of one colour on
# a single cell, so heights stay in the range [-12, 12]. We over-provision
# to 24 to keep the Zobrist key table valid even in degenerate test setups.
_MAX_ABS_HEIGHT = 24
_BOARD_CELLS = BOARD_N * BOARD_N

# [NEW] Zobrist keys. Generated once at import time with a fixed seed so
# every agent process produces the same hashes, which keeps TT entries
# consistent across iterative-deepening passes and across games.
# Layout: _ZOBRIST[cell_index, color_index, height-1].
_ZOBRIST_RNG = np.random.default_rng(seed=0xCA5CADE)
_ZOBRIST = _ZOBRIST_RNG.integers(
    low=1,
    high=2**63 - 1,
    size=(_BOARD_CELLS, 2, _MAX_ABS_HEIGHT),
    dtype=np.int64,
)
_ZOBRIST_TURN = int(_ZOBRIST_RNG.integers(1, 2**63 - 1, dtype=np.int64))


def _cell_index(r: int, c: int) -> int:
    return r * BOARD_N + c


def _zobrist_key(cell_idx: int, value: int) -> int:
    # value: +h for RED, -h for BLUE, 0 for empty (no key).
    if value > 0:
        return int(_ZOBRIST[cell_idx, 0, value - 1])
    if value < 0:
        return int(_ZOBRIST[cell_idx, 1, -value - 1])
    return 0


def _encode(color: PlayerColor, height: int) -> int:
    return height if color == PlayerColor.RED else -height


def _decode_color(value: int) -> PlayerColor | None:
    if value > 0:
        return PlayerColor.RED
    if value < 0:
        return PlayerColor.BLUE
    return None


class MyBoard:
    """
    Agent-side game state.

    The on-disk layout is intentionally minimal: a flat numpy int8 array for
    the 64 cells, plus a small amount of metadata. Hashing is incremental
    (Zobrist), so apply_action / undo_action only touch the cells that change.
    """

    __slots__ = (
        "_cells",
        "_turn_color",
        "_placement_count",
        "_play_phase_turn_count",
        "_hash",
        "_history",
        "_position_history",
    )

    def __init__(self):
        self._cells = np.zeros(_BOARD_CELLS, dtype=np.int8)
        self._turn_color: PlayerColor = PlayerColor.RED
        self._placement_count: int = 0
        self._play_phase_turn_count: int = 0
        # Initial hash includes the turn key for RED-to-move.
        self._hash: int = _ZOBRIST_TURN
        # Each history entry packs the data needed to undo one action.
        # We store mutations as a list of (cell_idx, prev_value) pairs.
        self._history: list[tuple[Action, list[tuple[int, int]], int, int]] = []
        # Position history for threefold repetition (play phase only).
        self._position_history: list[int] = []

    # ----------------------------------------------------------------- queries

    @property
    def turn_color(self) -> PlayerColor:
        return self._turn_color

    @property
    def turn_count(self) -> int:
        return len(self._history)

    @property
    def play_phase_turn_count(self) -> int:
        return self._play_phase_turn_count

    @property
    def phase(self) -> GamePhase:
        return GamePhase.PLACEMENT if self._placement_count < PLACEMENT_TURNS \
            else GamePhase.PLAY

    @property
    def turn_limit_reached(self) -> bool:
        return self._play_phase_turn_count >= MAX_TURNS

    @property
    def board_hash(self) -> int:
        # Incrementally maintained; cheap to read.
        return self._hash

    def cell(self, coord: Coord) -> tuple[PlayerColor | None, int]:
        # Return (color, height) for a coord. Empty -> (None, 0).
        value = int(self._cells[_cell_index(coord.r, coord.c)])
        return _decode_color(value), abs(value)

    def cell_value(self, coord: Coord) -> int:
        # Raw encoded value (+h / -h / 0). Used by hot loops in the agent.
        return int(self._cells[_cell_index(coord.r, coord.c)])

    def is_empty(self, coord: Coord) -> bool:
        return self._cells[_cell_index(coord.r, coord.c)] == 0

    def count_tokens(self, color: PlayerColor) -> int:
        if color == PlayerColor.RED:
            return int(self._cells[self._cells > 0].sum())
        return int(-self._cells[self._cells < 0].sum())

    def count_stacks(self, color: PlayerColor) -> int:
        if color == PlayerColor.RED:
            return int((self._cells > 0).sum())
        return int((self._cells < 0).sum())

    def iter_cells(self):
        # Generator yielding (coord, encoded_value) for non-empty cells.
        for idx in range(_BOARD_CELLS):
            value = int(self._cells[idx])
            if value != 0:
                yield Coord(idx // BOARD_N, idx % BOARD_N), value

    @staticmethod
    def is_within_bounds(r: int, c: int) -> bool:
        return 0 <= r < BOARD_N and 0 <= c < BOARD_N

    # --------------------------------------------------------- repetition info

    def threefold_repetition(self) -> bool:
        if len(self._position_history) < 3:
            return False
        return self._position_history.count(self._hash) >= 3

    def repetition_count(self) -> int:
        # How many times the current position has appeared so far.
        return self._position_history.count(self._hash)

    # ------------------------------------------------------------ legal moves

    def has_legal_actions(self) -> bool:
        # Used to detect stalemate. Short-circuits as soon as any move is found.
        color = self._turn_color
        for idx in range(_BOARD_CELLS):
            value = int(self._cells[idx])
            if value == 0:
                continue
            if _decode_color(value) != color:
                continue
            height = abs(value)
            r, c = divmod(idx, BOARD_N)
            for direction in CARDINAL_DIRECTIONS:
                nr, nc = r + direction.r, c + direction.c
                if not self.is_within_bounds(nr, nc):
                    continue
                dest_value = int(self._cells[_cell_index(nr, nc)])
                if dest_value == 0 or _decode_color(dest_value) == color:
                    return True
                if abs(dest_value) <= height:
                    return True
            if height >= 2:
                return True
        return False

    # ------------------------------------------------------------- game over

    @property
    def game_over(self) -> bool:
        if self.phase == GamePhase.PLACEMENT:
            return False
        if self.count_tokens(PlayerColor.RED) == 0:
            return True
        if self.count_tokens(PlayerColor.BLUE) == 0:
            return True
        if self.turn_limit_reached:
            return True
        if self.threefold_repetition():
            return True
        if not self.has_legal_actions():
            return True
        return False

    @property
    def winner_color(self) -> PlayerColor | None:
        if not self.game_over:
            return None
        red = self.count_tokens(PlayerColor.RED)
        blue = self.count_tokens(PlayerColor.BLUE)
        if red == 0:
            return PlayerColor.BLUE
        if blue == 0:
            return PlayerColor.RED
        if self.threefold_repetition():
            return None
        if not self.has_legal_actions():
            return None
        if red > blue:
            return PlayerColor.RED
        if blue > red:
            return PlayerColor.BLUE
        return None

    # ------------------------------------------------------- apply / undo

    def apply_action(self, action: Action) -> None:
        # Action validity is the caller's responsibility — the agent only ever
        # passes generated-legal or referee-validated actions.
        if isinstance(action, PlaceAction):
            mutations = self._mutations_place(action)
        elif isinstance(action, MoveAction):
            mutations = self._mutations_move(action)
        elif isinstance(action, EatAction):
            mutations = self._mutations_eat(action)
        elif isinstance(action, CascadeAction):
            mutations = self._mutations_cascade(action)
        else:
            raise ValueError(f"Unknown action type: {action!r}")

        # Apply cell mutations, recording previous values for undo.
        prev_snapshot: list[tuple[int, int]] = []
        for cell_idx, next_value in mutations:
            prev_value = int(self._cells[cell_idx])
            if prev_value == next_value:
                continue
            prev_snapshot.append((cell_idx, prev_value))
            # Incrementally update the Zobrist hash: XOR out the old key,
            # XOR in the new one.
            if prev_value != 0:
                self._hash ^= _zobrist_key(cell_idx, prev_value)
            if next_value != 0:
                self._hash ^= _zobrist_key(cell_idx, next_value)
            self._cells[cell_idx] = next_value

        # Side effects: placement counter, phase-aware turn counter, turn flip.
        was_placement = self.phase == GamePhase.PLACEMENT
        prev_placement_count = self._placement_count
        if isinstance(action, PlaceAction):
            self._placement_count += 1
        # Turn flip: XOR the turn key.
        self._hash ^= _ZOBRIST_TURN
        self._turn_color = self._turn_color.opponent

        # Play-phase turn counter is incremented only after placement is over.
        # Mirror the referee, which counts a turn once it has been applied.
        if not was_placement:
            self._play_phase_turn_count += 1

        # Track position history for threefold repetition (play phase only,
        # recorded AFTER the turn flip so the hash matches "this position
        # with the given player to move").
        if self.phase == GamePhase.PLAY:
            self._position_history.append(self._hash)

        # Remember whether we just appended a position-history entry, so undo
        # can pop the same amount it pushed.
        pushed_position = self.phase == GamePhase.PLAY
        self._history.append(
            (action, prev_snapshot, prev_placement_count, pushed_position),
        )

    def undo_action(self) -> None:
        if not self._history:
            raise IndexError("No actions to undo.")
        action, prev_snapshot, prev_placement_count, pushed_position \
            = self._history.pop()

        # Restore position history first, while phase still reflects the
        # post-apply state (matters when the action transitioned PLACEMENT
        # -> PLAY: that move pushed a hash but the counter we'd check after
        # undo would say PLACEMENT).
        if pushed_position and self._position_history:
            self._position_history.pop()

        # Determine whether the action belonged to the placement phase. The
        # placement counter has already been incremented during apply, so we
        # compare against prev_placement_count here.
        was_placement = prev_placement_count < PLACEMENT_TURNS
        if not was_placement:
            self._play_phase_turn_count -= 1

        # Undo turn flip.
        self._turn_color = self._turn_color.opponent
        self._hash ^= _ZOBRIST_TURN

        # Undo placement counter.
        self._placement_count = prev_placement_count

        # Undo cell mutations in reverse.
        for cell_idx, prev_value in reversed(prev_snapshot):
            current_value = int(self._cells[cell_idx])
            if current_value != 0:
                self._hash ^= _zobrist_key(cell_idx, current_value)
            if prev_value != 0:
                self._hash ^= _zobrist_key(cell_idx, prev_value)
            self._cells[cell_idx] = prev_value

    # --------------------------------------------------- mutation builders

    def _mutations_place(self, action: PlaceAction) -> list[tuple[int, int]]:
        idx = _cell_index(action.coord.r, action.coord.c)
        return [(idx, _encode(self._turn_color, INITIAL_STACK_HEIGHT))]

    def _mutations_move(self, action: MoveAction) -> list[tuple[int, int]]:
        src = action.coord
        dest = Coord(src.r + action.direction.r, src.c + action.direction.c)
        src_idx = _cell_index(src.r, src.c)
        dest_idx = _cell_index(dest.r, dest.c)

        src_value = int(self._cells[src_idx])
        dest_value = int(self._cells[dest_idx])

        if dest_value == 0:
            # Relocate.
            return [(src_idx, 0), (dest_idx, src_value)]
        # Merge with friendly: heights add, sign preserved.
        merged = src_value + dest_value  # both same sign, so adds correctly
        return [(src_idx, 0), (dest_idx, merged)]

    def _mutations_eat(self, action: EatAction) -> list[tuple[int, int]]:
        src = action.coord
        dest = Coord(src.r + action.direction.r, src.c + action.direction.c)
        src_idx = _cell_index(src.r, src.c)
        dest_idx = _cell_index(dest.r, dest.c)
        src_value = int(self._cells[src_idx])
        # Attacker moves into the target cell at its original height.
        # Captured tokens are eliminated.
        return [(src_idx, 0), (dest_idx, src_value)]

    def _mutations_cascade(self, action: CascadeAction) -> list[tuple[int, int]]:
        # Simulate cascade in a temporary dict and emit the diff. The logic
        # exactly matches the referee's _resolve_cascade_action and _push_stack
        # so search results stay consistent with what the referee will do.
        src = action.coord
        direction = action.direction
        src_idx = _cell_index(src.r, src.c)
        src_value = int(self._cells[src_idx])
        height = abs(src_value)
        color_sign = 1 if src_value > 0 else -1

        # Working copy keyed by cell index, only of cells we touch.
        working: dict[int, int] = {src_idx: 0}

        def get_value(r: int, c: int) -> int:
            idx = _cell_index(r, c)
            return working[idx] if idx in working else int(self._cells[idx])

        def push_stack(coord_r: int, coord_c: int):
            # Recursively push the stack at (coord_r, coord_c) one cell along.
            idx = _cell_index(coord_r, coord_c)
            value = working[idx] if idx in working else int(self._cells[idx])
            if value == 0:
                return
            dr, dc = coord_r + direction.r, coord_c + direction.c
            if not self.is_within_bounds(dr, dc):
                # Pushed off the board — eliminated.
                working[idx] = 0
                return
            dest_idx = _cell_index(dr, dc)
            dest_value = working[dest_idx] if dest_idx in working \
                else int(self._cells[dest_idx])
            if dest_value != 0:
                push_stack(dr, dc)
            working[dest_idx] = value
            working[idx] = 0

        # Lay down one token per step, pushing existing stacks ahead of it.
        for step in range(1, height + 1):
            tr = src.r + direction.r * step
            tc = src.c + direction.c * step
            if not self.is_within_bounds(tr, tc):
                # Cascading token falls off — discarded.
                continue
            target_idx = _cell_index(tr, tc)
            target_value = working[target_idx] if target_idx in working \
                else int(self._cells[target_idx])
            if target_value != 0:
                push_stack(tr, tc)
            # After pushing (if any), drop a height-1 token of our colour.
            working[target_idx] = color_sign

        # Return only cells whose final value differs from the current board.
        return [
            (idx, value)
            for idx, value in working.items()
            if value != int(self._cells[idx])
        ]