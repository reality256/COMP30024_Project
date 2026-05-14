# COMP30024_Project
A course project of COMP30024 in the University of Melbourne.
# Cascade Agent — Changes Since the Last Shared Version

This document summarises everything that has changed in `agent/program.py`
(and the new `agent/myboard.py`). The agent now sits at **vs random 100% / vs greedy 100% /
vs weak 95%** with an even RED/BLUE split and no early draws in self-play.

All new or modified code is annotated inline with `[NEW]` or `[CHANGED]`
tags. Existing comments are preserved verbatim.

---

## File overview

| File | Status | Purpose |
|---|---|---|
| `agent/program.py` | modified | Agent entry point; search + evaluation |
| `agent/myboard.py` | **new** | Agent-side board with incremental Zobrist hashing |
| `agent/__init__.py` | unchanged | Original template file |

`myboard.py` must be in the same package as `program.py` (the import is
`from .myboard import MyBoard`).

NumPy is the only new dependency — it is permitted by the project spec.

---

## 1. New board representation (`myboard.py`)

A custom board class replaces the referee's `Board` during search.

- Cells stored as a flat `numpy.int8` array (length 64), encoded
  `+h` for RED stacks of height `h` and `-h` for BLUE.
- **Zobrist hashing**: a fixed-seed key table is generated at import time;
  `apply_action` / `undo_action` update the hash incrementally by XOR-ing
  the keys of cells that change. The hash is therefore O(1) per cell
  change, where the referee's `_board_hash()` was O(N log N) per call
  on the whole board.
- `apply_action` / `undo_action` mirror every referee rule including the
  full CASCADE push-chain semantics. We keep a per-action mutation log
  so undo is exact.
- 64-bit hash collisions did not appear in the validation runs (20 random
  full games × ~200 actions each, hash-vs-state cross-check).

A referee `Board` is still kept in sync inside the `Agent` and used only
as a defensive legality check (`_referee_accepts`) on the final action
chosen each turn.

## 2. Search algorithm changes (`program.py`)

### Transposition table

Persistent across turns; flushed if it exceeds 400 000 entries (well
under the 250 MB memory limit).

- Entries: `(depth, flag, score, best_action)`.
- Three flag types (`TT_EXACT`, `TT_LOWER`, `TT_UPPER`) implement the
  standard fail-soft α-β bound semantics.
- **Depth-preferred replacement**: a shallower result never overwrites a
  deeper one.
- The cached `best_action` is promoted to the front of the move list on
  the next visit to that position (see `_promote_first`).

### Iterative-deepening continuity

At each new ID depth, the previous iteration's best root action is
tried first. This is implemented in `action()` via the same
`_promote_first` helper.

### Killer-move heuristic

Records quiet moves (non-EAT, non-CASCADE) that cause a β-cutoff.

- Table keyed on `(depth, side-to-move)`. **The side key is essential**:
  keying on depth alone caused a regression in testing because MAX-node
  killers were promoted at MIN nodes (and vice versa), tilting the
  opponent's ordering in the wrong direction.
- Cleared at the start of every `action()` call.
- Up to 2 killers per `(depth, side)` cell; promoted immediately after
  the TT best move.

## 3. Evaluation function changes (`program.py`)

### Phase-aware weights

Three weight sets (`_OPENING_WEIGHTS`, `_MIDGAME_WEIGHTS`,
`_ENDGAME_WEIGHTS`) selected by play-phase turns remaining.

| Phase | Range (play turns) | token | stack | eat | cascade |
|---|---|---|---|---|---|
| opening | 0–99 | 1000 | **60** | 100 | **3** |
| midgame | 100–199 | 1000 | 40 | 120 | 2 |
| endgame | 200–299 | **1600** | 25 | **130** | 2 |

**Motivation**:
- **Opening's higher `stack` weight** encourages MOVE-merge actions,
  which change the board topology and break threefold-repetition
  equilibria in self-play. Before this change, self-play games
  routinely ended in a 38- or 39-turn draw because both sides
  deterministically played into the same repeating position.
- **Endgame's higher `token` weight** sharpens the material signal as
  the turn limit approaches. The turn-limit tiebreaker is decided on
  token count, so a small material lead should be played more
  decisively in the final 100 turns.
- **Endgame's higher `eat` weight** rewards captures over passive
  stack-building when time is running out.
- Midgame is the previously tuned baseline; we left it untouched.

The placement phase falls through to the opening weights — placement
choices are dominated by `_placement_score`, which is unchanged.

### Threat differential (`_threat_score`)

A new evaluation term: for each side, total the height of every stack
that an enemy stack could capture next turn (the EAT precondition:
attacker height ≥ target height across a cardinal direction). The
difference (opponent threatened minus our own threatened) is added to
the evaluation with `THREAT_WEIGHT = 15`.

- This term changes nothing during placement (returns 0 early — EAT
  is not legal there).
- It rewards both attacks on the opponent and defence of our own
  pieces in a single signed signal.
- Weight `15` was chosen experimentally; we tested `10`, `15`, `20`
  and `15` won on vs-weak and self-play.

### Constant draw penalty (`DRAW_PENALTY = 100`)

Returned for terminal draws inside `_evaluate`. Small enough that a
clearly losing line is still worse than a forced draw, but large
enough that the agent will fight on when winning chances exist.

An earlier version made this asymmetric (different penalty depending on
the agent's current token lead). That version was rolled back because
it caused the side without material advantage to under-defend its own
tokens in the lead-up to an expected draw.

## 4. Things I tried and removed

- **Asymmetric draw penalty** (`_draw_value` that returned different
  values for different token-diff buckets): regressed vs weak from 95%
  to 35%. Removed.
- **Mobility score** (sum of legal MOVE/merge destinations per side,
  weight 8): regressed vs weak from 85% to 80% and self-play decisive
  games went down. The mobility term conflicted with the stack weight
  — friendly merges *reduce* future mobility, so the search learned
  to avoid them, which reintroduced the 39-turn self-play draw we
  fixed with the higher opening `stack` weight. Removed.
- **Repetition-avoidance bonus inside `_terminal_action_bonus`**:
  the version that was sketched in early was unreachable (early
  `return 0` short-circuited it). Removed; the in-evaluation
  `_repetition_score` covers the same idea more cleanly.

## 5. How to run

Same as before:

```
python -m referee agent agent_weak
```

## 6. Current scoreboard

| Opponent | Games | NEW result |
|---|---|---|
| agent_random | 5 | 100% |
| agent_greedy | 5 | 100% |
| agent_weak | 20 | 95% (19W-1L-0D, no draws) |
| self-play | 4 | 3 decisive, 0 early draws, 0 repetition draws |


## 7. Reading guide for the changes

If you are reviewing the diff:

1. Start with `myboard.py` — every line is new
   (encoding helpers → public queries → apply/undo → mutation builders).
2. In `program.py`, look at module-level constants first: `DRAW_PENALTY`,
   `THREAT_WEIGHT`, `_OPENING_WEIGHTS` / `_MIDGAME_WEIGHTS` /
   `_ENDGAME_WEIGHTS`.
3. Then `_alphabeta` for the TT-and-killer integration.
4. Finally `_evaluate`, `_threat_score`, `_phase_weights` for the
   evaluation changes.

Every `[NEW]` block is something I added; every `[CHANGED]` block is
a line in the previous version I had to modify (mostly to route
through `MyBoard` instead of poking the referee's internals).
