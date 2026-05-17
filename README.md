# COMP30024 Cascade Game-Playing Agent

This repository contains a game-playing agent for **Cascade**, developed for
COMP30024 Artificial Intelligence at the University of Melbourne.

The project includes the official-style referee, several baseline agents, and a
stronger final agent that uses adversarial search, time management, and a custom
fast board representation.

## Overview

Cascade is a two-player board game played by Red and Blue on an 8x8 grid. Each
agent receives game updates from the referee and must return legal actions under
strict time and memory limits.

The final agent lives in `agent/` and is designed for tournament play. It
combines:

- Iterative-deepening minimax search with alpha-beta pruning
- Phase-aware evaluation for placement, opening, midgame, and endgame play
- Move ordering for captures, cascades, merges, and positional control
- Transposition-table caching with Zobrist hashing
- Killer-move ordering for repeated alpha-beta cutoffs
- A NumPy-backed `MyBoard` implementation with fast apply/undo operations
- Time budgeting that adapts to the game phase, action count, and remaining CPU
  allowance

## Repository Structure

```text
.
+-- agent/             # Final search-based Cascade agent
|   +-- program.py     # Agent entry point and search/evaluation logic
|   +-- myboard.py     # Fast agent-side board with Zobrist hashing
+-- agent_random/      # Random baseline agent
+-- agent_greedy/      # One-ply greedy baseline agent
+-- agent_weak/        # Simple alpha-beta baseline agent
+-- referee/           # Referee, game rules, logging, and CLI runner
+-- game_spec.pdf      # Cascade game specification
+-- part_b.pdf         # Project Part B specification
+-- report.pdf         # Project report
+-- team.py            # Course team metadata
```

## Requirements

- Python 3.10 or newer
- NumPy

Install the only external runtime dependency with:

```bash
pip install numpy
```

The included `.devcontainer/` configuration can also be used with VS Code
Dev Containers.

## Running Games

Run a game by passing two agent packages to the referee. The first package plays
Red and the second package plays Blue.

```bash
python -m referee agent agent_random
```

Useful examples:

```bash
# Final agent against the greedy baseline
python -m referee agent agent_greedy

# Final agent against the weak alpha-beta baseline
python -m referee agent agent_weak

# Final agent mirror match
python -m referee agent agent

# Quieter output
python -m referee agent agent_random -v 0

# Write a game log
python -m referee agent agent_random -l game.log
```

For all referee options:

```bash
python -m referee --help
```

## Agent Design

The final agent uses iterative deepening so it always has a legal fallback move
available before searching deeper. Search is performed on `MyBoard`, a compact
board model that mirrors the referee rules while avoiding the overhead of
exception-driven legality checks in hot loops.

Evaluation is tuned around the main phases of the game:

- Placement and opening: central control, stable positions, and repetition
  avoidance
- Midgame: material balance, capture threats, stack structure, and cascade
  potential
- Endgame: token advantage and turn-limit conversion

The implementation also keeps a synchronized referee `Board` for final legality
checks before returning an action.

## Baseline Agents

The extra agents are included for local testing and comparison:

- `agent_random`: chooses uniformly from legal actions
- `agent_greedy`: evaluates all immediate legal moves and picks the best one
- `agent_weak`: uses a shallow alpha-beta search with a simpler evaluation

## Academic Integrity

This repository is shared as a portfolio and learning reference. Please do not
copy this work for current or future coursework submissions. If you are taking
COMP30024 or a similar subject, use this only to understand broad ideas after
checking your course's academic integrity policy.

## License

This project is released under the MIT License. See `LICENSE` for details.
