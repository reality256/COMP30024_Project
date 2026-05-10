# Cascade Agent

This version is still based on **minimax with alpha-beta pruning**, but it has been refined from the first version into a more practical tournament agent.

Compared with the previous version:

1. The search now uses **iterative deepening**, so the agent can return the best result from the deepest fully completed search when time is limited.
2. The time control is more careful. The agent allocates different budgets for placement and play phases, keeps a safety margin, and reduces depth when little time remains.
3. Legal actions are ordered before search. Captures, useful cascades, central placements, and merge moves are prioritised to improve alpha-beta pruning.
4. The branching factor is controlled by limiting the number of actions searched at the root and inside the tree.
5. The evaluation function now considers more game-specific features, including material, stack count, available captures, cascade potential, placement quality, turn-limit pressure, and repetition risk.
6. A fallback action and final referee legality check are used so the agent can still return a safe legal move if search is interrupted.

For a fuller design description, see `approach_report.md`.

## Known Issue

When testing against `agent_simple`, the game may be interrupted because `agent_simple` can sometimes produce an illegal action during the middle of the game. This is an issue with the simple testing opponent rather than the main agent. If this happens, the referee may stop the match before the result reflects the actual strength of the current agent.
