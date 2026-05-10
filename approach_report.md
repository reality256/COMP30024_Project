# Agent Approach Report

## Overview

My agent is a conventional game-playing program based on depth-limited adversarial search. It keeps an internal copy of the board, updates that board after every action reported by the referee, and uses the current board state to choose its next move.

The main design goal is to make a strong enough tactical player while staying safely inside the tournament time limit. For this reason, the agent does not try to search the full game tree. Instead, it combines alpha-beta search, iterative deepening, action ordering, a fixed-size move shortlist, and a hand-designed evaluation function.

No machine learning is used. I chose this because the game has clear tactical rules and a relatively small board, so a well-tuned search player is a more reliable first approach than training a model with limited data.

## Action Selection

On each turn, the agent follows this process:

1. Generate all legal actions for the current board state.
2. Order the actions using a fast tactical priority function.
3. Keep only the strongest-looking root actions, so the search remains manageable.
4. Select a fallback action using a one-ply evaluation.
5. Run iterative deepening alpha-beta search until the time budget for the turn is reached.
6. Return the best action from the deepest fully completed search.
7. Before returning, check that the chosen action is accepted by the referee board implementation.

The fallback action is important because the agent must always return a legal move, even if the search runs out of time. Iterative deepening also helps with this: the agent first completes a shallow search, then improves the decision if more time is available.

## Search Algorithm

The main search algorithm is minimax with alpha-beta pruning. The agent maximises the evaluation score on its own turns and minimises the score on the opponent's turns. Alpha-beta pruning was chosen because Cascade is a zero-sum, turn-based game with perfect information, so minimax is a natural fit. Alpha-beta keeps the same final result as minimax at a given depth, but avoids searching branches that cannot affect the final decision.

The search depth is not fixed for the whole game. The agent adjusts it based on the game phase, the number of candidate actions, the remaining time, and the number of turns left. Placement turns use a smaller depth because early positions have many legal placements and less immediate tactical contact. In the play phase, the default depth is higher. The agent may search deeper in tactical situations, such as when there are few legal actions or when the game is close to the turn limit.

The agent also uses iterative deepening. It searches depth 1 first, then depth 2, and so on up to the current depth limit. If time expires during a deeper search, the agent keeps the best result from the last completed depth. This makes the player more stable under time pressure.

## Changes to Basic Alpha-Beta

The implementation includes several practical modifications to basic alpha-beta search.

First, legal actions are ordered before searching. Captures are considered highly important, especially captures of taller enemy stacks. Cascades are also prioritised when they can push enemy stacks, create useful space, or remove enemy material. Placement and movement actions are given lower but still meaningful priority based on board position and merging potential. Good ordering improves alpha-beta pruning because strong moves are searched earlier.

Second, the agent limits the number of actions searched. At the root it keeps up to 36 actions, and inside the search tree it keeps up to 24 actions. This is a selective search decision. Cascade can create a large branching factor, and searching every legal action at depth 3 or 4 can become too slow. The shortlist is designed to preserve the most tactically relevant moves while preventing the search tree from growing too large.

Third, the agent uses a time-aware search. Each turn receives a dynamic time budget. The budget depends on the referee's reported remaining time, the phase of the game, the expected number of future turns, and the current branching factor. The agent keeps a safety margin below the total game budget. If there is very little time left, it reduces the depth and returns a fast fallback action.

Finally, terminal moves are detected during root ordering. A move that immediately wins is strongly preferred, and a move that immediately loses is strongly penalised. This helps the agent avoid missing simple end states because of action ordering or depth limits.

## Evaluation Function

The evaluation function estimates how good a non-terminal board state is for the agent. It is a weighted sum of several strategic features.

The most important feature is material, measured as the difference in total token count. This receives the largest weight because token count decides elimination wins and also decides the winner when the turn limit is reached. Losing material is usually hard to recover from, so material should dominate small positional preferences.

The second feature is stack count. Having more stacks can give more available moves and more board coverage. However, stacks are less important than total tokens, because a single tall stack can still be very powerful. This feature helps the agent value active board presence without ignoring material.

The third feature is immediate capture mobility. The agent counts how many Eat actions are available for each player. This rewards positions where the agent can threaten enemy stacks and penalises positions where the opponent has many direct captures. The motivation is simple: captures are direct material swings, so the agent should prefer positions with more capture pressure.

The fourth feature is placement quality. During and after placement, stacks closer to the centre receive a higher score, while edge stacks are penalised. Central stacks usually have more movement options and are harder to push off the board. Edge stacks are more vulnerable to cascades because a pushed stack can be eliminated by leaving the board.

The fifth feature is cascade potential. A full simulation of all possible cascades at every leaf would be expensive, so the evaluation uses a cheaper static estimate. It rewards cascades that can hit enemy stacks, push chains of stacks, or remove enemy material from the board. It penalises cascades that lose too many of the agent's own tokens off the edge or push friendly stacks into bad positions. This feature is important because cascade moves can change the board much more than a normal move.

The sixth feature is turn-limit pressure. As the game approaches the maximum number of play-phase turns, the evaluation gives more value to the current token difference. If the agent is ahead on material, it has more reason to preserve that lead; if it is behind, it has more reason to seek active changes instead of drifting toward a token-count loss.

The final feature handles repeated positions. Threefold repetition is a draw, so the agent adjusts the score when the current position has already appeared. If the agent is ahead, repetition is bad because it can throw away a winning position. If the agent is behind, repetition is more acceptable because a draw may be better than losing.

## Technical Details

The agent uses the referee board implementation as its internal state representation. For search, it applies an action, evaluates or searches the resulting state, and then undoes the action. This avoids repeatedly copying the full board and keeps the search implementation simple.

Legal action generation is implemented directly instead of trying actions and catching exceptions. This matters because action generation is called many times during search. Avoiding exception-heavy control flow makes the search faster and easier to reason about.

The program uses small helper functions for board geometry, such as neighbour lookup, distance to centre, and distance to edge. These helpers make the tactical scoring for placement and cascade actions clearer.

The final safety check uses the referee's own action resolver before returning a move. This is a defensive step: even if the custom legal action generator has a bug, the agent should avoid sending an illegal action when another legal action is available.

## Summary

Overall, the agent is a selective alpha-beta player. It searches concrete future action sequences where possible, but relies on action ordering and a strategic evaluation function to keep the search small enough for the time limit. The evaluation focuses on material, tactical capture chances, useful cascade pressure, central placement, endgame token advantage, and repetition risk. The result is a clear search-based design without machine learning, with most of the strength coming from careful move ordering and game-specific evaluation features.
