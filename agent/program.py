# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent

from math import inf

from referee.game import PlayerColor, Coord, Direction, CARDINAL_DIRECTIONS, \
    Action, PlaceAction, MoveAction, EatAction, CascadeAction
from referee.game.board import Board, GamePhase
from referee.game.constants import BOARD_N
from referee.game.exceptions import IllegalActionException


WIN_SCORE = 1_000_000
SEARCH_DEPTH = 2


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

        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED (first player)")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")

    def action(self, **referee: dict) -> Action:
        """
        This method is called by the referee each time it is the agent's turn
        to take an action. It must always return an action object.
        """

        legal_actions = self._legal_actions()
        if not legal_actions:
            raise RuntimeError("No legal actions found.")
        #Initial state
        best_action = legal_actions[0]
        best_score = -inf

        #Use alpha-beta pruning to search
        for action in self._ordered_actions(legal_actions):
            #Try to apply and then undo, so that we don't need to copy the board
            self._board.apply_action(action)
            score = self._alphabeta(SEARCH_DEPTH - 1, -inf, inf)
            self._board.undo_action()

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        This method is called by the referee after a player has taken their
        turn. You should use it to update the agent's internal game state.
        """
        self._board.apply_action(action)

    def _alphabeta(self, depth: int, alpha: float, beta: float) -> float:

        #End the search loop
        if depth == 0 or self._board.game_over:
            return self._evaluate()
        
        #Must have legal actions to continue
        legal_actions = self._legal_actions()
        if not legal_actions:
            return self._evaluate()
        
        #The agent's turn, maximizing score
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
        
        #The opponent's turn, minimizing score
        value = inf
        for action in self._ordered_actions(legal_actions):
            self._board.apply_action(action)
            value = min(value, self._alphabeta(depth - 1, alpha, beta))
            self._board.undo_action()

            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def _evaluate(self) -> int:
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
        
        '''
        parameter weights to be tuned
        '''
        return (
            1000 * (my_tokens - opp_tokens)
            + 100 * (my_stacks - opp_stacks)
            + 50 * (my_eats - opp_eats)
            + placement_score
        )

    def _legal_actions(self) -> list[Action]:
        actions: list[Action] = []

        #Placement phase
        if self._board.phase == GamePhase.PLACEMENT:
            for r in range(BOARD_N):
                for c in range(BOARD_N):
                    action = PlaceAction(Coord(r, c))
                    if self._is_legal(action):
                        actions.append(action)
            return actions

        #Only consider the active player(if red, it skips blue)
        for coord, cell in self._board._state.items():
            if cell.color != self._board.turn_color:
                continue

            for direction in CARDINAL_DIRECTIONS:
                for action in (
                    MoveAction(coord, direction),
                    EatAction(coord, direction),
                    CascadeAction(coord, direction),
                ):
                    if self._is_legal(action):
                        actions.append(action)

        return actions

    def _is_legal(self, action: Action) -> bool:
        #Use funtion in referee, don't need to do it ourselves.
        try:
            self._board.apply_action(action)
            self._board.undo_action()
            return True
        except IllegalActionException:
            return False

    def _ordered_actions(self, actions: list[Action]) -> list[Action]:
        return sorted(actions, key=self._action_priority, reverse=True)

    def _action_priority(self, action: Action) -> int:
        '''
        parameter weights to be tuned
        '''
        match action:
            case EatAction(coord, direction):
                target = self._try_add(coord, direction)
                if target is None:
                    return 100
                return 100 + self._board._state[target].height
            case CascadeAction():
                return 50
            case PlaceAction(coord):
                #Place stacks closer to the center
                return 20 - self._distance_to_center(coord)
            case MoveAction():
                return 10
            case _:
                return 0

    def _count_tokens(self, color: PlayerColor) -> int:
        #the sum of token heights, this is the most important factor
        #regarding the end condition of the game
        total = 0
        for cell in self._board._state.values():
            if cell.color == color:
                total += cell.height
        return total

    def _count_stacks(self, color: PlayerColor) -> int:
        #the number of stacks
        #I think tokens/stacks ratio can reflect the intensity of the game situation, but that is to be tuned
        total = 0
        for cell in self._board._state.values():
            if cell.color == color:
                total += 1
        return total

    def _count_eat_actions(self, color: PlayerColor) -> int:
        #count the number of possible eat actions
        total = 0
        for coord, cell in self._board._state.items():
            if cell.color != color:
                continue

            for direction in CARDINAL_DIRECTIONS:
                dest = self._try_add(coord, direction)
                if dest is None:
                    continue

                dest_cell = self._board._state[dest]
                if dest_cell.color == color.opponent and cell.height >= dest_cell.height:
                    total += 1

        return total

    def _placement_score(self, color: PlayerColor) -> int:
        #Closer to the center is better
        '''
        parameter weights to be tuned
        '''
        score = 0
        for coord, cell in self._board._state.items():
            if cell.color != color:
                continue

            score -= 5 * self._distance_to_center(coord)
            if coord.r in (0, BOARD_N - 1) or coord.c in (0, BOARD_N - 1):
                score -= 10

        return score

    def _distance_to_center(self, coord: Coord) -> int:
        # On the 8x8 board, the four centre cells all get the same best score.
        return min(
            abs(coord.r - r) + abs(coord.c - c)
            for r in (3, 4)
            for c in (3, 4)
        )

    def _try_add(self, coord: Coord, direction: Direction) -> Coord | None:
        #To simplify the out of board detection
        try:
            return coord + direction
        except ValueError:
            return None
