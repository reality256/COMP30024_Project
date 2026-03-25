# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part A: Single Player Cascade

import heapq
from .core import CellState, Coord, Direction, Action, MoveAction, EatAction, CascadeAction
from .utils import render_board

BOARD_N = 8
BoardState = frozenset


def _copy_board(board: dict[Coord, CellState]) -> dict[Coord, CellState]:
    """辅助函数：返回当前棋盘的浅拷贝。"""
    return dict(board)


def board_to_state(board: dict[Coord, CellState]) -> BoardState:
    return frozenset((coord, cell.color, cell.height) for coord, cell in board.items())


def is_goal(board: dict[Coord, CellState]) -> bool:
    return not any(
        cell.color is not None and cell.color.name == "BLUE"
        for cell in board.values()
    )


def push_stack(
    board: dict[Coord, CellState],
    coord: Coord,
    direction: Direction,
) -> dict[Coord, CellState]:
    if coord not in board:
        return board

    stack_cell = board[coord]
    target_r = coord.r + direction.r
    target_c = coord.c + direction.c

    if not (0 <= target_r < BOARD_N and 0 <= target_c < BOARD_N):
        # 推出边界：这个堆栈从棋盘上移除
        new_board = _copy_board(board)
        del new_board[coord]
        return new_board

    target = Coord(target_r, target_c)
    new_board = _copy_board(board)
    del new_board[coord]

    if target in new_board:
        new_board = push_stack(new_board, target, direction)

    new_board[target] = stack_cell
    return new_board


def apply_cascade(
    board: dict[Coord, CellState],
    coord: Coord,
    direction: Direction,
    height: int,
    source_color,
) -> dict[Coord, CellState]:
    # Cascade：移除源堆栈，并按方向分布高度个单元格的token。
    # 1..height每个位置都放入一个新的单元格，如果有阻碍则先推开原堆栈。
    new_board = _copy_board(board)
    if coord in new_board:
        del new_board[coord]

    # Process each distance from 1 to height
    for step in range(1, height + 1):
        target_r = coord.r + direction.r * step
        target_c = coord.c + direction.c * step

        # If target is off the board, the token is eliminated
        if not (0 <= target_r < BOARD_N and 0 <= target_c < BOARD_N):
            continue

        target = Coord(target_r, target_c)
        
        # Push any existing stack at this position
        if target in new_board:
            new_board = push_stack(new_board, target, direction)

        # Place the new token at this position
        new_board[target] = CellState(source_color, 1)

    return new_board


def get_possible_actions(board: dict[Coord, CellState]) -> list[tuple[Action, dict[Coord, CellState]]]:
    # 计算当前棋盘状态下的所有合法行动
    action_candidates = []

    for coord, cell in list(board.items()):
        if cell.color is None or cell.color.name != "RED":
            continue

        for direction in Direction:
            try:
                dest = coord + direction
            except ValueError:
                continue

            if not (0 <= dest.r < BOARD_N and 0 <= dest.c < BOARD_N):
                continue

            dest_cell = board.get(dest)

            # Move  操作：红色可直接移动或与红色合并
            if dest_cell is None:
                new_board = _copy_board(board)
                del new_board[coord]
                new_board[dest] = cell
                action_candidates.append((MoveAction(coord, direction), new_board))
            elif dest_cell.color is not None and dest_cell.color.name == "RED":
                new_board = _copy_board(board)
                del new_board[coord]
                merged_height = cell.height + dest_cell.height
                new_board[dest] = CellState(cell.color, merged_height)
                action_candidates.append((MoveAction(coord, direction), new_board))

            # Eat  操作：红色吃掉高度不超过自己的蓝色
            if (
                dest_cell is not None
                and dest_cell.color is not None
                and dest_cell.color.name == "BLUE"
                and cell.height >= dest_cell.height
            ):
                new_board = _copy_board(board)
                del new_board[coord]
                new_board[dest] = CellState(cell.color, cell.height)
                action_candidates.append((EatAction(coord, direction), new_board))

            # Cascade
            if cell.height >= 2:
                new_board = apply_cascade(board, coord, direction, cell.height, cell.color)
                action_candidates.append((CascadeAction(coord, direction), new_board))

    return action_candidates


def heuristic_function(board: dict[Coord, CellState]) -> int:
    """
    启发函数：估计从当前状态到目标状态所需的最少步骤数。
    
    使用多层级启发函数，确保admissible（永不高估）。
    层级：
    1. 蓝色堆栈数量（每个必须消除）
    2. 红色高度不足的补偿
    3. 位置优化：最近的红色到最近的蓝色
    """
    blue_stacks = []
    red_stacks = []
    
    for coord, cell in board.items():
        if cell.color is not None:
            if cell.color.name == "BLUE":
                blue_stacks.append((coord, cell.height))
            elif cell.color.name == "RED":
                red_stacks.append((coord, cell.height))
    
    # 目标状态
    if not blue_stacks:
        return 0
    
    # 无法解决
    if not red_stacks:
        return 1000
    
    # 基础启发值：蓝色堆栈数量
    num_blue = len(blue_stacks)
    max_blue_height = max(h for _, h in blue_stacks) if blue_stacks else 1
    max_red_height = max(h for _, h in red_stacks) if red_stacks else 0
    
    # 因素1: 必须消除的蓝色数量
    h_value = num_blue
    
    # 因素2: 如果红色不足以吃掉最强的蓝色，需要增长
    if max_red_height < max_blue_height:
        height_deficit = max_blue_height - max_red_height
        # 需要多少次合并来达到所需高度
        # 最坏情况：每次合并只增加1（实际上可以更优，但这保证admissible）
        h_value += height_deficit
    
    return h_value


def a_star_search(initial_board: dict[Coord, CellState]) -> list[Action] | None:
    """
    使用A*算法搜索解决方案。
    
    A* = 最优优先搜索，使用 f(n) = g(n) + h(n)
    其中：
    - g(n) = 从初始状态到节点n的实际成本
    - h(n) = 启发函数估计从n到目标的成本
    """
    if is_goal(initial_board):
        return []
    
    # 优先级队列，存储 (f_value, counter, board_state, actions)
    # counter用于打破相同f_value的平局，确保FIFO顺序
    open_set = []
    counter = 0
    
    initial_state = board_to_state(initial_board)
    h_initial = heuristic_function(initial_board)
    heapq.heappush(open_set, (h_initial, counter, initial_state, initial_board, []))
    counter += 1
    
    # 已访问的状态集合（g值最优的状态）
    visited: set[BoardState] = set()
    # 每个状态的最佳g值
    g_values: dict[BoardState, int] = {initial_state: 0}
    
    while open_set:
        f_value, _, current_state, current_board, actions = heapq.heappop(open_set)
        
        # 如果已访问过，跳过
        if current_state in visited:
            continue
        
        visited.add(current_state)
        g_current = len(actions)
        
        # 检查是否达到目标
        if is_goal(current_board):
            return actions
        
        # 探索所有后继节点
        for action, next_board in get_possible_actions(current_board):
            next_state = board_to_state(next_board)
            
            # 跳过已访问的状态
            if next_state in visited:
                continue
            
            g_next = g_current + 1
            
            # 如果找到了更优的路径或第一次访问此状态
            if next_state not in g_values or g_next < g_values[next_state]:
                g_values[next_state] = g_next
                h_next = heuristic_function(next_board)
                f_next = g_next + h_next
                
                new_actions = actions + [action]
                heapq.heappush(
                    open_set,
                    (f_next, counter, next_state, next_board, new_actions)
                )
                counter += 1
    
    # 没有找到解决方案
    return None


def search(
    board: dict[Coord, CellState]
) -> list[Action] | None:
    print(render_board(board, ansi=True))
    return a_star_search(board)