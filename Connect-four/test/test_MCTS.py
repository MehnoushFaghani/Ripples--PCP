from agents.common import (
    PlayerAction,
    PLAYER1,
    initialize_game_state,
    connected_four
)

from agents.agent_MCTS.MCTS import Node
from agents.agent_MCTS.MCTS import (generate_move_MCTS,
                                    MCTS,
                                    PLAYER)

# Nodes Functions


def test_getMoves():
    board = initialize_game_state()
    current_node = Node(state=board, player=PLAYER1)
    assert current_node.getMoves().shape[0] == 7


def test_selection():
    board = initialize_game_state()
    child_board01 = board.copy()
    child_board01[0, 0] = PLAYER1
    child_board02 = board.copy()
    child_board02[0, 3] = PLAYER1
    child_node01 = Node(state=child_board01, player=PLAYER1)
    child_node02 = Node(state=child_board02, player=PLAYER1)
    current_node = Node(state=board, player=PLAYER1)
    current_node.visits = 100
    child_node01.visits = 50
    child_node02.visits = 40
    child_node01.wins = 35
    child_node02.wins = 25
    children_array = [child_node01, child_node02]
    current_node.childNodes = children_array
    assert child_node01.__eq__(current_node.selection())


def test_expand():
    board = initialize_game_state()
    current_node = Node(state=board, player=PLAYER1)
    current_node.visits = 100
    current_node.untriedMoves = [0, 5]
    assert len(current_node.untriedMoves) == 2
    assert len(current_node.childNodes) == 0
    child_node = current_node.expand(5)
    assert len(current_node.untriedMoves) == 1
    assert len(current_node.childNodes) == 1
    assert current_node.untriedMoves[0] == 0
    assert child_node.__eq__(current_node.childNodes[0])


def test_update():
    board = initialize_game_state()
    current_node = Node(state=board, player=PLAYER1)
    current_node.update(result=10)
    assert current_node.wins == 10
    assert current_node.visits == 1


def test_generate_move_MCTS():
    board = initialize_game_state()
    action, _ = generate_move_MCTS(board, PLAYER1, {})
    assert type(action) == PlayerAction


def test_MCTS():
    # Selection
    board = initialize_game_state()
    child_board = initialize_game_state()
    child_board[0, 0] = PLAYER1
    current_node = Node(state=board)
    child_node = Node(state=child_board, parent=current_node)
    current_node.untriedMoves = [0, 3, 4]
    current_node.children = [child_node]
    selected_node = Node.selection(current_node)
    assert selected_node == current_node
    # Expand
    current_node.untriedMoves = [0, 3, 4]
    explored_node = Node.expand(current_node)
    assert len(current_node.untriedMoves) == 2
    assert explored_node != current_node
    # rollout
    current_node = Node(state=board, player=PLAYER1)
    won = connected_four(current_node.state, PLAYER1)
    assert won
    #backpropagate
    Node.update(current_node, result=[-1, 1])
    assert current_node.visits == 1

    selectedColumn = MCTS(board)
    assert selectedColumn



