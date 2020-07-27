import numpy as np
from typing import Optional
import time
from agents.common import BoardPiece, PlayerAction, SavedState, PLAYER1, PLAYER2, NO_PLAYER, GameState, HEIGHT, WIDTH
from agents.common import connected_four, check_end_state, apply_player_action, find_columns

PLAYER = NO_PLAYER
OPPONENT = NO_PLAYER
GLOBAL_TIME = 20


# Nodes of MCTS
class Node:
    # Data structure to keep track of our search
    def __init__(self, move=None, parent=None, state=None, player=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.player = player
        self.untriedMoves = self.getMoves()
        self.childNodes = []
        self.wins = 0
        self.visits = 0

    def getMoves(self) -> np.ndarray:
        """
        returns an array of available moves
        """
        if check_end_state(self.state, self.player) == GameState.IS_WIN:
            return np.array([])  # if terminal state, return empty an array
        else:
            return np.array(find_columns(self.state))

    def selection(self):
        """
        return child with largest UCT value
        """
        bestScore = -10000000.0
        bestChildren = None

        for child in self.childNodes:
            score = child.wins / child.visits + np.sqrt(2) * np.sqrt(
                np.log(self.visits) / child.visits)
            if score > bestScore:
                bestChildren = child
                bestScore = score
        return bestChildren

    def expand(self, action: PlayerAction):
        """
        return child when move is taken
        """
        opponent = PLAYER1
        if self.player == PLAYER1:
            opponent = PLAYER2
        new_board, original_board = apply_player_action(
            self.state.copy(), action, opponent)
        child = Node(move=action, parent=self,
                     state=new_board, player=opponent)
        self.childNodes.append(child)
        # remove move from current node
        self.untriedMoves = np.setdiff1d(self.untriedMoves, action)

        return child

    def update(self, result: int):
        """
        Update the win and visits value of a node
        """
        self.wins += result
        self.visits += 1


# Monte Carlo Tree Search
def generate_move_MCTS(board: np.ndarray, player: BoardPiece,
                       saved_state: Optional[SavedState]) \
        -> object:

    global PLAYER
    global OPPONENT

    PLAYER = player
    if PLAYER == PLAYER1:
        OPPONENT = PLAYER2
    else:
        OPPONENT = PLAYER1

    action = MCTS(board)
    return PlayerAction(action), SavedState()


def MCTS(board: np.ndarray) -> PlayerAction:

    rootNode = Node(state=board, player=PLAYER)

    global GLOBAL_TIME
    end = time.time() + GLOBAL_TIME
    while time.time() < end:

        node = rootNode

        #############
        # selection #
        #############
        # keep going down the tree based on best UCT values until terminal or unexpanded node
        while not np.any(node.untriedMoves) and node.childNodes != []:
            node = node.selection()

        #############
        #  Expand   #
        #############
        if np.any(node.untriedMoves):
            # Choose a random action from available moves
            action = np.random.choice(node.untriedMoves)
            node = node.expand(action)

        #############
        #  rollout  #
        #############
        board = node.state.copy()
        win_game_flag = False
        currentPlayer = node.player
        while np.any(find_columns(board)) and not win_game_flag:
            if currentPlayer == PLAYER2:
                currentPlayer = PLAYER1
            else:
                currentPlayer = PLAYER2
            action = np.random.choice(find_columns(board))
            board, _ = apply_player_action(board, action, currentPlayer)
            win_game_flag = connected_four(board, currentPlayer)

        #################
        # backpropagate #
        #################
        if win_game_flag:
            if currentPlayer == PLAYER:
                result = 1  # The player won
            else:
                result = -1  # The player lost against the opponent
        else:
            result = 0
        while node is not None:
            node.update(result)
            node = node.parent

    bestScore = -10000000.0
    selectedColumn = - 1
    for child in rootNode.childNodes:
        if connected_four(child.state, child.player):
            return child.move
        else:
            score = child.wins / child.visits
            if score > bestScore:
                selectedColumn = child.move
                bestScore = score
    return selectedColumn
