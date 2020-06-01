import numpy as np
import math
from typing import Optional, Tuple
from agents.common import BoardPiece, PlayerAction, SavedState, PLAYER1, PLAYER2, NO_PLAYER, GameState, HEIGHT, WIDTH
from agents.common import connected_four, check_end_state, apply_player_action


def generate_move_minimax(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    alpha = -math.inf
    beta = math.inf
    depth = 4

    # Choose a valid, non-full column that maximizes score and return it as `action`
    PlayerAction = minimax(board, depth, alpha, beta, player, True)[0]

    return PlayerAction, saved_state


def find_moves(board: np.ndarray) -> list:
    '''
    Find the all empty columns which equal to NO_PLAYER
    gets board as an input
    '''
    return np.argwhere(board[board.shape[0] - 1, :] == NO_PLAYER).flatten()


def Score_func(score_four: list, player: BoardPiece) -> int:
    '''
    computing scores depend on different move
    :return: computed score
    '''

    score = 0

    # check which player score to maximize and which player to block
    if player == PLAYER1:
        opponent_player = PLAYER2
    else:
        opponent_player = PLAYER1

    # check if agent (player) is close to getting a win by placing 4 adjacent pieces
    if score_four.count(player) == 4:
        score += 100
    elif score_four.count(player) == 3 and score_four.count(NO_PLAYER) == 1:
        score += 5
    elif score_four.count(player) == 2 and score_four.count(NO_PLAYER) == 2:
        score += 2

    # block opponent from getting a win
    if score_four.count(opponent_player) == 3 and score_four.count(NO_PLAYER) == 1:
        score -= 4

    return score


def heuristic(board: np.ndarray, player: BoardPiece) -> int:
    '''
	Calculates score considering 4 adjacent spots of the board in each row, column, and diagonal
	(checks how many empty and filled spots there are in 4 adjacent spots in all directions)
	:param board: current state of board
	:param player: player who wants to maximize score
	:return: score that can be achieve by playing open position
	'''
    num_rows = board.shape[0]
    num_columns = board.shape[1]

    score = 0

    # Score center column
    center_array = [int(i) for i in list(board[:, num_columns // 2])]
    center_count = center_array.count(player)
    score += center_count * 3

    # first check horizontal
    for row in range(num_rows):  # loop through every row
        row = board[row, :]
        # now loop through each column and check 4 adjacent spots
        for col in range(num_columns - 3):  # only need to loop through first 4 cols, since adjacent_4 would touch (j+3)
            score_four = list(row[col:col + 4])  # convert to list to apply count() lalter
            # now count the number of pieces for each player
            score += Score_func(score_four, player)

    # vertical
    for col in range(num_columns):
        col = board[:, col]
        for row in range(num_rows - 3):
            score_four = list(col[row:row + 4])
            score += Score_func(score_four, player)

    # score positive sloped diagonal
    for row in range(num_rows - 3):  # rows
        for col in range(num_columns - 3):  # cols
            score_four = [board[row + i, col + i] for i in range(4)]
            score += Score_func(score_four, player)

    # score negative diagonal
    for row in range(num_rows - 3):  # rows
        for col in range(num_columns - 3):  # cols
            score_four = [board[row + 3 - i, col + i] for i in range(4)]  # col increases but row decreases
            score += Score_func(score_four, player)

    return score


def minimax(board: np.ndarray, depth: int, alpha: int, beta: int, player: BoardPiece, maximizing_player: bool) -> Tuple[
    int, int]:
    # check which player is the agent so that we don't max/min for wrong player
    if player == PLAYER1:
        opponent = PLAYER2
    else:
        opponent = PLAYER1

    # check NO_PLAYER columns
    finding_moves = find_moves(board)

    # check if depth is 0
    if depth == 0:
        score = heuristic(board, player)
        return None, score

    # check if we're at a leaf/terminal node
    if check_end_state(board, player) != GameState.STILL_PLAYING:
        if connected_four(board, player):  # agent won
            return None, 10000000
        if connected_four(board, opponent):  # opponent won
            return None, -10000000
        else:  # must be a draw
            return None, 0

    if maximizing_player:  # get max score for agent
        score = -math.inf
        for column in finding_moves:
            board, board_copy = apply_player_action(board, column, player, True)
            next_score = minimax(board_copy, depth - 1, alpha, beta, player, False)[1]
            if next_score > score:
                score = next_score
                action_column = column
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return action_column, score

    else:
        score = math.inf
        for column in finding_moves:
            board, action_board = apply_player_action(board, column, opponent, True)
            next_score = minimax(action_board, depth - 1, alpha, beta, player, True)[1]
            if next_score < score:
                score = next_score
                action_column = column
            beta = min(beta, score)  # get min score for opponent
            if alpha >= beta:
                break
        return action_column, score
