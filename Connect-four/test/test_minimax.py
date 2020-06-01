from typing import List
import numpy as np
import math
from agents.agent_minimax.minimax import (
    minimax,
    heuristic,
    Score_func,
    generate_move_minimax
)
from agents.common import BoardPiece, PLAYER2, PLAYER1, NO_PLAYER, GameState, initialize_game_state


def test_generate_move_minimax():
    board = initialize_game_state()
    ret = generate_move_minimax(board, PLAYER1, board)
    assert ret == (3, board)


def test_Score_func():
    board = initialize_game_state()
    # horizontal
    board[0, 0] = PLAYER1
    board[0, 1] = PLAYER1
    board[0, 2] = PLAYER1
    board[0, 3] = PLAYER1
    ret = Score_func(list(board[0, 0:4]), PLAYER1)

    assert ret == 100

    # vertical
    board[0, 6] = PLAYER1
    board[1, 6] = PLAYER1
    board[2, 6] = PLAYER1
    board[3, 6] = PLAYER1
    ret = Score_func(list(board[0:4, 6]), PLAYER1)

    assert ret == 100

    # diagonal
    board = initialize_game_state()
    board[0, 0] = PLAYER1
    board[1, 1] = PLAYER1
    board[2, 2] = PLAYER1
    pos_list = (board[0, 0], board[1, 1], board[2, 2], board[3, 3])
    ret = Score_func(pos_list, PLAYER1)

    assert ret == 5

    # diagonal
    board[0, 6] = PLAYER1
    board[1, 5] = PLAYER1
    pos_list = (board[0, 6], board[1, 5], board[2, 4], board[3, 3])
    ret = Score_func(pos_list, PLAYER1)

    assert ret == 2

    # block the opponent
    board[2, 4] = PLAYER1
    pos_list = (board[0, 6], board[1, 5], board[2, 4], board[3, 3])
    ret = Score_func(pos_list, PLAYER2)

    assert ret == -4


def test_heuristic():
    board = initialize_game_state()

    # should return 3 for first move
    ret = heuristic(board, PLAYER1)

    assert ret == 0

    board[0, 0] = PLAYER1
    board[0, 1] = PLAYER1
    board[0, 2] = PLAYER1

    ret = heuristic(board, PLAYER1)

    score = Score_func(list(board[0, 0:4]), PLAYER1) + Score_func(list(board[0, 1:5]), PLAYER1)

    assert score == ret


def test_minimax():
    board = initialize_game_state()

    # center column 3 = first move
    assert minimax(board, 4, -math.inf, math.inf, PLAYER1, True) == (3, 6)
