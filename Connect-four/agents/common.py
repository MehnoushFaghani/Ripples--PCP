from enum import Enum
from typing import Optional
from typing import Callable, Tuple
import numpy as np
from scipy.signal.sigtools import _convolve2d
from numba import njit
import random as rnd


disable_jit = True
if disable_jit:
    import os
    os.environ['NUMBA_DISABLE_JIT'] = '1'


BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER, the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece
PlayerAction = np.int8  # The column to be played
HEIGHT = 6
WIDTH = 7
CONNECT_N = 4


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    legal_players = ["X", "O"]
    board = [[] for x in range(WIDTH)]
    return np.full((HEIGHT, WIDTH), NO_PLAYER)


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output:
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """

    new_board = "|==============|\n"

    for row in board:
        new_board += "|"
        for position in row:
            if position == NO_PLAYER:
                new_board += "  "
            elif position == PLAYER1:
                new_board += "X"
            elif position == PLAYER2:
                new_board += "O "
        new_board += "|\n"
    new_board += "|==============|\n"
    new_board += "|0 1 2 3 4 5 6 |\n"
    return new_board


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.
    """
    lines = pp_board.split("\n")
    game_state = np.zeros((HEIGHT, WIDTH))
    for ind, row in enumerate(lines[1:WIDTH]):
        build_row = np.zeros(WIDTH)
        for ind2, char in enumerate(row[:-1]):
            if char == '' and ind2 % 2 == 1:
                build_row[int((ind2 - 1) / 2)] = NO_PLAYER
            elif char == 'X':
                build_row[int((ind2 - 1) / 2)] = PLAYER1
            elif char == 'O':
                build_row[int((ind2 - 1) / 2)] = PLAYER2
        game_state[ind] = build_row
    return game_state


def apply_player_action(
    board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """

    if copy:
        board_copy = np.copy(board)

    for row in range(HEIGHT):
        if board[row, action] == NO_PLAYER:
            board[row, action] = player
            break

    if copy:
        return board_copy, board
    else:
        return board


col_kernel = np.ones((CONNECT_N, 1), dtype=BoardPiece)
row_kernel = np.ones((1, CONNECT_N), dtype=BoardPiece)
dia_l_kernel = np.diag(np.ones(CONNECT_N, dtype=BoardPiece))
dia_r_kernel = np.array(np.diag(np.ones(CONNECT_N, dtype=BoardPiece))[::-1, :])


def connected_four(
    board: np.ndarray, player: BoardPiece, _last_action: Optional[PlayerAction] = None
) -> bool:
    board = board.copy()

    other_player = BoardPiece(player % 2 + 1)
    board[board == other_player] = NO_PLAYER
    board[board == player] = BoardPiece(1)

    for kernel in (col_kernel, row_kernel, dia_l_kernel, dia_r_kernel):
        result = _convolve2d(board, kernel, 1, 0, 0, BoardPiece(0))
        if np.any(result == CONNECT_N):
            return True
    return False


def check_end_state(
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    if connected_four(board, player):
        return GameState.IS_WIN
    elif np.count_nonzero(board) == HEIGHT*WIDTH:
        return GameState.IS_DRAW
    else:
        return GameState.STILL_PLAYING
