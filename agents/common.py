from enum import Enum
from typing import Optional
from typing import Callable, Tuple
import numpy as np


BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER, the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece
PlayerAction = np.int8  # The column to be played
HEIGHT = 6
WIDTH = 7



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
        new_board = "|"
        for position in row:
            if position == NO_PLAYER:
                new_board += "  "
            if position == PLAYER1:
                new_board += "X"
            if position == PLAYER2:
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
    new_board = board.copy() if copy else board
    next_row = np.count_nonzero(new_board, axis=0)[action] - 1
    new_board[next_row][action] = player
    return new_board

def connected_four(
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """
    bools = board == player

    def verticalSeq(row, col):
        """Return True if it found a vertical sequence with the required length
        """
        count = 0
        for rowIndex in range(row, HEIGHT):
            if board[rowIndex][col] == board[row][col]:
                count += 1
            else:
                break
        if count >= WIDTH:
            return True
        else:
            return False

    def horizontalSeq(row, col):
        """Return True if it found a horizontal sequence with the required length
        """
        count = 0
        for colIndex in range(col, WIDTH):
            if board[row][colIndex] == board[row][col]:
                count += 1
            else:
                break
        if count >= WIDTH:
            return True
        else:
            return False

    def negDiagonalSeq(row, col):
        """Return Ture if it found a negative diagonal sequence with the required length
        """
        count = 0
        colIndex = col
        for rowIndex in range(row, -1, -1):
            if colIndex > HEIGHT:
                break
            elif board[rowIndex][colIndex] == board[row][col]:
                count += 1
            else:
                break
            colIndex += 1 # increment column when row is incremented
        if count >= WIDTH:
            return True
        else:
            return False

    def posDiagonalSeq(row, col):
        """Return True if it found a positive diagonal sequence with the required length
        """
        count = 0
        colIndex = col
        for rowIndex in range(row, HEIGHT):
            if colIndex > HEIGHT:
                break
            elif board[rowIndex][colIndex] == board[row][col]:
                count += 1
            else:
                break
            colIndex += 1  # increment column when row incremented
        if count >= WIDTH:
            return True
        else:
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


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]
