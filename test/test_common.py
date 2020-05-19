from typing import List
import numpy as np
from agents.common import (
    PlayerAction,
    BoardPiece,
    NO_PLAYER,
    PLAYER1,
    PLAYER2,
    initialize_game_state,
    string_to_board
)

# Define global variables for use in tests
# String representations of initial states
board1 = "|==============|\n" + \
         "|              |\n" + \
         "|              |\n" + \
         "|              |\n" + \
         "|              |\n" + \
         "|              |\n" + \
         "|              |\n" + \
         "|==============|\n" + \
         "|0 1 2 3 4 5 6 |"
board2 = "|==============|\n" + \
         "|              |\n" + \
         "|              |\n" + \
         "|    X X       |\n" + \
         "|    O X X     |\n" + \
         "|  O X O O     |\n" + \
         "|  O O X X     |\n" + \
         "|==============|\n" + \
         "|0 1 2 3 4 5 6 |"
boards: List[str] = [board1, board2]

# np.ndarray representations of initial states
game_state1 = initialize_game_state()
game_state2 = string_to_board(board2)
game_states: np.ndarray = [game_state1, game_state2]

# list of actions and players to test on initial states
actions = [PlayerAction(0), PlayerAction(3), PlayerAction(6)]
players = [PLAYER1, PLAYER2]

# games state 1 with all possible player action combo
game_state1_with_action1_player1 = game_state1.copy()
game_state1_with_action1_player1[5][0] = PLAYER1
game_state1_with_action2_player1 = game_state1.copy()
game_state1_with_action2_player1[5][3] = PLAYER1
game_state1_with_action3_player1 = game_state1.copy()
game_state1_with_action3_player1[5][6] = PLAYER1
game_state1_with_action1_player2 = game_state1.copy()
game_state1_with_action1_player2[5][0] = PLAYER2
game_state1_with_action2_player2 = game_state1.copy()
game_state1_with_action2_player2[5][3] = PLAYER2
game_state1_with_action3_player2 = game_state1.copy()
game_state1_with_action3_player2[5][6] = PLAYER2
game_state1_action_player_combo = [
    [
        game_state1_with_action1_player1,
        game_state1_with_action2_player1,
        game_state1_with_action3_player1
    ],
    [
        game_state1_with_action1_player2,
        game_state1_with_action2_player2,
        game_state1_with_action3_player2
    ]
]

# games state 2 with all possible player action combo
game_state2_with_action1_player1 = game_state2.copy()
game_state2_with_action1_player1[5][0] = PLAYER1
game_state2_with_action2_player1 = game_state2.copy()
game_state2_with_action2_player1[1][3] = PLAYER1
game_state2_with_action3_player1 = game_state2.copy()
game_state2_with_action3_player1[5][6] = PLAYER1
game_state2_with_action1_player2 = game_state2.copy()
game_state2_with_action1_player2[5][0] = PLAYER2
game_state2_with_action2_player2 = game_state2.copy()
game_state2_with_action2_player2[1][3] = PLAYER2
game_state2_with_action3_player2 = game_state2.copy()
game_state2_with_action3_player2[5][6] = PLAYER2
game_state2_action_player_combo = [
    [
        game_state2_with_action1_player1,
        game_state2_with_action2_player1,
        game_state2_with_action3_player1
    ],
    [
        game_state2_with_action1_player2,
        game_state2_with_action2_player2,
        game_state2_with_action3_player2
    ]
]

game_states_action_player_combos = [
    game_state1_action_player_combo,
    game_state2_action_player_combo
]

# end game states for both players
# Horizontal
ho_board1 = "|==============|\n" + \
            "|              |\n" + \
            "|              |\n" + \
            "|              |\n" + \
            "|    X X       |\n" + \
            "|    X X X X   |\n" + \
            "|  O X O O O   |\n" + \
            "|==============|\n" + \
            "|0 1 2 3 4 5 6 |"
ho_board2 = "|==============|\n" + \
            "|              |\n" + \
            "|              |\n" + \
            "|    X X       |\n" + \
            "|    O X X     |\n" + \
            "|  O X O O O O |\n" + \
            "|  O O X X X O |\n" + \
            "|==============|\n" + \
            "|0 1 2 3 4 5 6 |"
end_game_state1_player1 = string_to_board(ho_board1)
end_game_state1_player2 = string_to_board(ho_board2)

# Vertical
ve_board1 = "|==============|\n" + \
            "|              |\n" + \
            "|              |\n" + \
            "|    X         |\n" + \
            "|    X X       |\n" + \
            "|    X O X X   |\n" + \
            "|  O X O O O   |\n" + \
            "|==============|\n" + \
            "|0 1 2 3 4 5 6 |"
ve_board2 = "|==============|\n" + \
            "|              |\n" + \
            "|              |\n" + \
            "|  O X X       |\n" + \
            "|  O O X X     |\n" + \
            "|  O X O X O O |\n" + \
            "|  O O X X X O |\n" + \
            "|==============|\n" + \
            "|0 1 2 3 4 5 6 |"
end_game_state2_player1 = string_to_board(ve_board1)
end_game_state2_player2 = string_to_board(ve_board2)

# Diagonal \
nd_board1 = "|==============|\n" + \
            "|              |\n" + \
            "|              |\n" + \
            "|  X           |\n" + \
            "|  O X X       |\n" + \
            "|  O X X O X   |\n" + \
            "|  O X O X O   |\n" + \
            "|==============|\n" + \
            "|0 1 2 3 4 5 6 |"
nd_board2 = "|==============|\n" + \
            "|              |\n" + \
            "|O             |\n" + \
            "|X O X X       |\n" + \
            "|O O O X X     |\n" + \
            "|X X X O O X O |\n" + \
            "|X O O X X X O |\n" + \
            "|==============|\n" + \
            "|0 1 2 3 4 5 6 |"
end_game_state3_player1 = string_to_board(nd_board1)
end_game_state3_player2 = string_to_board(nd_board2)

# Diagonal /
pd_board1 = "|==============|\n" + \
            "|              |\n" + \
            "|              |\n" + \
            "|  X       X   |\n" + \
            "|  O O X X O   |\n" + \
            "|  O X X O X   |\n" + \
            "|  O X O X O   |\n" + \
            "|==============|\n" + \
            "|0 1 2 3 4 5 6 |"
pd_board2 = "|==============|\n" + \
            "|      O       |\n" + \
            "|X   O X       |\n" + \
            "|X O X X       |\n" + \
            "|O O O X X     |\n" + \
            "|X X X O O X O |\n" + \
            "|X O O X X X O |\n" + \
            "|==============|\n" + \
            "|0 1 2 3 4 5 6 |"
end_game_state4_player1 = string_to_board(pd_board1)
end_game_state4_player2 = string_to_board(pd_board2)

end_game_states = [
    [end_game_state1_player1, end_game_state1_player2],
    [end_game_state2_player1, end_game_state2_player2],
    [end_game_state3_player1, end_game_state3_player2],
    [end_game_state4_player1, end_game_state4_player2]
]

# Draw State
draw_board = "|==============|\n" + \
             "|O O X O X O X |\n" + \
             "|X X O O X X X |\n" + \
             "|O O X X O X O |\n" + \
             "|X X O X X O O |\n" + \
             "|X O X O O X X |\n" + \
             "|O O O X X O O |\n" + \
             "|==============|\n" + \
             "|0 1 2 3 4 5 6 |"

draw_state = string_to_board(draw_board)


def test_initialize_game_state():
    from agents.common import initialize_game_state

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == BoardPiece
    assert ret.shape == (6, 7)
    assert np.all(ret == NO_PLAYER)


def test_pretty_print_board():
    from agents.common import pretty_print_board

    pretty_boards = [
        pretty_print_board(game_state) for game_state in game_states
    ]

    for index, board in enumerate(pretty_boards):
        assert isinstance(board, str)
        assert board == boards[index]


def test_string_to_board():
    from agents.common import string_to_board

    game_states_from_string = [string_to_board(board) for board in boards]

    for index, game_state in enumerate(game_states_from_string):
        assert isinstance(game_state, np.ndarray)
        assert (game_state == game_states[index]).all


def test_apply_player_action():
    from agents.common import apply_player_action

    for index1, game_state in enumerate(game_states):
        for index2, player in enumerate(players):
            for index3, action in enumerate(actions):
                # Test with copying
                ret = apply_player_action(
                    game_state, action, player, copy=True
                )
                assert isinstance(ret, np.ndarray)
                assert (game_state == ret).all
                sta = game_states_action_player_combos[index1][index2][index3]
                assert (sta == ret).all
                # Test without copying
                ret = apply_player_action(game_state, action, player)
                assert isinstance(ret, np.ndarray)
                assert (game_state != ret).all
                sta = game_states_action_player_combos[index1][index2][index3]
                assert (sta == ret).all


def test_connected_four():
    from agents.common import connected_four

    for game_state in game_states:
        for player in players:
            assert not connected_four(game_state, player)

    for end_game_state in end_game_states:
        for index, player in enumerate(players):
            assert connected_four(end_game_state[index], player)


def test_check_end_state():
    from agents.common import check_end_state, GameState

    for game_state in game_states:
        for player in players:
            state = check_end_state(game_state, player)
            assert state == GameState.STILL_PLAYING

    for end_game_state in end_game_states:
        for index, player in enumerate(players):
            state = check_end_state(end_game_state[index], player)
            assert state == GameState.IS_WIN

    for player in players:
        assert check_end_state(draw_state, player) == GameState.IS_DRAW