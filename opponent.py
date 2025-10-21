import random
from extension.board_utils import list_legal_moves_for

def opponent(board, player, var):
    # Hints:
    # List of players on the current board game: list(board.players) - default list: [Player (white), Player (black)]
    # board.players[0].name = "white" and board.players[1].name = "black"
    # Name of the player assigned to the opponent (either "white" or "black"): player.name
    # list of pieces of the current player: list(board.get_player_pieces(player))
    # List of pieces and corresponding moves for each pieces of the player: piece, move_opt = list_legal_moves_for(board, player)
    piece, move_opt = None, None
    
    if player.name == "white":

        while not move_opt:
            piece = random.choice(list(board.get_player_pieces(player)))
            mov = piece.get_move_options()
            if mov:
                move_opt = random.choice(mov)
                break
    else:
       
        while not move_opt:
            piece = random.choice(list(board.get_player_pieces(player)))
            mov = piece.get_move_options()
            if mov:
                move_opt = random.choice(mov)
                break

    return piece, move_opt
