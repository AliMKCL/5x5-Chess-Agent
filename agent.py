import random
from extension.board_utils import list_legal_moves_for

def agent(board, player, var):
    # Hints:
    # List of players on the current board game: list(board.players) - default list: [Player (white), Player (black)]
    # board.players[0].name = "white" and board.players[1].name = "black"
    # Name of the player assigned to the agent (either "white" or "black"): player.name
    # list of pieces of the current player: list(board.get_player_pieces(player))
    # List of pieces and corresponding moves for each pieces of the player: piece, move_opt = list_legal_moves_for(board, player)
    # List of legal move for a corresponding pieces: piece.get_move_options()
    piece, move_opt = None, None
    
    if player.name == "white":
        legal = list_legal_moves_for(board, player)
        if legal:
            # Randome choice from a list of legal move from all pieces
            piece, move_opt = random.choice(legal)
    else:
        legal = list_legal_moves_for(board, player)
        if legal:
            # Randome choice from a list of legal move from all pieces
            piece, move_opt = random.choice(legal)

    return piece, move_opt