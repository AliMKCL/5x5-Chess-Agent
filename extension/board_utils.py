FILE_NAME = "notes_agent.txt"

def print_board_ascii(board):
    """
    Display the current board state in a simple ASCII format.
    
    Parameters
    ----------
    board: The given game board object

    Returns
    -------
    None
    """
    piece_map = {"pawn": "P", "right": "R", "knight": "N", "bishop": "B", "queen": "Q", "king": "K"}
    grid = [["." for _ in range(5)] for _ in range(5)]
    for piece in board.get_pieces():
        pos = piece.position
        ch = piece_map.get(piece.name.lower(), "?")
        grid[pos.y][pos.x] = ch.upper() if piece.player.name.lower() == "white" else ch.lower()
    print("  0 1 2 3 4")
    for row in (range(5)):
        print(f"{row} " + " ".join(grid[row]))

def list_legal_moves_for(board, player):
    """
    Retrieve all legal move options for a given player on the board.

    Parameters
    ----------
    board: The given game board object
    player: The player whose legal moves are to be listed

    Returns
    -------
    pairs: A list of (piece, move) pairs
    """
    pairs = []
    for pc in board.get_player_pieces(player):
        for opt in pc.get_move_options():
            pairs.append((pc, opt))
    return pairs

def copy_piece_move(board, piece, move):
    """"
    Find and return the corresponding temp_piece and temp_move object on the given board.

    This function searches for a piece on the board that matches the input piece
    and then finds the equivalent move within that piece's move options.

    Parameters
    ----------
    board: The given game board object
    piece: The original piece object whose equivalent is to be found on the board.
    move: The original move object (associated piece) to match

    Returns
    -------
    temp_piece: the matching piece found (on the given game board)
    temp_move: the matching move option for that piece (on the given game board)
    """
    try:
        if piece and move:
            for tp in board.get_player_pieces(piece.player):
                if type(tp) is type(piece) and tp.position == piece.position:
                    temp_piece = tp
                    break
            if temp_piece is None:
                return board, None, None
            # find the equivalent move option on the cloned piece
            dest = getattr(move, "position", None)
            temp_move = None
            for m in temp_piece.get_move_options():
                m_dest = getattr(m, "position", None)
                if m_dest == dest:
                    temp_move = m
                    return board, temp_piece, temp_move
            return board, temp_piece, None
        else:
            return board, None, None
    except Exception:
        return board, None, None

def take_notes(note):
    """"
    This function allows you to append a note (any variable or message) when running agent.py
    For example: take_notes(move_opt)

    Parameters
    ----------
    note: The variable or message to be noted

    Returns
    -------
    None

    Example
    -------
    take_notes("Agent started evaluating moves...")
    take_notes(best_move)
    
    Use this function with care, taking many notes slightly increase
    the processing time of of your agent.py
    """
    try:
        with open(FILE_NAME, 'a', encoding='utf-8') as file:
            file.write(str(note) + '\n')
    except Exception as e:
        pass