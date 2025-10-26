import random
import time
from extension.board_utils import list_legal_moves_for
from chessmaker.chess.pieces import King, Queen, Bishop, Knight, Pawn
from extension.piece_right import Right
from extension.board_rules import get_result

PIECE_VALUES = {
    'pawn': 100,
    'knight': 330,
    'bishop': 320,
    'right': 500,
    'queen': 900,
    'king': 20000
}


PAWN_TABLE = [
    [10,  10,  10,  10,  10],
    [5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5],
    [-5,  5, -5, 5,  0],
    [0,  0,  0,  0,  0]
]

KNIGHT_TABLE = [
    [-5,  -5,  -5,  -5,  -5],
    [-5, 0, 0, 0, -5],
    [-5, 0, 0, 0, -5],
    [-5,  0, 0, 0,  -5],
    [-5,  -5,  -5,  -5,  -5]
]

BISHOP_TABLE = [
    [-10,  -5,  -5,  -5,  -10],
    [-5, 0, 0, 0, -5],
    [-5, 0, 5, 0, -5],
    [-5,  0, 0, 0,  -5],
    [-10,  -5,  -5,  -5,  -10]
]

RIGHT_TABLE = [
    [-5,  5,  5,  5,  -5],
    [0, 5, 5, 5, 0],
    [0, 0, 0, 0, 0],
    [0,  0, 0, 0,  0],
    [-5,  0,  0,  0,  -5]
]

QUEEN_TABLE = [
    [-5,  0,  0,  0,  -5],
    [0, 5, 5, 5, 0],
    [0, 5, 5, 5, 0],
    [0,  5, 5, 5,  0],
    [-5,  0,  0,  0,  -5]
]

KING_TABLE = [
    [-20, -20, -20, -20, -20],
    [-15, -15, -15, -15, -15],
    [-10, -10, -10, -10, -10],
    [-5, -5, -5, -5, -5],
    [5, 5, 5, 5, 5]
]

KING_TABLE_ENDGAME = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

def attacker_defender_ratio(board, target_position, attacking_player, defending_player):
    """
    Analyzes the attacker/defender balance for a given square and calculates exchange outcomes.
    
    Args:
        board: Current board state
        target_position: The position to analyze (Position object with x, y)
        attacking_player: The player whose pieces might attack this square
        defending_player: The player whose pieces might defend this square
    
    Returns:
        (num_diff, val_diff): Tuple containing:
            - num_diff: Number of attackers minus defenders (NOT including piece on square)
            - val_diff: Material outcome of the exchange sequence
                       - None if num_diff <= 0 (defenders hold or outnumber attackers)
                       - Otherwise: net material gain/loss after complete exchange
    
    Logic:
        - If defenders >= attackers: Exchange won't happen or defenders win
        - If attackers > defenders: Calculate exchange assuming optimal play
          Only count pieces that will actually be traded (exclude excess attackers)
    
    Example:
        Square has: Pawn (defending)
        Defenders: Pawn (100)
        Attackers: Pawn (100), Knight (330), Queen (900)
        
        num_diff = 3 - 1 = 2 (2 excess attackers)
        Exchange sequence (excluding top 2 attackers: Knight, Queen):
          1. Pawn captures Pawn: +100 (attacker wins pawn on square)
          2. Pawn recaptures: -100 (attacker loses attacking pawn)
          3. No more defenders, exchange ends
        val_diff = +100 - 100 = 0
    """
    # ===== STEP 1: Find the piece on the target square =====
    target_piece = None
    for piece in board.get_pieces():
        if piece.position == target_position:
            target_piece = piece
            break
    
    # Get all pieces for both players
    attacking_pieces = board.get_player_pieces(attacking_player)
    defending_pieces = board.get_player_pieces(defending_player)
    
    # ===== STEP 2: Collect all attackers and their values (OPTIMIZED) =====
    attacker_list = []  # List of (piece, value) tuples
    
    for piece in attacking_pieces:
        # OPTIMIZATION: Cache move options to avoid repeated calls
        move_options = piece.get_move_options()
        
        # Filter attacking pieces
        for move in move_options:
            # Check for access to cell or capturability of cell
            if ((hasattr(move, 'position') and move.position == target_position) or
                (hasattr(move, 'captures') and move.captures and target_position in move.captures)):
                    attacker_list.append((piece, get_piece_value(piece)))
                    break   # Stop checking moves for this piece

    # ===== STEP 3: Collect all defenders and their values (OPTIMIZED) =====
    defender_list = []  # List of (piece, value) tuples
    
    for piece in defending_pieces:
        # Skip the piece on the target square - it can't defend itself
        if piece.position == target_position:
            continue
        
        # OPTIMIZATION: Cache move options to avoid repeated calls
        move_options = piece.get_move_options()
        
        # Filter defending pieces
        for move in move_options:
            # Check for access to cell or capturability of cell
            if ((hasattr(move, 'position') and move.position == target_position) or
                (hasattr(move, 'captures') and move.captures and target_position in move.captures)):
                defender_list.append((piece, get_piece_value(piece)))
                break  # Stop checking moves for this piece
    
    # ===== STEP 4: Calculate num_diff =====
    num_attackers = len(attacker_list)
    num_defenders = len(defender_list)
    num_diff = num_attackers - num_defenders
    
    # ===== STEP 5: If defenders hold or outnumber, exchange won't favor attackers =====
    if num_diff <= 0:
        return (num_diff, None)
    
    # ===== STEP 6: Calculate exchange value (attackers outnumber defenders) =====
    # Sort attackers by value (lowest to highest) - use cheapest pieces first
    attacker_list.sort(key=lambda x: x[1])
    
    # Sort defenders by value (lowest to highest) - defenders respond with cheapest
    defender_list.sort(key=lambda x: x[1])
    
    # Exclude the top 'num_diff' most valuable attackers (they won't be traded)
    # These pieces will still be standing after the exchange
    attackers_in_exchange = attacker_list[:-num_diff] if num_diff > 0 else attacker_list
    
    # Start exchange calculation
    material_gained = 0
    material_lost = 0
    
    # Add the initial capture (target piece value, if it exists and belongs to defender)
    if target_piece and target_piece.player == defending_player:
        material_gained += get_piece_value(target_piece)
    
    # ===== STEP 7: Simulate the exchange sequence =====
    # Exchange proceeds alternately: attacker captures, defender recaptures, etc.
    # Example: 2 pawns attack a pawn, defended by a pawn.
    #
    # Start with attacker capturing, as in the end attacker will stay onn top in this case
    #   1. Attacker's pawn captures target pawn: +100 gained
    #        Iterate through the remaining exchange:
    #           2. Defender's pawn recaptures: -100 lost
    #           3. Attacker's pawn recaptures: +100 gained
    #   Net: +100 - 100 + 100 = +100
    
    exchange_length = min(len(attackers_in_exchange), len(defender_list))
    
    # Where exchange is defender recaptures, attacker recaptures.
    for i in range(exchange_length):
        # The first capturing piece is lost/defender recaptures it (in the exchange)
        material_lost += attackers_in_exchange[i][1]
        
        # The piece defendered recaptured with is captured by attacker
        material_gained += defender_list[i][1]

    
    # Calculate net material difference
    val_diff = material_gained - material_lost
    
    return (num_diff, val_diff)

def get_piece_value(piece):
    return PIECE_VALUES.get(piece.name.lower(), 0)

def get_positional_value(piece, is_white, board=None):
    x, y = piece.position.x, piece.position.y
    if not is_white:
        y = 4 - y
    if piece.name.lower() == 'pawn':
        return PAWN_TABLE[y][x]
    elif piece.name.lower() == 'knight':
        return KNIGHT_TABLE[y][x]
    elif piece.name.lower() == 'king':
        # Use endgame king table if player has 4 or fewer pieces
        if board is not None:
            player_piece_count = sum(1 for p in board.get_pieces() if p.player == piece.player)
            if player_piece_count <= 4:
                return KING_TABLE_ENDGAME[y][x]
        return KING_TABLE[y][x]
    elif piece.name.lower() == 'bishop':
        return BISHOP_TABLE[y][x]
    elif piece.name.lower() == 'right':
        return RIGHT_TABLE[y][x]
    elif piece.name.lower() == 'queen':
        return QUEEN_TABLE[y][x]
    return 0

def is_stalemate(board):
    """
    Helper function to determine if a move results in a stalemate.
        
    Args:
        board: The current board state.

    Returns:
        True if the move results in a stalemate (draw), False otherwise.
    """
    result = get_result(board)
    
    # If result is None, there's no stalemate
    if result is None:
        return False
    
    res = result.lower()
    # Check if the result indicates a stalemate or draw
    if (res == "stalemate (no more possible moves) - black loses" or 
        res == "stalemate (no more possible moves) - white loses" or 
        res == "draw - only 2 kings left" or 
        res == "draw - fivefold repetition"):
        return True
    
    return False