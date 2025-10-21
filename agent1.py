"""
COMP2321 Chess Agent Implementation
====================================
This agent uses minimax algorithm with alpha-beta pruning and iterative deepening
to play chess on a 5x5 board with custom pieces.

Main Algorithm: Minimax with Alpha-Beta Pruning
Optimization: Iterative Deepening with Time Management
Enhancement: Move Ordering for better pruning efficiency
"""

import random
import time
from extension.board_utils import list_legal_moves_for
from chessmaker.chess.pieces import King, Queen, Bishop, Knight, Pawn
from extension.piece_right import Right

# ============================================================================
# EVALUATION CONSTANTS
# ============================================================================

# Piece values for material evaluation
# These values determine how much each piece is worth in centipawns (1 pawn = 100)
PIECE_VALUES = {
    'pawn': 100,      
    'knight': 320,    
    'bishop': 330,    
    'right': 500,     
    'queen': 900,     
    'king': 20000     # Infinite value - losing king means losing game
}

# Position tables for piece positioning bonuses (white perspective)
PAWN_TABLE = [
    [0,  0,  0,  0,  0],
    [50, 50, 50, 50, 50],
    [10, 10, 20, 30, 10],
    [5,  5, 10, 25,  5],
    [0,  0,  0, 0,  0]
]

KNIGHT_TABLE = [
    [-50, -40, -30, -40, -50],
    [-40, -20,   0, -20, -40],
    [-30,   0,  10,   0, -30],
    [-40, -20,   0, -20, -40],
    [-50, -40, -30, -40, -50]
]

KING_TABLE = [
    [-30, -40, -40, -40, -30],
    [-30, -40, -40, -40, -30],
    [-30, -40, -40, -40, -30],
    [-20, -30, -30, -30, -20],
    [ 0,  0,   10,   10,  0]
]

def get_piece_value(piece):
    """Get the base value of a piece"""
    piece_name = piece.name.lower()
    return PIECE_VALUES.get(piece_name, 0)

def get_positional_value(piece, is_white):
    """Get positional bonus for a piece"""
    piece_name = piece.name.lower()
    x, y = piece.position.x, piece.position.y
    
    # Flip y coordinate for black pieces
    if not is_white:
        y = 4 - y
    
    if piece_name == 'pawn':
        return PAWN_TABLE[y][x]
    elif piece_name == 'knight':
        return KNIGHT_TABLE[y][x]
    elif piece_name == 'king':
        return KING_TABLE[y][x]
    
    return 0

def evaluate_board(board, player_name):
    """
    Evaluate the board position from the perspective of player_name
    Positive score = good for player, negative = bad for player
    """
    score = 0
    player_pieces = []
    opponent_pieces = []
    
    # Categorize pieces
    for piece in board.get_pieces():
        if piece.player.name == player_name:
            player_pieces.append(piece)
        else:
            opponent_pieces.append(piece)
    
    # Material and positional evaluation    ???
    for piece in player_pieces:
        is_white = piece.player.name == "white"
        score += get_piece_value(piece)
        score += get_positional_value(piece, is_white) * 0.1
    
    for piece in opponent_pieces:
        is_white = piece.player.name == "white"
        score -= get_piece_value(piece)
        score -= get_positional_value(piece, is_white) * 0.1
    
    # Mobility bonus - more moves = better position
    player_moves = len(list_legal_moves_for(board, board.current_player if board.current_player.name == player_name else [p for p in board.players if p.name == player_name][0]))
    score += player_moves * 10
    
    # King safety - penalize exposed king
    for piece in player_pieces:
        if isinstance(piece, King):
            # Check if king is near the back rank (safer)
            if piece.player.name == "white" and piece.position.y == 4:
                score += 50
            elif piece.player.name == "black" and piece.position.y == 0:
                score += 50
    
    return score

def order_moves(board, moves, player_name):
    """
    Order moves to improve alpha-beta pruning efficiency
    Prioritize: captures, checks, then other moves
    """
    scored_moves = []
    
    for piece, move in moves:
        score = 0
        
        # Prioritize captures (check if move has captures attribute)
        if hasattr(move, 'captures') and move.captures:
            # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
            for capture_pos in move.captures:
                for target in board.get_pieces():
                    if target.position == capture_pos:
                        score += get_piece_value(target) * 10 - get_piece_value(piece)
                        break
        
        # Prioritize pawn promotion
        if isinstance(piece, Pawn):
            dest_y = move.position.y
            if (piece.player.name == "white" and dest_y == 0) or (piece.player.name == "black" and dest_y == 4):
                score += 800
        
        # Prioritize moving to center
        center_distance = abs(move.position.x - 2) + abs(move.position.y - 2)
        score += (8 - center_distance) * 3
        
        scored_moves.append((score, piece, move))
    
    # Sort by score descending
    scored_moves.sort(reverse=True, key=lambda x: x[0])
    return [(piece, move) for _, piece, move in scored_moves]

def is_terminal(board):
    """Check if the game is over"""
    from extension.board_rules import get_result
    return get_result(board) is not None

def minimax(board, depth, alpha, beta, maximizing, player_name, start_time, time_limit):
    """
    Minimax algorithm with alpha-beta pruning
    """
    # Time check
    if time.time() - start_time > time_limit:
        return None
    
    # Terminal node or depth limit
    if depth == 0 or is_terminal(board):
        return evaluate_board(board, player_name)
    
    current_player = board.current_player
    is_maximizing = (current_player.name == player_name)
    
    legal_moves = list_legal_moves_for(board, current_player)
    
    if not legal_moves:
        # No legal moves available
        return evaluate_board(board, player_name)
    
    # Order moves for better pruning
    legal_moves = order_moves(board, legal_moves, player_name)
    
    if is_maximizing:
        max_eval = float('-inf')
        for piece, move in legal_moves:
            # Make move on a cloned board
            new_board = board.clone()
            
            # Find corresponding piece and move on cloned board
            new_piece = None
            for p in new_board.get_player_pieces(current_player):
                if type(p) == type(piece) and p.position == piece.position:
                    new_piece = p
                    break
            
            if new_piece:
                new_move = None
                for m in new_piece.get_move_options():
                    if hasattr(m, 'position') and m.position == move.position:
                        new_move = m
                        break
                
                if new_move:
                    try:
                        new_piece.move(new_move)
                        eval_score = minimax(new_board, depth - 1, alpha, beta, False, player_name, start_time, time_limit)
                        
                        if eval_score is None:  # Timeout
                            return None
                        
                        max_eval = max(max_eval, eval_score)
                        alpha = max(alpha, eval_score)
                        
                        if beta <= alpha:
                            break  # Beta cutoff
                    except:
                        continue
        
        return max_eval
    else:
        min_eval = float('inf')
        for piece, move in legal_moves:
            # Make move on a cloned board
            new_board = board.clone()
            
            # Find corresponding piece and move on cloned board
            new_piece = None
            for p in new_board.get_player_pieces(current_player):
                if type(p) == type(piece) and p.position == piece.position:
                    new_piece = p
                    break
            
            if new_piece:
                new_move = None
                for m in new_piece.get_move_options():
                    if hasattr(m, 'position') and m.position == move.position:
                        new_move = m
                        break
                
                if new_move:
                    try:
                        new_piece.move(new_move)
                        eval_score = minimax(new_board, depth - 1, alpha, beta, True, player_name, start_time, time_limit)
                        
                        if eval_score is None:  # Timeout
                            return None
                        
                        min_eval = min(min_eval, eval_score)
                        beta = min(beta, eval_score)
                        
                        if beta <= alpha:
                            break  # Alpha cutoff
                    except:
                        continue
        
        return min_eval

def find_best_move(board, player, time_limit=4.5):
    """
    Find the best move using iterative deepening
    """
    start_time = time.time()
    legal_moves = list_legal_moves_for(board, player)
    
    if not legal_moves:
        return None, None
    
    if len(legal_moves) == 1:
        return legal_moves[0]
    
    best_move = legal_moves[0]  # Fallback
    
    # Iterative deepening
    for depth in range(1, 50):  # Max depth 50 (very unlikely to reach)
        if time.time() - start_time > time_limit:
            break
        
        best_score = float('-inf')
        current_best = None
        alpha = float('-inf')
        beta = float('inf')
        
        # Order moves
        ordered_moves = order_moves(board, legal_moves, player.name)
        
        for piece, move in ordered_moves:
            if time.time() - start_time > time_limit:
                break
            
            # Make move on cloned board
            new_board = board.clone()
            
            # Find corresponding piece and move
            new_piece = None
            for p in new_board.get_player_pieces(player):
                if type(p) == type(piece) and p.position == piece.position:
                    new_piece = p
                    break
            
            if new_piece:
                new_move = None
                for m in new_piece.get_move_options():
                    if hasattr(m, 'position') and m.position == move.position:
                        new_move = m
                        break
                
                if new_move:
                    try:
                        new_piece.move(new_move)
                        score = minimax(new_board, depth - 1, alpha, beta, False, player.name, start_time, time_limit)
                        
                        if score is None:  # Timeout
                            break
                        
                        if score > best_score:
                            best_score = score
                            current_best = (piece, move)
                        
                        alpha = max(alpha, score)
                    except:
                        continue
        
        if current_best:
            best_move = current_best
    
    return best_move

def agent(board, player, var):
    """
    Main agent function that returns the best move
    """
    piece, move_opt = find_best_move(board, player, time_limit=4.5)
    
    if piece is None or move_opt is None:
        # Fallback to random legal move
        legal = list_legal_moves_for(board, player)
        if legal:
            piece, move_opt = random.choice(legal)
    
    return piece, move_opt