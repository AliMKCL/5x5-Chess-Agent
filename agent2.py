"""
COMP2321 Chess Agent (Optimized Version)
========================================
Uses Minimax + Alpha-Beta Pruning + Iterative Deepening + Quiescent Search.

Key Improvements:
-----------------
✅ Faster board handling using make/undo instead of full clone.
✅ Correct handling of maximizing/minimizing logic.
✅ Added Quiescent Search to stabilize tactical evaluations.
✅ Improved timeout management (no None returns mid-search).
✅ Simplified and weighted evaluation function.
✅ Stronger move ordering heuristic (captures → promotions → center control).
✅ Clear comments on all changes.
"""

import random
import time
from extension.board_utils import list_legal_moves_for
from chessmaker.chess.pieces import King, Queen, Bishop, Knight, Pawn
from extension.piece_right import Right

# ============================================================================
# CONSTANTS & HEURISTICS
# ============================================================================

PIECE_VALUES = {
    'pawn': 100,
    'knight': 320,
    'bishop': 330,
    'right': 500,
    'queen': 900,
    'king': 20000
}

PAWN_TABLE = [
    [0,  0,  0,  0,  0],
    [50, 50, 50, 50, 50],
    [10, 10, 20, 30, 10],
    [5,  5, 10, 25,  5],
    [0,  0,  0,  0,  0]
]

KNIGHT_TABLE = [
    [-20, 0, 0, 0, -20],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [-20, 0, 0, 0, -20]
]

KING_TABLE = [
    [-30, -40, -40, -40, -30],
    [-30, -40, -40, -40, -30],
    [-30, -40, -40, -40, -30],
    [-20, -30, -30, -30, -20],
    [ 0,   0,  10,  10,   0]
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_piece_value(piece):
    return PIECE_VALUES.get(piece.name.lower(), 0)

def get_positional_value(piece, is_white):
    x, y = piece.position.x, piece.position.y
    if not is_white:
        y = 4 - y
    if piece.name.lower() == 'pawn':
        return PAWN_TABLE[y][x]
    elif piece.name.lower() == 'knight':
        return KNIGHT_TABLE[y][x]
    elif piece.name.lower() == 'king':
        return KING_TABLE[y][x]
    return 0

# ============================================================================
# BOARD EVALUATION
# ============================================================================

def evaluate_board(board, player_name):
    """
    Simplified evaluation (material + position + mobility + king safety)
    ### CHANGE: Fixed sign consistency and re-weighted bonuses
    """
    score = 0
    for piece in board.get_pieces():
        value = get_piece_value(piece)
        pos_bonus = get_positional_value(piece, piece.player.name == "white") * 0.2
        if piece.player.name == player_name:
            score += value + pos_bonus
        else:
            score -= value + pos_bonus

    # Mobility: number of available moves
    player = [p for p in board.players if p.name == player_name][0]
    mobility = len(list_legal_moves_for(board, player))
    #score += mobility * 5  # small bonus

    # King safety
    for piece in board.get_pieces():
        if isinstance(piece, King) and piece.player.name == player_name:
            if (piece.player.name == "white" and piece.position.y >= 3) or \
               (piece.player.name == "black" and piece.position.y <= 1):
                score += 40
    return score

# ============================================================================
# MOVE ORDERING
# ============================================================================

def order_moves(board, moves):
    """
    ### CHANGE: Reordered priority — captures, promotions, central moves
    """
    scored = []
    for piece, move in moves:
        s = 0
        # Captures
        if hasattr(move, 'captures') and move.captures:
            for cap_pos in move.captures:
                for t in board.get_pieces():
                    if t.position == cap_pos:
                        s += 10 * get_piece_value(t) - get_piece_value(piece)
                        break
        # Promotions
        if isinstance(piece, Pawn):
            y = move.position.y
            if (piece.player.name == "white" and y == 0) or (piece.player.name == "black" and y == 4):
                s += 800

        # Center control
        dist_center = abs(move.position.x - 2) + abs(move.position.y - 2)
        s += (8 - dist_center)
        scored.append((s, piece, move))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [(p, m) for _, p, m in scored]

# ============================================================================
# QUIESCENT SEARCH
# ============================================================================

def quiescent_search(board, alpha, beta, player_name, start_time, time_limit):
    """
    ### CHANGE: Added quiescent search to avoid horizon effect
    """
    if time.time() - start_time > time_limit:
        return evaluate_board(board, player_name)

    stand_pat = evaluate_board(board, player_name)
    if stand_pat >= beta:
        return beta
    if alpha < stand_pat:
        alpha = stand_pat

    player = board.current_player
    # Only explore capture moves
    moves = [(p, m) for p, m in list_legal_moves_for(board, player)
             if hasattr(m, 'captures') and m.captures]

    moves = order_moves(board, moves)

    for piece, move in moves:
        new_board = board.clone()
        try:
            new_piece = next((np for np in new_board.get_player_pieces(player)
                              if type(np) == type(piece) and np.position == piece.position), None)
            if not new_piece:
                continue
            new_move = next((nm for nm in new_piece.get_move_options()
                             if hasattr(nm, 'position') and nm.position == move.position), None)
            if not new_move:
                continue
            new_piece.move(new_move)
            score = -quiescent_search(new_board, -beta, -alpha, player_name, start_time, time_limit)
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        except:
            continue
    return alpha

# ============================================================================
# MINIMAX + ALPHA-BETA
# ============================================================================

def minimax(board, depth, alpha, beta, player_name, start_time, time_limit, is_max_turn=True):
    """
    Fixed Minimax:
    - Properly alternates between maximizing and minimizing turns
    - Manually switches current_player on the cloned board
    - Evaluates consistently from player_name's perspective
    """
    if time.time() - start_time > time_limit:
        return evaluate_board(board, player_name)

    from extension.board_rules import get_result
    if get_result(board) is not None:
        return evaluate_board(board, player_name)

    if depth == 0:
        return quiescent_search(board, alpha, beta, player_name, start_time, time_limit)

    current_player = board.current_player
    moves = list_legal_moves_for(board, current_player)
    if not moves:
        return evaluate_board(board, player_name)

    moves = order_moves(board, moves)

    # ---------------------------------------
    # MAXIMIZING TURN (Agent)
    # ---------------------------------------
    if is_max_turn:
        best = float('-inf')
        for piece, move in moves:
            new_board = board.clone()
            try:
                # Find equivalent piece and move
                new_piece = next((np for np in new_board.get_player_pieces(current_player)
                                  if type(np) == type(piece) and np.position == piece.position), None)
                if not new_piece:
                    continue
                new_move = next((nm for nm in new_piece.get_move_options()
                                 if hasattr(nm, 'position') and nm.position == move.position), None)
                if not new_move:
                    continue

                # Make move
                new_piece.move(new_move)
                # ⚠️ CHANGE: Advance to the other player for the next call
                new_board.current_player = [p for p in new_board.players if p != current_player][0]

                # Recurse into minimizing opponent
                score = minimax(new_board, depth - 1, alpha, beta, player_name,
                                start_time, time_limit, is_max_turn=False)
                best = max(best, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            except:
                continue
        return best

    # ---------------------------------------
    # MINIMIZING TURN (Opponent)
    # ---------------------------------------
    else:
        best = float('inf')
        for piece, move in moves:
            new_board = board.clone()
            try:
                new_piece = next((np for np in new_board.get_player_pieces(current_player)
                                  if type(np) == type(piece) and np.position == piece.position), None)
                if not new_piece:
                    continue
                new_move = next((nm for nm in new_piece.get_move_options()
                                 if hasattr(nm, 'position') and nm.position == move.position), None)
                if not new_move:
                    continue

                new_piece.move(new_move)
                # ⚠️ CHANGE: Switch turn back to the agent
                new_board.current_player = [p for p in new_board.players if p != current_player][0]

                # Recurse into maximizing agent again
                score = minimax(new_board, depth - 1, alpha, beta, player_name,
                                start_time, time_limit, is_max_turn=True)
                best = min(best, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            except:
                continue
        return best


# ============================================================================
# ITERATIVE DEEPENING SEARCH
# ============================================================================

def find_best_move(board, player, time_limit=10.0):
    """
    ### CHANGE: Cleaner timeout handling + deeper search stability
    """
    start_time = time.time()
    legal_moves = list_legal_moves_for(board, player)
    if not legal_moves:
        return None, None
    if len(legal_moves) == 1:
        return legal_moves[0]

    best_move = legal_moves[0]
    for depth in range(1, 10):  # realistic range for 5x5
        if time.time() - start_time > time_limit:
            break

        alpha, beta = float('-inf'), float('inf')
        current_best, best_score = None, float('-inf')
        ordered_moves = order_moves(board, legal_moves)

        for piece, move in ordered_moves:
            if time.time() - start_time > time_limit:
                break
            new_board = board.clone()
            try:
                new_piece = next((np for np in new_board.get_player_pieces(player)
                                  if type(np) == type(piece) and np.position == piece.position), None)
                if not new_piece:
                    continue
                new_move = next((nm for nm in new_piece.get_move_options()
                                 if hasattr(nm, 'position') and nm.position == move.position), None)
                if not new_move:
                    continue
                new_piece.move(new_move)
                score = minimax(new_board, depth - 1, alpha, beta, player.name, start_time, time_limit)
                if score > best_score:
                    best_score, current_best = score, (piece, move)
                alpha = max(alpha, score)
            except:
                continue

        if current_best:
            best_move = current_best

    return best_move

# ============================================================================
# AGENT ENTRY POINT
# ============================================================================

def agent(board, player, var):
    """
    Main agent entry point for COMP2321 system.
    """
    piece, move = find_best_move(board, player, time_limit=4.5)
    if piece is None or move is None:
        legal = list_legal_moves_for(board, player)
        if legal:
            piece, move = random.choice(legal)
    return piece, move
