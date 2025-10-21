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
    'knight': 330,
    'bishop': 320,
    'right': 500,
    'queen': 900,
    'king': 20000
}


PAWN_TABLE = [
    [0,  0,  0,  0,  0],
    [50, 50, 50, 50, 50],
    [15, 10, 20, 10, 10],
    [-5,  5, -5, 5,  0],
    [0,  0,  0,  0,  0]
]

KNIGHT_TABLE = [
    [-10,  -5,  -5,  -5,  -10],
    [-5, 0, 0, 0, -5],
    [-5, 0, 0, 0, -5],
    [-5,  0, 0, 0,  -5],
    [-10,  -5,  -5,  -5,  -10]
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
    elif piece.name.lower() == 'bishop':
        return BISHOP_TABLE[y][x]
    elif piece.name.lower() == 'right':
        return RIGHT_TABLE[y][x]
    elif piece.name.lower() == 'queen':
        return QUEEN_TABLE[y][x]
    return 0

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

# Estimates how good or bad a board position is for the given player.
def evaluate_board(board, player_name):
    """
    Evaluates the board state based on:
      1) Material balance
      2) Positional values from piece-square tables
      3) King safety (bonus if ≥ 2 friendly pieces within 1-cell radius)
    Returns:
        Positive score → good for 'player_name'
        Negative score → good for opponent
    """
    score = 0

    # Get lists of all pieces and players
    all_pieces = board.get_pieces()
    player = next(p for p in board.players if p.name == player_name)
    opponent = next(p for p in board.players if p.name != player_name)

    # --- 1. MATERIAL + POSITIONAL VALUE -----------------------------------
    for piece in all_pieces:
        base_value = get_piece_value(piece)
        is_white = piece.player.name == "white"
        pos_value = get_positional_value(piece, is_white)

        # Combine material and positional value
        total_piece_value = base_value + pos_value

        if piece.player.name == player_name:
            score += total_piece_value
        else:
            score -= total_piece_value

    # --- 2. KING SAFETY ---------------------------------------------------
    def king_safety_bonus(king_piece):
        """Return bonus if at least two friendly pieces near the king."""
        kx, ky = king_piece.position.x, king_piece.position.y
        friendly_pieces = [
            p for p in all_pieces
            if p.player == king_piece.player and p != king_piece
        ]

        # Count how many friendly pieces are within 1 square (Chebyshev distance ≤ 1)
        nearby_allies = 0
        for p in friendly_pieces:
            px, py = p.position.x, p.position.y
            if abs(px - kx) <= 1 and abs(py - ky) <= 1:
                nearby_allies += 1

        # Return a safety bonus if at least two are adjacent
        if nearby_allies >= 2:
            return 50   # Safe king (well protected)
        elif nearby_allies == 1:
            return 20   # Partially protected
        else:
            return -50  # Exposed king (penalty)

    # Find both kings and adjust score accordingly
    for piece in all_pieces:
        if piece.name.lower() == 'king':
            bonus = king_safety_bonus(piece)
            if piece.player.name == player_name:
                score += bonus
            else:
                score -= bonus

    return score

# ============================================================================
# MOVE ORDERING
# ============================================================================

def order_moves(board, moves):
    """
    Orders moves by tactical and positional importance.

    Priority:
      1. Checkmate moves
      2. Valuable or safe captures (victim >= attacker OR target is undefended)
      3. Positional improvement (piece-square tables)
    """
    scored_moves = []

    # Checks if a square is defended before capturing it.
    def is_square_defended(board, square, defending_player):
        """Check if any of defending_player's pieces can move to 'square'."""
        defending_moves = list_legal_moves_for(board, defending_player)
        for dpiece, dmove in defending_moves:
            if hasattr(dmove, "position") and dmove.position == square:
                return True
        return False

    for piece, move in moves:
        score = 0
        piece_name = piece.name.lower()
        is_white = piece.player.name == "white"
        attacker_value = get_piece_value(piece)
        added_capture_bonus = False

        # 1️⃣ Checkmate
        if hasattr(move, "checkmate") and move.checkmate:
            score += 100000000

        # 2️⃣ Valuable or safe captures
        if hasattr(move, "captures") and move.captures:
            for capture_pos in move.captures:
                for target in board.get_pieces():
                    if target.position == capture_pos:
                        victim_value = get_piece_value(target)
                        opponent = next(p for p in board.players if p != piece.player)

                        # Check if capture square is defended
                        defended = is_square_defended(board, capture_pos, opponent)

                        # Reward captures that are either favorable or undefended
                        if victim_value >= attacker_value or not defended:
                            score += (victim_value * 10) - attacker_value
                            added_capture_bonus = True
                        break

        # 3️⃣ Positional improvement (if no good capture bonus)
        if not added_capture_bonus:
            old_pos_value = get_positional_value(piece, is_white)

            new_x, new_y = move.position.x, move.position.y
            if not is_white:
                new_y = 4 - new_y

            piece_tables = {
                'pawn': PAWN_TABLE,
                'knight': KNIGHT_TABLE,
                'bishop': BISHOP_TABLE,
                'right': RIGHT_TABLE,
                'queen': QUEEN_TABLE,
                'king': KING_TABLE
            }
            new_pos_value = piece_tables.get(piece_name, [[0]*5]*5)[new_y][new_x]

            score += (new_pos_value - old_pos_value)

        scored_moves.append((score, piece, move))

    scored_moves.sort(reverse=True, key=lambda x: x[0])
    return [(p, m) for _, p, m in scored_moves]


# ============================================================================
# SEARCH
# ============================================================================

import time
import random

def find_best_move(board, player, max_depth=10, time_limit=10.0):
    """
    Iterative deepening search using alpha-beta minimax.

    Args:
        board: Current game board object
        player: The player whose move we’re finding
        max_depth: Maximum depth to search (default = 10)
        time_limit: Max search time in seconds (default = 5.0)

    Returns:
        (piece, move) tuple representing the best move found
    """
    start_time = time.time()
    legal_moves = list_legal_moves_for(board, player)

    if not legal_moves:
        return None, None  # no legal moves
    if len(legal_moves) == 1:
        return legal_moves[0]  # only one option, no search needed

    best_move = random.choice(legal_moves)  # fallback move
    best_score = float('-inf')

    # Iterative deepening loop
    for depth in range(1, max_depth + 1):
        # Time check before each new depth
        if time.time() - start_time >= time_limit:
            break

        alpha, beta = float('-inf'), float('inf')
        current_best_move = None
        current_best_score = float('-inf')

        ordered_moves = order_moves(board, legal_moves)

        # Try each move at this depth
        for piece, move in ordered_moves:
            if time.time() - start_time >= time_limit:
                break  # stop if time runs out

            # Clone board and apply move
            new_board = board.clone()
            try:
                new_piece = next((p for p in new_board.get_player_pieces(player)
                                  if type(p) == type(piece) and p.position == piece.position), None)
                if not new_piece:
                    continue

                new_move = next((m for m in new_piece.get_move_options()
                                 if hasattr(m, "position") and m.position == move.position), None)
                if not new_move:
                    continue

                new_piece.move(new_move)
                # Switch turn
                new_board.current_player = [p for p in new_board.players if p != player][0]

                # Run minimax for OPPONENT's reply
                # This runs after our hypothetical move (that we are checking) so it works.
                score = minimax(
                    new_board,
                    depth - 1,
                    alpha,
                    beta,
                    player_name=player.name,
                    time_limit = time_limit,
                    start_time = start_time,
                    is_max_turn=False
                )

                # If the current move is better than the previous best at this depth, update new best move.
                if score > current_best_score:
                    current_best_score = score
                    current_best_move = (piece, move)

                alpha = max(alpha, score)
                if beta <= alpha:
                    break  # prune

            except Exception:
                continue

        # If we found a valid move at this depth, remember it as the best so far
        if current_best_move:
            best_move = current_best_move
            best_score = current_best_score

        # Stop if time runs out mid-search
        if time.time() - start_time >= time_limit:
            break

    return best_move

# ============================================================================
# MINIMAX + ALPHA-BETA
# ============================================================================

import time

def minimax(board, depth, alpha, beta, player_name, time_limit, start_time, is_max_turn=True):
    """
    Minimax search with alpha-beta pruning, time awareness, and custom early pruning rule.

    Args:
        board: Current board state
        depth: Remaining depth to search
        alpha, beta: Alpha-beta pruning bounds
        player_name: The agent's name (to evaluate from its perspective)
        time_limit: Max allowed time (seconds)
        start_time: When search started
        is_max_turn: True if this turn belongs to the agent

    Returns:
        Evaluation score for the position (float)
    """
    from extension.board_rules import get_result

    # --- 0 Time check ----------------------------------------------------
    if time.time() - start_time >= time_limit:
        # Timeout: return static evaluation immediately
        return evaluate_board(board, player_name)

    # --- 1 Terminal / base cases ----------------------------------------
    result = get_result(board)
    if result is not None:
        res = result.lower()
        if "win" in res:
            return 999999 if player_name in res else -999999
        elif "draw" in res:
            return 0

    if depth == 0:
        return evaluate_board(board, player_name)

    # --- 2 Get legal moves ----------------------------------------------
    current_player = board.current_player
    legal_moves = list_legal_moves_for(board, current_player)
    if not legal_moves:
        return evaluate_board(board, player_name)

    # --- 3 Initialize best value ----------------------------------------
    best_value = float('-inf') if is_max_turn else float('inf')

    # --- 4 Order moves for better pruning -------------------------------
    ordered_moves = order_moves(board, legal_moves)

    # --- 5 Explore moves ------------------------------------------------
    for piece, move in ordered_moves:
        # Check time inside loop too
        if time.time() - start_time >= time_limit:
            break

        new_board = board.clone()
        try:
            # Locate piece & move equivalents on cloned board
            new_piece = next((p for p in new_board.get_player_pieces(current_player)
                              if type(p) == type(piece) and p.position == piece.position), None)
            if not new_piece:
                continue
            new_move = next((m for m in new_piece.get_move_options()
                             if hasattr(m, "position") and m.position == move.position), None)
            if not new_move:
                continue

            new_piece.move(new_move)
            # Switch turn
            new_board.current_player = [p for p in new_board.players if p != current_player][0]

            # --- 7️⃣ Recursive minimax call -------------------------------
            value = minimax(
                new_board,
                depth - 1,
                alpha,
                beta,
                player_name,
                time_limit,
                start_time,
                is_max_turn=not is_max_turn
            )

            # --- 8️⃣ Alpha-beta updates -----------------------------------
            if is_max_turn:
                best_value = max(best_value, value)
                alpha = max(alpha, value)
            else:
                best_value = min(best_value, value)
                beta = min(beta, value)

            if beta <= alpha:
                break  # Prune rest of branch

        except Exception:
            continue

    return best_value



















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