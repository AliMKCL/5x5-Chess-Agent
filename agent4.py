import random
import time
from extension.board_utils import list_legal_moves_for
from chessmaker.chess.pieces import King, Queen, Bishop, Knight, Pawn
from extension.piece_right import Right
from extension.board_rules import get_result

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
    [100,  100,  100,  100,  100],
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
    # Find both kings and count nearby allies in a single pass
    player_king = None
    opponent_king = None
    player_pieces = []
    opponent_pieces = []
    
    for piece in all_pieces:
        if piece.name.lower() == 'king':
            if piece.player.name == player_name:
                player_king = piece
            else:
                opponent_king = piece
        elif piece.player.name == player_name:
            player_pieces.append(piece)
        else:
            opponent_pieces.append(piece)
    
    # Calculate king safety for both kings
    if player_king:
        kx, ky = player_king.position.x, player_king.position.y
        nearby_allies = sum(1 for p in player_pieces 
                           if abs(p.position.x - kx) <= 1 and abs(p.position.y - ky) <= 1)
        if nearby_allies >= 2:
            score += 50
        elif nearby_allies == 1:
            score += 20
        else:
            score -= 50
    
    if opponent_king:
        kx, ky = opponent_king.position.x, opponent_king.position.y
        nearby_allies = sum(1 for p in opponent_pieces 
                           if abs(p.position.x - kx) <= 1 and abs(p.position.y - ky) <= 1)
        if nearby_allies >= 2:
            score -= 50
        elif nearby_allies == 1:
            score -= 20
        else:
            score += 50

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
    
    # Create position-to-piece lookup for faster capture evaluation
    pos_to_piece = {piece.position: piece for piece in board.get_pieces()}

    for piece, move in moves:
        score = 0
        piece_name = piece.name.lower()
        is_white = piece.player.name == "white"
        attacker_value = get_piece_value(piece)
        added_capture_bonus = False

        # 1 Checkmate   POTENTIALLY NOT WORKING AS INTENDED??
        if hasattr(move, "checkmate") and move.checkmate:
            score += 100000000

        # 2 Valuable captures
        # Prioritize high-value captures (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)
        if hasattr(move, "captures") and move.captures:
            for capture_pos in move.captures:
                target = pos_to_piece.get(capture_pos)
                if target:
                    victim_value = get_piece_value(target)
                    
                    # MVV-LVA: prioritize capturing valuable pieces with cheap pieces
                    # Winning captures (victim >= attacker) get extra bonus
                    if victim_value >= attacker_value:
                        score += (victim_value * 10) - attacker_value
                    else:
                        # Losing captures still considered but with lower priority
                        score += (victim_value * 10) - attacker_value
                    
                    added_capture_bonus = True

        # 3 Positional improvement (if no good capture bonus)
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

def find_best_move(board, player, max_depth=10, time_limit=30.0):
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
    
    # Global counter for nodes explored (for debugging)
    global nodes_explored
    nodes_explored = 0

    # Iterative deepening loop
    for depth in range(1, max_depth + 1):
        depth_start_time = time.time()
        depth_nodes_before = nodes_explored
        print(f"\n=== Depth {depth} ===")

        # Time check before each new depth
        if time.time() - start_time >= time_limit:
            break

        alpha, beta = float('-inf'), float('inf')
        current_best_move = None
        current_best_score = float('-inf')

        ordered_moves = order_moves(board, legal_moves)

        # Try each move at this depth
        found_checkmate = False
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
                print(f"Testing move at depth {depth}: {piece} to ({move.position.x},{move.position.y})")

                # Switch turn
                new_board.current_player = [p for p in new_board.players if p != player][0]

                
                # Check if this move causes a stalemate
                if is_stalemate(new_board):
                    current_eval = evaluate_board(board, player.name)

                    # If we're winning (eval > -500), skip this stalemate move
                    if current_eval > -500:
                        print(f"  -> Skipping stalemate move (current eval: {current_eval:.0f})")
                        continue
                    else:
                        # If we're losing badly, stalemate is acceptable
                        print(f"  -> Accepting stalemate move (current eval: {current_eval:.0f})")
                

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
                    is_max_turn=False,
                    indent_level=1  # Start with indent level 1 for opponent moves
                )

                # If the current move is better than the previous best at this depth, update new best move.
                if score > current_best_score:
                    current_best_score = score
                    current_best_move = (piece, move)
                    # Debug: Show when we find a very good move (potential checkmate)
                    if score >= 999999:
                        print(f"  *** FOUND FORCED CHECKMATE: {piece} to ({move.position.x},{move.position.y}) with score {score}")
                        found_checkmate = True
                        break  # Stop evaluating other moves - we have a forced win!

                # Update alpha for the minimax search (used in recursive calls)
                # but DON'T prune at root level - we want to evaluate all moves
                alpha = max(alpha, score)

            except Exception:
                continue

        # If we found a valid move at this depth, remember it as the best so far
        if current_best_move:
            best_move = current_best_move
            best_score = current_best_score
        
        # Print statistics for this depth
        depth_nodes = nodes_explored - depth_nodes_before
        depth_time = time.time() - depth_start_time
        print(f"Depth {depth} complete: {depth_nodes} nodes explored in {depth_time:.3f}s (Total: {nodes_explored} nodes)")
        
        # EARLY TERMINATION: If we found a forced checkmate, stop iterative deepening immediately
        # No need to search deeper - we already have a guaranteed winning sequence!
        if found_checkmate or current_best_score >= 999999:
            print(f"*** FORCED CHECKMATE SEQUENCE FOUND AT DEPTH {depth} - Stopping search immediately ***")
            print(f"*** Playing: {best_move[0]} to ({best_move[1].position.x},{best_move[1].position.y}) ***")
            break

        # Stop if time runs out mid-search
        if time.time() - start_time >= time_limit:
            break
    return best_move

# ============================================================================
# MINIMAX + ALPHA-BETA
# ============================================================================

import time

def minimax(board, depth, alpha, beta, player_name, time_limit, start_time, is_max_turn=True, indent_level=0):
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
        indent_level: For debug printing (optional)

    Returns:
        Evaluation score for the position (float)
    """
    from extension.board_rules import get_result
    
    # Increment global node counter
    global nodes_explored
    nodes_explored += 1

    # --- 0 Time check ----------------------------------------------------
    if time.time() - start_time >= time_limit:
        # Timeout: return static evaluation immediately
        return evaluate_board(board, player_name)

    # --- 1 Terminal / base cases ----------------------------------------
    result = get_result(board)

    if result is not None:
        res = result.lower()
        if "checkmate - black loses" in res:
            # Prefer faster checkmates: add bonus for shallower depth
            # If we're at depth 5, checkmate is 5 moves away → bonus = (10-5) = 5
            # If we're at depth 1, checkmate is 1 move away → bonus = (10-1) = 9
            mate_bonus = (10 - depth) * 1000
            
            # Check if we (player_name) won or the opponent won
            if player_name in res or (player_name == "white" and "checkmate - black loses" in res) or (player_name == "black" and "checkmate - white loses" in res):
                return 999999 + mate_bonus  # We win: prefer faster mate
            else:
                return -999999 - mate_bonus  # We lose: delay mate as long as possible
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

        # DEBUG: Print what's being explored (comment out for production)
        indent = "  " * indent_level
        turn_type = "MAX" if is_max_turn else "MIN"
        print(f"{indent}[Depth {depth}, {turn_type}] Testing: {piece.name} to ({move.position.x},{move.position.y})")

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

            # Check for stalemate
            if not list_legal_moves_for(new_board, new_board.current_player) and evaluate_board(new_board, player_name) > 0:
                if is_max_turn:
                    return -50000  # Bad for maximizing player
                else:
                    return 50000   # Good for minimizing player
                


            # --- 7 Recursive minimax call -------------------------------
            value = minimax(
                new_board,
                depth - 1,
                alpha,
                beta,
                player_name,
                time_limit,
                start_time,
                is_max_turn=not is_max_turn,
                indent_level=indent_level + 1
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