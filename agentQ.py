import random
import time
from extension.board_utils import list_legal_moves_for
from chessmaker.chess.pieces import King, Queen, Bishop, Knight, Pawn
from extension.piece_right import Right
from extension.board_rules import get_result
from helpers import *

# ============================================================================
# CONSTANTS & HEURISTICS
# ============================================================================

# Global file handle for logging
LOG_FILE = None
MOVE_COUNTER = 0

# Quiescence search configuration
QUIESCENCE_ENABLED = True  # Toggle quiescence search on/off
MAX_QUIESCENCE_DEPTH = 7  # Maximum quiescence search depth

def init_log_file():
    """Initialize the log file for this game session."""
    global LOG_FILE
    LOG_FILE = open("game_log.txt", "a")
    LOG_FILE.write("=== GAME LOG (with Quiescence Search) ===\n\n")

def close_log_file():
    """Close the log file."""
    global LOG_FILE
    if LOG_FILE:
        LOG_FILE.close()
        LOG_FILE = None

def log_message(message):
    """Write a message to the log file."""
    global LOG_FILE
    if LOG_FILE:
        LOG_FILE.write(message + "\n")
        LOG_FILE.flush()

def log_board_state(board, title=""):
    """Log the current board state to file."""
    global LOG_FILE
    if not LOG_FILE:
        return

    if title:
        LOG_FILE.write(f"\n{title}\n")

    # Write board representation
    LOG_FILE.write("  0 1 2 3 4\n")
    for y in range(5):
        LOG_FILE.write(f"{y} ")
        for x in range(5):
            piece = None
            for p in board.get_pieces():
                if p.position.x == x and p.position.y == y:
                    piece = p
                    break
            if piece:
                # Map piece names to their correct symbols
                piece_name_lower = piece.name.lower()
                if piece_name_lower == "knight":
                    symbol = 'n'
                elif piece_name_lower == "king":
                    symbol = 'k'
                elif piece_name_lower == "queen":
                    symbol = 'q'
                elif piece_name_lower == "bishop":
                    symbol = 'b'
                elif piece_name_lower == "right":
                    symbol = 'r'
                elif piece_name_lower == "pawn":
                    symbol = 'p'
                else:
                    symbol = piece.name[0].lower()

                # Uppercase for white pieces
                if piece.player.name == "white":
                    symbol = symbol.upper()
                LOG_FILE.write(f"{symbol} ")
            else:
                LOG_FILE.write(". ")
        LOG_FILE.write("\n")
    LOG_FILE.write("\n")
    LOG_FILE.flush()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_move_cache(board):
    """
    Build a cache of move options for all pieces on the board.

    This is CRITICAL-2 optimization: Instead of calling piece.get_move_options()
    multiple times for the same piece in the same board state, we cache all moves
    once and reuse them.

    Time Complexity:
    - WITHOUT cache: O(n²) - n pieces × n calls per piece = n²
    - WITH cache: O(n) - one call per piece

    Returns:
        dict: {(piece.position.x, piece.position.y, piece.name, piece.player.name): [moves]}
    """
    cache = {}
    for piece in board.get_pieces():
        # Create a unique key for this piece (position + type + player)
        key = (piece.position.x, piece.position.y, piece.name, piece.player.name)
        cache[key] = piece.get_move_options()
    return cache

def create_position_map(board):
    """
    Create a position-to-piece lookup table for O(1) piece lookups by coordinates.

    This is CRITICAL-3 optimization: Instead of scanning board.get_pieces() linearly
    (O(n) per lookup), we build a dictionary mapping (x, y) -> piece (O(1) lookups).

    Time Complexity:
    - Building the map: O(n) where n = number of pieces
    - Lookups: O(1) instead of O(n)

    Args:
        board: The board to build the position map for

    Returns:
        dict: {(x, y): piece} mapping coordinates to piece objects
    """
    pos_map = {}
    for piece in board.get_pieces():
        pos_map[(piece.position.x, piece.position.y)] = piece
    return pos_map

def get_cached_moves(piece, cache):
    """
    Retrieve cached move options for a piece.

    Args:
        piece: The piece to get moves for
        cache: The move cache dictionary

    Returns:
        list: Cached move options, or fresh moves if not cached
    """
    key = (piece.position.x, piece.position.y, piece.name, piece.player.name)
    if key in cache:
        return cache[key]
    # Fallback: if somehow not in cache, compute fresh
    print("Warning: Piece moves not found in cache, computing fresh.")
    return piece.get_move_options()


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

# Estimates how good or bad a board position is for the given player.
def evaluate_board(board, player_name, pos_map=None):
    """
    Evaluates the board state based on:
      1) Material balance
      2) Positional values from piece-square tables
      3) King safety (bonus if ≥ 2 friendly pieces within 1-cell radius)

    Args:
        board: Current board state
        player_name: Name of the player to evaluate for
        pos_map: Optional position-to-piece lookup dict for O(1) piece lookups

    Returns:
        Positive score → good for 'player_name'
        Negative score → good for opponent
    """
    score = 0

    # Get lists of all pieces and players
    all_pieces = board.get_pieces()
    player = next(p for p in board.players if p.name == player_name)
    opponent = next(p for p in board.players if p.name != player_name)

    # --- SINGLE-PASS EVALUATION -------------------------------------------
    # Collect material, positional values, and piece positions in one iteration
    player_king = None
    opponent_king = None
    player_pieces = []
    opponent_pieces = []

    for piece in all_pieces:
        piece_name_lower = piece.name.lower()
        is_player_piece = piece.player.name == player_name

        # 1. Material + Positional Value
        base_value = get_piece_value(piece)
        is_white = piece.player.name == "white"
        pos_value = get_positional_value(piece, is_white, board)
        total_piece_value = base_value + pos_value

        if is_player_piece:
            score += total_piece_value
        else:
            score -= total_piece_value

        # 2. Collect pieces for king safety (done in same loop)
        if piece_name_lower == 'king':
            if is_player_piece:
                player_king = piece
            else:
                opponent_king = piece
        elif is_player_piece:
            player_pieces.append(piece)
        else:
            opponent_pieces.append(piece)

    # --- 3. KING SAFETY ---------------------------------------------------
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

def order_moves(board, moves, move_cache=None, pos_map=None):
    """
    Orders moves by tactical and positional importance.

    Priority:
      1. Checkmate moves
      2. Valuable or safe captures (victim >= attacker OR target is undefended)
      3. Positional improvement (piece-square tables)

    Args:
        board: Current board state
        moves: List of (piece, move) tuples to order
        move_cache: Optional cache of piece move options to avoid redundant calculations
        pos_map: Optional position-to-piece lookup dict for O(1) piece lookups
    """
    scored_moves = []

    # Build move cache if not provided (for backwards compatibility)
    if move_cache is None:
        move_cache = build_move_cache(board)

    # CRITICAL-3 OPTIMIZATION: Create position-to-piece lookup for O(1) access
    if pos_map is None:
        pos_map = create_position_map(board)

    # Cache piece counts per player for endgame king table lookup (computed once instead of per-move)
    all_pieces = board.get_pieces()
    piece_counts = {}
    for p in all_pieces:
        if p.player not in piece_counts:
            piece_counts[p.player] = 0
        piece_counts[p.player] += 1

    for piece, move in moves:
        score = 0
        piece_name = piece.name.lower()
        is_white = piece.player.name == "white"
        attacker_value = get_piece_value(piece)
        added_capture_bonus = False

        # ===== 1 Checkmate (if the move object has this attribute set) ===============
        if hasattr(move, "checkmate") and move.checkmate:
            score += 100000000
            print(f"Move ordering: Found checkmate move {piece.name} to ({move.position.x},{move.position.y})")
            log_message(f"Move ordering: Checkmate move detected - {piece.name} to ({move.position.x},{move.position.y})")

        #=====  2 Valuable captures ===============================================
        # Prioritize high-value captures (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)

        if hasattr(move, "captures") and move.captures:
            for capture_pos in move.captures:
                # CRITICAL-3 OPTIMIZATION: O(1) lookup instead of O(n) scan
                target = pos_map.get((capture_pos.x, capture_pos.y))
                if target:
                    victim_value = get_piece_value(target)

                    # Get opponent player for exchange evaluation
                    opponent = next(p for p in board.players if p != piece.player)
                    num_diff, val_diff = attacker_defender_ratio(board, move.position, opponent, piece.player, move_cache, pos_map)

                    # Base MVV-LVA score: prefer low-value attackers capturing high-value victims
                    # Using (victim * 10) ensures victim value is prioritized
                    base_mvv_lva = (victim_value * 10) - attacker_value

                    # Case 1: More or equal defenders than attackers
                    if not val_diff or val_diff < 0:
                        score += base_mvv_lva    # Check if weakest attacker < victim_value
                        #log_message(f"Equal/losing numbers but potentially favorable: {piece.name} captures {target.name}, num_diff={num_diff}, score = {score} move {piece.name} to ({move.position.x},{move.position.y})")

                    # Case 2: More attackers than defenders, and positive trade
                    elif val_diff > 0:
                        score += base_mvv_lva + 1000    # Capture with the weakest attacker and prefer the capture
                        #log_message(f"Winning exchange: {piece.name} captures {target.name}, net={val_diff}, score={score}, move {piece.name} to ({move.position.x},{move.position.y})")



        # ===== 3 Positional improvement (if no good capture bonus)==================
        old_pos_value = get_positional_value(piece, is_white, board)

        new_x, new_y = move.position.x, move.position.y
        if not is_white:
            new_y = 4 - new_y

        # Use cached piece count instead of recalculating for every move
        player_piece_count = piece_counts.get(piece.player, 0)
        piece_tables = {
            'pawn': PAWN_TABLE,
            'knight': KNIGHT_TABLE,
            'bishop': BISHOP_TABLE,
            'right': RIGHT_TABLE,
            'queen': QUEEN_TABLE,
            'king': KING_TABLE_ENDGAME if player_piece_count <= 4 else KING_TABLE
        }
        new_pos_value = piece_tables.get(piece_name, [[0]*5]*5)[new_y][new_x]

        score += (new_pos_value - old_pos_value)

        scored_moves.append((score, piece, move))

    scored_moves.sort(reverse=True, key=lambda x: x[0])
    return [(p, m) for _, p, m in scored_moves]


# ============================================================================
# QUIESCENCE SEARCH (PHASE 1)
# ============================================================================

def quiescence_search(board, alpha, beta, player_name, time_limit, start_time, is_max_turn=True, q_depth=0):
    """
    Quiescence search: extend search for tactical moves only.
    Prevents horizon effect by resolving capture sequences to quiet positions.
    # ADD CHECKS as well

    Uses STANDARD MINIMAX (not negamax) for clarity and correctness.
    - is_max_turn=True: Maximizing player (agent) tries to maximize score
    - is_max_turn=False: Minimizing player (opponent) tries to minimize score
    - Evaluation always from player_name's perspective (positive = good for player_name)

    Based on game_log.txt analysis:
    - Quiet positions: scores typically in range [-100, +100]
    - Captures can swing: [-500, +500]
    - We search until no more captures improve the position

    Args:
        board: Current board state
        alpha, beta: Alpha-beta bounds
        player_name: The agent's perspective (fixed throughout search tree)
        time_limit: Max time allowed
        start_time: When search started
        is_max_turn: True if current turn belongs to the agent (maximizing)
        q_depth: Current quiescence depth (for limiting)

    Returns:
        Evaluation score (int) from player_name's perspective
    """
    global nodes_explored
    nodes_explored += 1

    # Check timeout
    if time.time() - start_time >= time_limit:
        eval_score = evaluate_board(board, player_name)
        indent = "  " * (q_depth + 2)
        log_message(f"{indent}[Q-depth {q_depth}] TIMEOUT - returning static eval={eval_score}")
        return eval_score

    # Limit quiescence depth to prevent infinite loops
    if q_depth >= MAX_QUIESCENCE_DEPTH:
        eval_score = evaluate_board(board, player_name)
        indent = "  " * (q_depth + 2)
        log_message(f"{indent}[Q-depth {q_depth}] MAX Q-DEPTH REACHED - returning static eval={eval_score}")
        return eval_score

    indent = "  " * (q_depth + 2)

    # Check for terminal game states
    result = get_result(board)
    if result is not None:
        res = result.lower()
        if "checkmate" in res:
            # Return high/low scores based on who won
            if player_name in res or (player_name == "white" and "black loses" in res) or (player_name == "black" and "white loses" in res):
                log_message(f"{indent}[Q-depth {q_depth}] CHECKMATE (we win) - returning +999999")
                return 999999  # We win
            else:
                log_message(f"{indent}[Q-depth {q_depth}] CHECKMATE (we lose) - returning -999999")
                return -999999  # We lose
        elif "draw" in res or "stalemate" in res:
            log_message(f"{indent}[Q-depth {q_depth}] DRAW/STALEMATE - returning 0")
            return 0

    # Stand-pat score: evaluate current position without any moves
    # This is the "quiet" baseline - if we don't make a capture, this is what we get
    stand_pat = evaluate_board(board, player_name)
    turn_type = "MAX" if is_max_turn else "MIN"
    log_message(f"{indent}[Q-depth {q_depth}, {turn_type}] Stand-pat evaluation = {stand_pat} (alpha={alpha}, beta={beta})")

    # STANDARD MINIMAX LOGIC:
    # - MAX turn: if stand_pat >= beta, this position is too good, MIN player won't allow it (beta cutoff)
    #            if stand_pat > alpha, update our guaranteed minimum (alpha)
    # - MIN turn: if stand_pat <= alpha, this position is too bad, MAX player won't allow it (alpha cutoff)
    #            if stand_pat < beta, update opponent's guaranteed maximum (beta)

    if is_max_turn:
        # Maximizing player
        if stand_pat >= beta:
            log_message(f"{indent}[Q-depth {q_depth}] BETA CUTOFF (stand-pat {stand_pat} >= beta {beta}) - pruning")
            return beta
        if stand_pat > alpha:
            original_alpha = alpha
            alpha = stand_pat
            log_message(f"{indent}[Q-depth {q_depth}] Stand-pat improves alpha: {original_alpha} -> {alpha}")
    else:
        # Minimizing player
        if stand_pat <= alpha:
            log_message(f"{indent}[Q-depth {q_depth}] ALPHA CUTOFF (stand-pat {stand_pat} <= alpha {alpha}) - pruning")
            return alpha
        if stand_pat < beta:
            original_beta = beta
            beta = stand_pat
            log_message(f"{indent}[Q-depth {q_depth}] Stand-pat improves beta: {original_beta} -> {beta}")

    # Get all legal moves
    current_player = board.current_player
    all_moves = list_legal_moves_for(board, current_player)

    if not all_moves:
        log_message(f"{indent}[Q-depth {q_depth}] No legal moves available - returning stand-pat={stand_pat}")
        return stand_pat

    # Filter for TACTICAL moves only (captures)
    # In quiescence, we only consider forcing moves that might change evaluation
    tactical_moves = []
    for piece, move in all_moves:
        is_capture = hasattr(move, "captures") and move.captures
        if is_capture:
            tactical_moves.append((piece, move))

    # If no tactical moves, position is "quiet" - return stand-pat
    if not tactical_moves:
        log_message(f"{indent}[Q-depth {q_depth}] No captures available - QUIET POSITION (stand-pat={stand_pat})")
        print(f"{'  ' * (q_depth + 1)}[Q-depth {q_depth}] Quiet position, eval={stand_pat}")
        return stand_pat

    log_message(f"{indent}[Q-depth {q_depth}] Found {len(tactical_moves)} captures to analyze from {len(all_moves)} total moves")
    print(f"{'  ' * (q_depth + 1)}[Q-depth {q_depth}] Analyzing {len(tactical_moves)} captures...")

    # Order tactical moves by MVV-LVA for better pruning
    move_cache = build_move_cache(board)
    pos_map = create_position_map(board)
    ordered_tactical = order_moves(board, tactical_moves, move_cache, pos_map)

    # Initialize best score based on who's turn it is (standard minimax)
    # MAX player starts at -infinity, MIN player starts at +infinity
    best_q_score = stand_pat  # Start with stand-pat as baseline
    moves_tried = 0

    # Search only capture moves
    for piece, move in ordered_tactical:
        if time.time() - start_time >= time_limit:
            log_message(f"{indent}[Q-depth {q_depth}] Timeout during capture search - returning best so far: {best_q_score}")
            break

        moves_tried += 1
        new_board = board.clone()
        try:
            # Apply move (same as minimax)
            new_piece = next((p for p in new_board.get_player_pieces(current_player)
                             if type(p) == type(piece) and p.position == piece.position), None)
            if not new_piece:
                continue

            cached_moves = get_cached_moves(new_piece, move_cache)
            new_move = next((m for m in cached_moves
                            if hasattr(m, "position") and m.position == move.position), None)
            if not new_move:
                continue

            # Get capture target info for logging
            capture_target = None
            if hasattr(new_move, "captures") and new_move.captures:
                for cap_pos in new_move.captures:
                    target = pos_map.get((cap_pos.x, cap_pos.y))
                    if target:
                        capture_target = target.name
                        break

            new_piece.move(new_move)
            new_board.current_player = [p for p in new_board.players if p != current_player][0]

            log_message(f"{indent}[Q-depth {q_depth}] Try #{moves_tried}: {piece.name}x{capture_target or '?'} at ({move.position.x},{move.position.y})")

            # STANDARD MINIMAX: Recursive call with flipped is_max_turn
            # NO negation, NO alpha/beta flip - just like regular minimax
            score = quiescence_search(new_board, alpha, beta, player_name, time_limit, start_time, not is_max_turn, q_depth + 1)

            log_message(f"{indent}[Q-depth {q_depth}]   -> {piece.name}x{capture_target or '?'} returned score={score} (alpha={alpha}, beta={beta})")

            # STANDARD MINIMAX ALPHA-BETA UPDATE (same as main minimax function)
            if is_max_turn:
                # Maximizing player
                best_q_score = max(best_q_score, score)
                if score > alpha:
                    old_alpha = alpha
                    alpha = score
                    log_message(f"{indent}[Q-depth {q_depth}]   -> NEW BEST! Alpha improved: {old_alpha} -> {alpha}")
                    print(f"{'  ' * (q_depth + 1)}[Q-depth {q_depth}] New best: {piece.name}x{capture_target}, score={score}")
                if beta <= alpha:
                    log_message(f"{indent}[Q-depth {q_depth}]   -> BETA CUTOFF! (beta={beta} <= alpha={alpha}) - pruning remaining {len(ordered_tactical) - moves_tried} captures")
                    print(f"{'  ' * (q_depth + 1)}[Q-depth {q_depth}] Beta cutoff at {piece.name}x{capture_target}, score={score}")
                    break
            else:
                # Minimizing player
                best_q_score = min(best_q_score, score)
                if score < beta:
                    old_beta = beta
                    beta = score
                    log_message(f"{indent}[Q-depth {q_depth}]   -> NEW BEST! Beta improved: {old_beta} -> {beta}")
                    print(f"{'  ' * (q_depth + 1)}[Q-depth {q_depth}] New best: {piece.name}x{capture_target}, score={score}")
                if beta <= alpha:
                    log_message(f"{indent}[Q-depth {q_depth}]   -> ALPHA CUTOFF! (beta={beta} <= alpha={alpha}) - pruning remaining {len(ordered_tactical) - moves_tried} captures")
                    print(f"{'  ' * (q_depth + 1)}[Q-depth {q_depth}] Alpha cutoff at {piece.name}x{capture_target}, score={score}")
                    break

        except Exception as e:
            log_message(f"{indent}[Q-depth {q_depth}] Exception during {piece.name} capture: {e}")
            continue

    log_message(f"{indent}[Q-depth {q_depth}] Finished analyzing {moves_tried}/{len(ordered_tactical)} captures - best score={best_q_score}")
    print(f"{'  ' * (q_depth + 1)}[Q-depth {q_depth}] Done, best score={best_q_score}")

    # Return the best score found (stand-pat if no captures improved position)
    return best_q_score


# ============================================================================
# SEARCH
# ============================================================================

import time
import random

def find_best_move(board, player, max_depth=10, time_limit=40.0):
    """
    Iterative deepening search using alpha-beta minimax with quiescence search.

    Args:
        board: Current game board object
        player: The player whose move we're finding
        max_depth: Maximum depth to search (default = 10)
        time_limit: Max search time in seconds (default = 40.0)

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

    # CRITICAL-2 OPTIMIZATION: Build move cache ONCE for the root position
    # This avoids redundant get_move_options() calls in order_moves
    root_move_cache = build_move_cache(board)

    # CRITICAL-3 OPTIMIZATION: Build position map ONCE for the root position
    # This avoids redundant O(n) scans for piece lookups
    root_pos_map = create_position_map(board)

    # Iterative deepening loop
    for depth in range(1, max_depth + 1):
        depth_start_time = time.time()
        depth_nodes_before = nodes_explored
        print(f"\n=== Depth {depth} (Quiescence {'ON' if QUIESCENCE_ENABLED else 'OFF'}) ===")

        # Time check before each new depth
        if time.time() - start_time >= time_limit:
            break

        # alpha: Best value that maximizing player can guarantee.
        # beta: Best value that the minimizing player can guarantee.

        alpha, beta = float('-inf'), float('inf')
        current_best_move = None
        current_best_score = float('-inf')

        ordered_moves = order_moves(board, legal_moves, root_move_cache, root_pos_map)

        # Try each move at this depth
        found_checkmate = False
        moves_evaluated = 0  # Track how many moves we fully evaluated
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

                # CRITICAL-2 OPTIMIZATION: Use cached moves instead of calling get_move_options()
                cached_moves = get_cached_moves(new_piece, root_move_cache)
                new_move = next((m for m in cached_moves
                                 if hasattr(m, "position") and m.position == move.position), None)
                if not new_move:
                    continue

                new_piece.move(new_move)
                print(f"Testing move at depth {depth}: {piece} to ({move.position.x},{move.position.y})")
                log_message(f"  Testing move at depth {depth}: {piece.name} from ({piece.position.x},{piece.position.y}) to ({move.position.x},{move.position.y})")

                # Switch turn
                new_board.current_player = [p for p in new_board.players if p != player][0]

                # Check game result after this move
                result = get_result(new_board)

                # Check for CHECKMATE - instant win!
                if result and "checkmate" in result.lower():
                    # Determine if we won or opponent won
                    opponent_name = "black" if player.name == "white" else "white"
                    if opponent_name in result.lower() and "loses" in result.lower():
                        # We delivered checkmate! This is the best possible move
                        print(f"  *** CHECKMATE FOUND: {piece} to ({move.position.x},{move.position.y}) ***")
                        log_message(f"  *** CHECKMATE FOUND: {piece.name} to ({move.position.x},{move.position.y}) - Instant win! ***")
                        return (piece, move)  # Return immediately, this is the best move possible

                # Check if this move causes a stalemate
                if is_stalemate(new_board):
                    current_eval = evaluate_board(board, player.name)

                    # If we're winning (eval > -500), skip this stalemate move
                    if current_eval > -500:
                        print(f"  -> Skipping stalemate move (current eval: {current_eval:.0f})")
                        log_message(f"  -> Skipping stalemate move (current eval: {current_eval:.0f})")
                        continue
                    else:
                        # If we're losing badly, stalemate is acceptable
                        print(f"  -> Accepting stalemate move (current eval: {current_eval:.0f})")
                        log_message(f"  -> Accepting stalemate move (current eval: {current_eval:.0f})")


                # Run minimax for OPPONENT's reply
                # This runs after our hypothetical move (that we are checking) so it works.
                log_message(f"  >> Starting minimax search for opponent's reply (depth={depth-1})")
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

                # Log the raw score from minimax
                log_message(f"    -> Minimax returned score: {score} for {player.name}")
                log_message(f"    -> Current best score: {current_best_score}, Testing if {score} > {current_best_score}")
                print(f"      Score: {score}")

                # If the current move is better than the previous best at this depth, update new best move.
                if score > current_best_score and score != float('inf') and score != float('-inf'):
                    current_best_score = score
                    current_best_move = (piece, move)
                    log_message(f"    -> New best move! Score: {score} (player: {player.name}, piece: {piece.name}, move: ({move.position.x},{move.position.y}))")
                    # Debug: Show when we find a very good move (potential checkmate)
                    # Check for inf/nan and treat as non-checkmate
                    if score >= 999999 and score < float('inf'):
                        print(f"  *** FOUND FORCED CHECKMATE: {piece} to ({move.position.x},{move.position.y}) with score {score}")
                        log_message(f"    *** FOUND FORCED CHECKMATE: {piece.name} to ({move.position.x},{move.position.y}) with score {score}")
                        found_checkmate = True
                        break  # Stop evaluating other moves - we have a forced win!

                # Update alpha for the minimax search (used in recursive calls)
                # but DON'T prune at root level - we want to evaluate all moves
                alpha = max(alpha, score)

                # Increment moves evaluated counter
                moves_evaluated += 1

            except Exception:
                continue

        # ============= Partial Depth Results Handling (Due to timeout) =============

        # Determine if we should use the results from this depth
        # We trust the results if:
        # 1. We evaluated at least one move, AND
        # 2. Either we finished all moves OR the new best score is significantly better
        total_moves = len(ordered_moves)
        all_moves_searched = (moves_evaluated == total_moves)

        if current_best_move and moves_evaluated > 0:
            # We found at least one move - decide whether to use it
            if all_moves_searched:
                # Completed the full depth - always use it
                best_move = current_best_move
                best_score = current_best_score
                print(f"Depth {depth} complete: Best move is {best_move[0].name} to ({best_move[1].position.x},{best_move[1].position.y}) with score {best_score}")
                log_message(f"Depth {depth} FULLY COMPLETED ({moves_evaluated}/{total_moves} moves) - Updated best move")
            else:
                # Partial search - only use if significantly better than previous depth
                improvement = current_best_score - best_score
                if improvement > 0:  # Improvement threshold
                    best_move = current_best_move
                    best_score = current_best_score
                    print(f"Depth {depth} PARTIAL ({moves_evaluated}/{total_moves} moves): Using move {best_move[0].name} with score {best_score} (improvement: +{improvement})")
                    log_message(f"Depth {depth} PARTIAL but IMPROVED - Updated best move (searched {moves_evaluated}/{total_moves})")
                else:
                    print(f"Depth {depth} INCOMPLETE ({moves_evaluated}/{total_moves} moves): Found {current_best_move[0].name} with score {current_best_score}, keeping previous best (score {best_score}, improvement only +{improvement})")
                    log_message(f"Depth {depth} INCOMPLETE - Keeping previous depth's best move (searched {moves_evaluated}/{total_moves})")
        else:
            # No moves were fully evaluated at this depth (likely due to timeout or all moves failed)
            if moves_evaluated == 0:
                print(f"Depth {depth} TIMEOUT: No moves completed ({total_moves} moves available) - using previous depth's best")
                log_message(f"Depth {depth} TIMEOUT before completing first move - keeping previous best")
            else:
                print(f"Depth {depth} ERROR: Moves evaluated but no valid move found - using previous depth's best")
                log_message(f"Depth {depth} ERROR: {moves_evaluated} moves evaluated but current_best_move is None")

        # Print statistics for this depth
        depth_nodes = nodes_explored - depth_nodes_before
        depth_time = time.time() - depth_start_time
        nodes_per_sec = depth_nodes / depth_time if depth_time > 0 else 0

        print(f"\n{'='*60}")
        print(f"Depth {depth} Summary:")
        print(f"  Nodes explored: {depth_nodes} ({nodes_per_sec:.0f} nodes/sec)")
        print(f"  Time taken: {depth_time:.3f}s")
        print(f"  Moves evaluated: {moves_evaluated}/{total_moves}")
        print(f"  Total nodes so far: {nodes_explored}")
        if current_best_move:
            print(f"  Best move: {current_best_move[0].name} -> ({current_best_move[1].position.x},{current_best_move[1].position.y})")
            print(f"  Best score: {current_best_score}")
        print(f"{'='*60}\n")

        log_message(f"\n{'='*60}")
        log_message(f"DEPTH {depth} SUMMARY:")
        log_message(f"  Nodes explored: {depth_nodes} in {depth_time:.3f}s ({nodes_per_sec:.0f} nodes/sec)")
        log_message(f"  Total nodes: {nodes_explored}")
        log_message(f"  Moves evaluated: {moves_evaluated}/{total_moves}")
        if current_best_move:
            log_message(f"  Best move at depth {depth}: {current_best_move[0].name} to ({current_best_move[1].position.x},{current_best_move[1].position.y}) with score {current_best_score}")
        else:
            log_message(f"  No valid move found at depth {depth}, keeping previous best")
        log_message(f"{'='*60}\n")

        # EARLY TERMINATION: If we found a forced checkmate, stop iterative deepening immediately
        # No need to search deeper - we already have a guaranteed winning sequence!
        if found_checkmate or current_best_score >= 999999:
            print(f"*** FORCED CHECKMATE SEQUENCE FOUND AT DEPTH {depth} - Stopping search immediately ***")
            print(f"*** Playing: {best_move[0]} to ({best_move[1].position.x},{best_move[1].position.y}) ***")
            log_message(f"*** FORCED CHECKMATE SEQUENCE FOUND AT DEPTH {depth} - Stopping search immediately ***")
            break

        # Stop if time runs out mid-search
        if time.time() - start_time >= time_limit:
            print(f"Time limit reached after depth {depth}")
            log_message(f"Time limit reached after depth {depth}")
            break

    # Log final decision
    total_time = time.time() - start_time
    print(f"\n{'#'*60}")
    print(f"### SEARCH COMPLETE ###")
    print(f"{'#'*60}")
    print(f"Final Move: {best_move[0].name} -> ({best_move[1].position.x},{best_move[1].position.y})")
    print(f"Final Score: {best_score}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Total Nodes: {nodes_explored}")
    print(f"Nodes/Second: {nodes_explored / total_time:.0f}")
    print(f"Quiescence: {'ENABLED' if QUIESCENCE_ENABLED else 'DISABLED'}")
    print(f"{'#'*60}\n")

    log_message(f"\n{'#'*60}")
    log_message(f"### FINAL MOVE SELECTION ###")
    log_message(f"{'#'*60}")
    log_message(f"Move: {best_move[0].name} from ({best_move[0].position.x},{best_move[0].position.y}) to ({best_move[1].position.x},{best_move[1].position.y})")
    log_message(f"Score: {best_score}")
    log_message(f"Total search time: {total_time:.2f}s")
    log_message(f"Total nodes explored: {nodes_explored}")
    log_message(f"Average nodes/second: {nodes_explored / total_time:.0f}")
    log_message(f"Quiescence search: {'ENABLED' if QUIESCENCE_ENABLED else 'DISABLED'}")
    log_message(f"{'#'*60}\n")

    return best_move

# ============================================================================
# MINIMAX + ALPHA-BETA + QUIESCENCE
# ============================================================================

import time

def minimax(board, depth, alpha, beta, player_name, time_limit, start_time, is_max_turn=True, indent_level=0) -> float:
    """
    Minimax search with alpha-beta pruning, time awareness, and quiescence search.

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
        # FIX: Check for ANY checkmate, not just "checkmate - black loses"
        if "checkmate" in res:
            # Prefer faster checkmates: add bonus for shallower depth
            # If we're at depth 5, checkmate is 5 moves away → bonus = (10-5) = 5
            # If we're at depth 1, checkmate is 1 move away → bonus = (10-1) = 9
            mate_bonus = (10 - depth) * 1000

            # Check if we (player_name) won or the opponent won
            # Example results: "checkmate - black loses" or "checkmate - white loses"
            if player_name in res or (player_name == "white" and "black loses" in res) or (player_name == "black" and "white loses" in res):
                return 999999 + mate_bonus  # We win: prefer faster mate
            else:
                return -999999 - mate_bonus  # We lose: delay mate as long as possible
        elif "draw" in res:
            return 0

    # --- QUIESCENCE SEARCH AT LEAF NODES ---
    if depth == 0:
        if QUIESCENCE_ENABLED:
            # Instead of static evaluation, call quiescence search
            # This resolves tactical sequences to quiet positions
            # Pass is_max_turn to maintain consistent perspective
            indent = "  " * indent_level
            log_message(f"{indent}[Depth 0] Entering QUIESCENCE SEARCH (alpha={alpha}, beta={beta}, is_max_turn={is_max_turn})")
            q_score = quiescence_search(board, alpha, beta, player_name, time_limit, start_time, is_max_turn, q_depth=0)
            log_message(f"{indent}[Depth 0] Quiescence returned: {q_score}")
            return q_score
        else:
            static_eval = evaluate_board(board, player_name)
            indent = "  " * indent_level
            log_message(f"{indent}[Depth 0] Static evaluation (Q-search OFF): {static_eval}")
            return static_eval

    # --- 2 Get legal moves ----------------------------------------------
    current_player = board.current_player
    legal_moves = list_legal_moves_for(board, current_player)
    if not legal_moves:
        return evaluate_board(board, player_name)

    # --- 3 Initialize best value ----------------------------------------
    best_value = float('-inf') if is_max_turn else float('inf')

    # --- 4 Order moves for better pruning -------------------------------
    # CRITICAL-2 OPTIMIZATION: Build move cache once per board state
    move_cache = build_move_cache(board)

    # CRITICAL-3 OPTIMIZATION: Build position map once per board state
    pos_map = create_position_map(board)

    ordered_moves = order_moves(board, legal_moves, move_cache, pos_map)

    # --- 5 Explore moves ------------------------------------------------
    for piece, move in ordered_moves:
        # Check time inside loop too
        if time.time() - start_time >= time_limit:
            break

        # DEBUG: Print what's being explored (comment out for production)
        indent = "  " * indent_level
        turn_type = "MAX" if is_max_turn else "MIN"
        print(f"{indent}[Depth {depth}, {turn_type}] Testing: {piece.name} to ({move.position.x},{move.position.y})")
        log_message(f"{indent}[Depth {depth}, {turn_type}] Testing: {piece.name} to ({move.position.x},{move.position.y})")

        new_board = board.clone()
        try:
            # Locate piece & move equivalents on cloned board
            new_piece = next((p for p in new_board.get_player_pieces(current_player)
                              if type(p) == type(piece) and p.position == piece.position), None)
            if not new_piece:
                continue

            # CRITICAL-2 OPTIMIZATION: Use cached moves instead of calling get_move_options()
            cached_moves = get_cached_moves(new_piece, move_cache)
            new_move = next((m for m in cached_moves
                             if hasattr(m, "position") and m.position == move.position), None)

            if not new_move:
                continue

            new_piece.move(new_move)
            # Switch turn
            new_board.current_player = [p for p in new_board.players if p != current_player][0]

            # Check for stalemate using our helper function
            if is_stalemate(new_board):
                # Stalemate in the middle of a minimax tree:
                # If we're maximizing and caused a stalemate, that's bad (unless we're losing)
                # If we're minimizing and caused a stalemate, that's good (opponent forced a draw)
                current_eval = evaluate_board(new_board, player_name)
                if is_max_turn and current_eval > 0:
                    # We caused stalemate - bad if we're winning
                    return -50000
                else:
                    # Opponent caused stalemate - good if we're losing
                    return 50000

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

            # Log the returned value for debugging
            log_message(f"{indent}  -> Returned: {value} (alpha={alpha}, beta={beta})")

            # --- 8️⃣ Alpha-beta updates -----------------------------------
            if is_max_turn:
                best_value = max(best_value, value)
                alpha = max(alpha, value)
                if value > best_value or best_value == float('-inf'):
                    log_message(f"{indent}  -> New MAX best: {value}")
            else:
                best_value = min(best_value, value)
                beta = min(beta, value)
                if value < best_value or best_value == float('inf'):
                    log_message(f"{indent}  -> New MIN best: {value}")

            if beta <= alpha:
                log_message(f"{indent}  -> PRUNED! (beta={beta} <= alpha={alpha})")
                break  # Prune rest of branch

        except Exception:
            continue

    return best_value if best_value != float('inf') and best_value != float('-inf') else evaluate_board(board, player_name)

# ============================================================================
# AGENT ENTRY POINT
# ============================================================================

def agent(board, player, var):
    """
    Main agent entry point for COMP2321 system.

    This version includes Phase 1: Quiescence Search
    """
    piece, move = find_best_move(board, player, time_limit=12.5)
    if piece is None or move is None:
        legal = list_legal_moves_for(board, player)
        if legal:
            piece, move = random.choice(legal)
            log_message(f"No best move found, playing random move: {piece.name} to ({move.position.x},{move.position.y})")
    else:
        log_message(f"\n*** FINAL DECISION: Playing {piece.name} from ({piece.position.x},{piece.position.y}) to ({move.position.x},{move.position.y}) ***")

    return piece, move
