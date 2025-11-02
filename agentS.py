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

def init_log_file():
    """Initialize the log file for this game session."""
    global LOG_FILE
    LOG_FILE = open("game_log.txt", "w")
    LOG_FILE.write("=== GAME LOG ===\n\n")

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

def order_moves(board, moves, move_cache=None):
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
    """
    scored_moves = []
    
    # Build move cache if not provided (for backwards compatibility)
    if move_cache is None:
        move_cache = build_move_cache(board)
    
    # Create position-to-piece lookup for faster capture evaluation
    pos_to_piece = {piece.position: piece for piece in board.get_pieces()}
    
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
                target = pos_to_piece.get(capture_pos)
                if target:
                    victim_value = get_piece_value(target)
                    
                    # Get opponent player for exchange evaluation
                    opponent = next(p for p in board.players if p != piece.player)
                    num_diff, val_diff = attacker_defender_ratio(board, move.position, opponent, piece.player, move_cache)
                    
                    # Base MVV-LVA score: prefer low-value attackers capturing high-value victims
                    # Using (victim * 10) ensures victim value is prioritized
                    base_mvv_lva = (victim_value * 10) - attacker_value

                    # Case 1: More or equal defenders than attackers
                    if not val_diff or val_diff < 0:
                        score += base_mvv_lva    # Check if weakest attacker < victim_value
                        log_message(f"Equal/losing numbers but potentially favorable: {piece.name} captures {target.name}, num_diff={num_diff}, score = {score} move {piece.name} to ({move.position.x},{move.position.y})")

                    # Case 2: More attackers than defenders, and positive trade
                    elif val_diff > 0:
                        score += base_mvv_lva + 1000    # Capture with the weakest attacker and prefer the capture
                        log_message(f"Winning exchange: {piece.name} captures {target.name}, net={val_diff}, score={score}, move {piece.name} to ({move.position.x},{move.position.y})")


        
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
    
    # CRITICAL-2 OPTIMIZATION: Build move cache ONCE for the root position
    # This avoids redundant get_move_options() calls in order_moves
    root_move_cache = build_move_cache(board)

    # Iterative deepening loop
    for depth in range(1, max_depth + 1):
        depth_start_time = time.time()
        depth_nodes_before = nodes_explored
        print(f"\n=== Depth {depth} ===")

        # Time check before each new depth
        if time.time() - start_time >= time_limit:
            break

        # alpha: Best value that maximizing player can guarantee.
        # beta: Best value that the minimizing player can guarantee.

        alpha, beta = float('-inf'), float('inf')
        current_best_move = None
        current_best_score = float('-inf')

        ordered_moves = order_moves(board, legal_moves, root_move_cache)

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

                # If the current move is better than the previous best at this depth, update new best move.
                if score > current_best_score:
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
        print(f"Depth {depth} stats: {depth_nodes} nodes explored in {depth_time:.3f}s (Total: {nodes_explored} nodes)")
        log_message(f"Depth {depth} complete: {depth_nodes} nodes explored in {depth_time:.3f}s (Total: {nodes_explored} nodes)")
        if current_best_move:
            log_message(f"Best move at depth {depth}: {current_best_move[0].name} to ({current_best_move[1].position.x},{current_best_move[1].position.y}) with score {current_best_score}")
        else:
            log_message(f"No valid move found at depth {depth}, keeping previous best")
        
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
    print(f"\n*** FINAL MOVE SELECTION: {best_move[0].name} to ({best_move[1].position.x},{best_move[1].position.y}) with score {best_score} ***")
    log_message(f"\n*** FINAL MOVE SELECTION: {best_move[0].name} to ({best_move[1].position.x},{best_move[1].position.y}) with score {best_score} ***")
    
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
    # CRITICAL-2 OPTIMIZATION: Build move cache once per board state
    move_cache = build_move_cache(board)
    ordered_moves = order_moves(board, legal_moves, move_cache)

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

    return best_value

# ============================================================================
# AGENT ENTRY POINT
# ============================================================================

def agent(board, player, var):
    """
    Main agent entry point for COMP2321 system.
    """
    piece, move = find_best_move(board, player, time_limit=60)
    if piece is None or move is None:
        legal = list_legal_moves_for(board, player)
        if legal:
            piece, move = random.choice(legal)
            log_message(f"No best move found, playing random move: {piece.name} to ({move.position.x},{move.position.y})")
    else:
        log_message(f"\n*** FINAL DECISION: Playing {piece.name} from ({piece.position.x},{piece.position.y}) to ({move.position.x},{move.position.y}) ***")
    
    # Add a 1-second delay before continuing after a move is made
    #time.sleep(1)
    return piece, move