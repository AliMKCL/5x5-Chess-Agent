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
                    num_diff, val_diff = attacker_defender_ratio(board, move.position, opponent, piece.player)
                    
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

        ordered_moves = order_moves(board, legal_moves)

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

                new_move = next((m for m in new_piece.get_move_options()
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
            print(f"Depth {depth} complete: No valid move found at this depth (using previous depth's best)")
        
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
        log_message(f"{indent}[Depth {depth}, {turn_type}] Testing: {piece.name} to ({move.position.x},{move.position.y})")

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