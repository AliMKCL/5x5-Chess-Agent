"""
Bitboard-Based Chess Agent with Minimax Search

This agent uses the bitboard infrastructure from helpersBitboard.py to achieve
dramatically faster move generation and position evaluation.

FEATURES:
- Minimax search with alpha-beta pruning
- Iterative deepening (depth 1-12)
- Transposition table (TT) for position caching
- Quiescence search for tactical sequences
- MVV-LVA move ordering
- Time management with fallback

EXPECTED PERFORMANCE:
- 50,000-200,000 nodes/second (vs 1,000-20,000 without bitboards)
- Effective search depth: 8-12 plies (vs 2-3 without bitboards)
- TT hit rate: 30-60% at depth 8

INTERFACE:
- agent(board, player, time_remaining) → (piece, move_option)
  Main entry point compatible with chessmaker framework
"""

import time
from typing import List, Optional, Tuple
from helpersBitboard import (
    board_to_bitboard, generate_legal_moves, apply_move, evaluate_bitboard,
    is_in_check, static_exchange_eval, BitboardState, BBMove, index_to_xy, square_index,
    PIECE_VALUES, PAWN, KNIGHT, BISHOP, QUEEN, KING, RIGHT
)
from helpersT import TranspositionTable
from chessmaker.chess.base.board import Position

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Transposition table (64MB default - adjust based on available memory)
TRANSPOSITION_TABLE = TranspositionTable(size_mb=64)

# Search configuration
MAX_DEPTH = 12          # Maximum search depth for iterative deepening
QUIESCENCE_MAX_DEPTH = 7  # Maximum quiescence search depth
TIME_LIMIT = 12.5       # Time limit in seconds (leave buffer for move conversion)

# Score constants
CHECKMATE_SCORE = 999_999
STALEMATE_SCORE = 0
DRAW_SCORE = 0

# Move ordering piece values (for MVV-LVA)
# Must match PIECE_VALUES from helpersBitboard.py exactly!
MVV_LVA_VALUES = [100, 330, 320, 900, 20000, 500]  # P, N, B, Q, K, Right (indices 0-5)

# Statistics tracking
stats = {
    'nodes_searched': 0,
    'quiescence_nodes': 0,
    'tt_hits': 0,
    'tt_stores': 0,
    'cutoffs': 0,
    'depth_reached': 0
}

# Logging
LOG_FILE = "game_log.txt"
LOGGING_ENABLED = True


# ============================================================================
# MOVE ORDERING
# ============================================================================

def score_move(move: BBMove, bb_state: BitboardState, tt_best_move: Optional[Tuple] = None) -> int:
    """
    Score a move for ordering purposes (higher = better).

    Ordering priority:
    1. TT best move (from previous search)
    2. Captures sorted by MVV-LVA + SEE (Static Exchange Evaluation)
    3. Non-captures (neutral score)

    Args:
        move: BBMove to score
        bb_state: Current bitboard state (needed for SEE)
        tt_best_move: Optional best move from transposition table as (from_sq, to_sq) tuple

    Returns:
        Integer score (higher scores searched first)
    """
    # TT move gets highest priority
    if tt_best_move and (move.from_sq, move.to_sq) == tt_best_move:
        return 10_000_000

    # Captures: MVV-LVA scoring + SEE bonus
    if move.captured_type != -1:  # -1 means no capture (not None!)
        victim_value = MVV_LVA_VALUES[move.captured_type]
        attacker_value = MVV_LVA_VALUES[move.piece_type]

        # Base MVV-LVA score: prefer high-value victims with low-value attackers
        base_score = (victim_value * 10) - attacker_value

        # Static Exchange Evaluation: analyze if this capture is safe/profitable
        see_score = static_exchange_eval(bb_state, move)

        # Scaled SEE bonus: if SEE > 0 (winning exchange), add SEE value + 500
        # This matches agentT's philosophy: favorable exchanges get significant boost
        if see_score > 0:
            # Winning capture: big bonus (like agentT's +1000, but scaled by actual gain)
            return base_score + see_score + 500
        else:
            # Losing/equal capture: just use MVV-LVA
            # Still might be worth it tactically, so don't penalize heavily
            return base_score

    # Non-captures get neutral score
    return 0


def order_moves(moves: List[BBMove], bb_state: BitboardState, tt_best_move: Optional[Tuple] = None) -> List[BBMove]:
    """
    Order moves for optimal alpha-beta pruning.

    Args:
        moves: List of BBMove objects to order
        bb_state: Current bitboard state (needed for SEE in capture scoring)
        tt_best_move: Optional (from_sq, to_sq) tuple from TT

    Returns:
        Sorted list of moves (best moves first)
    """
    return sorted(moves, key=lambda m: score_move(m, bb_state, tt_best_move), reverse=True)


# ============================================================================
# QUIESCENCE SEARCH
# ============================================================================

def quiescence_search(bb_state: BitboardState, alpha: int, beta: int,
                     depth: int, max_depth: int, start_time: float,
                     is_maximizing: bool, player_is_white: bool) -> int:
    """
    Quiescence search to resolve tactical sequences (captures only).

    STANDARD MINIMAX VERSION (not negamax):
    - is_maximizing=True: Maximizing player tries to maximize score
    - is_maximizing=False: Minimizing player tries to minimize score
    - Evaluation always from player_is_white's perspective

    This prevents the horizon effect where the engine misses tactical blows
    just beyond the search depth. We search capture sequences until reaching
    a "quiet" position (no captures available or max depth).

    Args:
        bb_state: Current bitboard position
        alpha: Alpha bound (best score maximizer can guarantee)
        beta: Beta bound (best score minimizer can guarantee)
        depth: Current quiescence depth (starts at 0, increments each ply)
        max_depth: Maximum quiescence depth allowed
        start_time: Search start time for timeout checking
        is_maximizing: True if maximizing player's turn, False for minimizing
        player_is_white: True if evaluating from white's perspective (constant throughout search)

    Returns:
        Static evaluation or best capture sequence score
    """
    global stats
    stats['quiescence_nodes'] += 1

    # Timeout check
    if time.time() - start_time > TIME_LIMIT:
        return evaluate_bitboard(bb_state, player_is_white)

    # Stand-pat score: current position evaluation (always from player's perspective)
    stand_pat = evaluate_bitboard(bb_state, player_is_white)

    # Standard minimax cutoff logic (not negamax)
    if is_maximizing:
        # Maximizing player: if stand-pat already >= beta, opponent won't allow this
        if stand_pat >= beta:
            return beta
        # Update alpha
        if stand_pat > alpha:
            alpha = stand_pat
    else:
        # Minimizing player: if stand-pat already <= alpha, we won't allow this
        if stand_pat <= alpha:
            return alpha
        # Update beta
        if stand_pat < beta:
            beta = stand_pat

    # Max depth reached: return static evaluation
    if depth >= max_depth:
        return stand_pat

    # Generate captures only
    captures = generate_legal_moves(bb_state, captures_only=True)

    # No captures available: quiet position
    if not captures:
        return stand_pat

    # Order captures by MVV-LVA + SEE
    captures = order_moves(captures, bb_state)

    # Initialize best score based on whose turn it is
    best_score = stand_pat

    # Search captures (STANDARD MINIMAX - no negation)
    for move in captures:
        new_state = apply_move(bb_state, move)

        # Recursive call with flipped is_maximizing (no negation!)
        score = quiescence_search(new_state, alpha, beta, depth + 1, max_depth,
                                 start_time, not is_maximizing, player_is_white)

        # Standard minimax update
        if is_maximizing:
            # Maximizing player
            best_score = max(best_score, score)
            if score > alpha:
                alpha = score
            if beta <= alpha:
                return beta  # Beta cutoff
        else:
            # Minimizing player
            best_score = min(best_score, score)
            if score < beta:
                beta = score
            if beta <= alpha:
                return alpha  # Alpha cutoff

    return best_score


# ============================================================================
# MINIMAX WITH ALPHA-BETA PRUNING
# ============================================================================

def minimax(bb_state: BitboardState, depth: int, alpha: int, beta: int,
           is_maximizing: bool, start_time: float, root_depth: int,
           player_is_white: bool) -> int:
    """
    Minimax search with alpha-beta pruning and transposition table.

    STANDARD MINIMAX VERSION (not negamax):
    - is_maximizing=True: Maximizing player tries to maximize score
    - is_maximizing=False: Minimizing player tries to minimize score
    - Evaluation always from player_is_white's perspective
    - NO negation of scores or alpha/beta bounds

    This is the core search algorithm that explores the game tree to find
    the best move. Alpha-beta pruning eliminates branches that cannot
    influence the final decision.

    Args:
        bb_state: Current bitboard position
        depth: Remaining depth to search
        alpha: Best score maximizer can guarantee (lower bound)
        beta: Best score minimizer can guarantee (upper bound)
        is_maximizing: True if maximizing player's turn
        start_time: Search start time for timeout checking
        root_depth: Initial depth (for mate bonus calculation)
        player_is_white: True if evaluating from white's perspective (constant throughout search)

    Returns:
        Best evaluation score from this position
    """
    global stats
    stats['nodes_searched'] += 1

    # Timeout check
    if time.time() - start_time > TIME_LIMIT:
        return evaluate_bitboard(bb_state, player_is_white)

    # Probe transposition table
    tt_score, tt_move = TRANSPOSITION_TABLE.probe(bb_state.zobrist_hash, depth)
    if tt_score is not None:
        stats['tt_hits'] += 1
        return tt_score

    # Terminal depth: enter quiescence search
    if depth <= 0:
        return quiescence_search(bb_state, alpha, beta, 0, QUIESCENCE_MAX_DEPTH,
                                start_time, is_maximizing, player_is_white)

    # Generate all legal moves
    moves = generate_legal_moves(bb_state)

    # Terminal node: checkmate or stalemate
    if not moves:
        in_check = is_in_check(bb_state, bb_state.side_to_move == 0)
        if in_check:
            # Checkmate: the current side_to_move is checkmated
            # Determine if that's good or bad for player_is_white
            mate_bonus = (root_depth - depth) * 1000

            # If white is in checkmate
            if bb_state.side_to_move == 0:  # white to move but checkmated
                if player_is_white:
                    # Bad for white (we lose)
                    return -CHECKMATE_SCORE - mate_bonus
                else:
                    # Good for black (we win)
                    return CHECKMATE_SCORE + mate_bonus
            else:  # black to move but checkmated
                if player_is_white:
                    # Good for white (we win)
                    return CHECKMATE_SCORE + mate_bonus
                else:
                    # Bad for black (we lose)
                    return -CHECKMATE_SCORE - mate_bonus
        else:
            # Stalemate
            return STALEMATE_SCORE

    # Extract TT move for ordering (if available)
    tt_best_move_tuple = None
    if tt_move:
        # Convert from framework format (((x1,y1), (x2,y2))) to (from_sq, to_sq)
        if isinstance(tt_move, tuple) and len(tt_move) == 2:
            from_pos, to_pos = tt_move
            if isinstance(from_pos, tuple) and isinstance(to_pos, tuple):
                from_sq = square_index(from_pos[0], from_pos[1])
                to_sq = square_index(to_pos[0], to_pos[1])
                tt_best_move_tuple = (from_sq, to_sq)

    # Order moves
    moves = order_moves(moves, bb_state, tt_best_move_tuple)

    best_score = -float('inf') if is_maximizing else float('inf')
    best_move = None

    # STANDARD MINIMAX: No negation of scores or bounds
    for move in moves:
        new_state = apply_move(bb_state, move)

        # Recursive call with flipped is_maximizing (NO negation!)
        score = minimax(new_state, depth - 1, alpha, beta, not is_maximizing,
                       start_time, root_depth, player_is_white)

        # Update best score based on whose turn it is
        if is_maximizing:
            # Maximizing player
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
        else:
            # Minimizing player
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, score)

        # Alpha-beta cutoff
        if beta <= alpha:
            stats['cutoffs'] += 1
            break

    # Store in transposition table
    if best_move:
        # Convert BBMove to framework format for TT
        from_x, from_y = index_to_xy(best_move.from_sq)
        to_x, to_y = index_to_xy(best_move.to_sq)
        tt_move_tuple = ((from_x, from_y), (to_x, to_y))
        TRANSPOSITION_TABLE.store(bb_state.zobrist_hash, depth, best_score, tt_move_tuple)
        stats['tt_stores'] += 1

    return best_score


# ============================================================================
# ITERATIVE DEEPENING
# ============================================================================

def find_best_move(bb_state: BitboardState, max_depth: int, time_limit: float,
                  player_is_white: bool) -> Optional[BBMove]:
    """
    Find the best move using iterative deepening.

    STANDARD MINIMAX VERSION:
    - Root node is always maximizing player
    - player_is_white determines evaluation perspective

    Iterative deepening searches progressively deeper (depth 1, 2, 3, ...)
    until time runs out. This provides:
    1. Anytime algorithm: always have a valid move ready
    2. Better move ordering: previous depth results guide next depth
    3. Time management: can stop gracefully when time expires

    Args:
        bb_state: Current bitboard position
        max_depth: Maximum depth to search
        time_limit: Time limit in seconds
        player_is_white: True if evaluating from white's perspective

    Returns:
        Best BBMove found, or None if no legal moves
    """
    global stats
    start_time = time.time()
    best_move = None
    best_score = -float('inf')

    # Get all legal moves
    moves = generate_legal_moves(bb_state)
    if not moves:
        return None  # No legal moves (checkmate or stalemate)

    # Iterative deepening loop
    for depth in range(1, max_depth + 1):
        # Check if we have time for this depth
        elapsed = time.time() - start_time
        if elapsed > time_limit * 0.9:  # Reserve 10% buffer
            if LOGGING_ENABLED:
                log_message(f"Time limit approaching, stopping at depth {depth-1}")
            print(f"Time limit approaching, stopping at depth {depth-1}")
            break

        depth_start_time = time.time()
        depth_nodes_before = stats['nodes_searched']
        depth_best_move = None
        depth_best_score = -float('inf')

        # Removed verbose depth start logging - only show completion
        if LOGGING_ENABLED:
            log_message(f"\n=== Depth {depth} (Quiescence ON) ===")

        # Probe TT for move ordering hint
        tt_score, tt_move = TRANSPOSITION_TABLE.probe(bb_state.zobrist_hash, depth)
        tt_best_move_tuple = None
        if tt_move:
            if isinstance(tt_move, tuple) and len(tt_move) == 2:
                from_pos, to_pos = tt_move
                if isinstance(from_pos, tuple) and isinstance(to_pos, tuple):
                    from_sq = square_index(from_pos[0], from_pos[1])
                    to_sq = square_index(to_pos[0], to_pos[1])
                    tt_best_move_tuple = (from_sq, to_sq)

        # Order moves for this depth
        ordered_moves = order_moves(moves, bb_state, tt_best_move_tuple)

        # Search all moves at current depth
        # Root is always maximizing (we're finding our best move)
        alpha = -float('inf')
        beta = float('inf')

        for move in ordered_moves:
            new_state = apply_move(bb_state, move)

            # After our move, opponent plays (is_maximizing=False)
            # NO negation in standard minimax!
            score = minimax(new_state, depth - 1, alpha, beta, False,
                          start_time, depth, player_is_white)

            if score > depth_best_score:
                depth_best_score = score
                depth_best_move = move
                alpha = score

            # Timeout check
            if time.time() - start_time > time_limit:
                if LOGGING_ENABLED:
                    log_message(f"Timeout during depth {depth}, using previous depth result")
                return best_move if best_move else depth_best_move

        # Update best move
        best_move = depth_best_move
        best_score = depth_best_score
        stats['depth_reached'] = depth

        depth_time = time.time() - depth_start_time
        depth_nodes = stats['nodes_searched'] - depth_nodes_before

        # Simple depth completion logging
        print(f"Depth {depth} complete, in {depth_time:.2f}s, nodes visited: {depth_nodes}")
        if LOGGING_ENABLED:
            log_message(f"Depth {depth} complete, in {depth_time:.2f}s, nodes visited: {depth_nodes}")

        # Early termination: found forced checkmate
        if abs(best_score) >= CHECKMATE_SCORE - 10000:
            if LOGGING_ENABLED:
                log_message(f"Checkmate found at depth {depth}, stopping search")
            break

    return best_move


# ============================================================================
# CHESSMAKER FRAMEWORK BRIDGE
# ============================================================================

def bbmove_to_framework_move(bb_move: BBMove, board):
    """
    Convert a BBMove to chessmaker framework format.

    Args:
        bb_move: BBMove object from bitboard search
        board: Chessmaker board object

    Returns:
        Tuple of (piece, move_option) compatible with framework
    """
    # Convert square indices to (x, y) coordinates
    from_x, from_y = index_to_xy(bb_move.from_sq)
    to_x, to_y = index_to_xy(bb_move.to_sq)

    # Find the piece at the from position
    from_pos = Position(from_x, from_y)
    piece = None
    for p in board.get_pieces():
        if p.position == from_pos:
            piece = p
            break

    if not piece:
        raise ValueError(f"No piece found at position ({from_x}, {from_y})")

    # Find the matching move option
    to_pos = Position(to_x, to_y)
    for move_option in piece.get_move_options():
        if move_option.position == to_pos:
            return (piece, move_option)

    # Should not reach here if move generation is correct
    raise ValueError(f"Move option not found: {piece} from ({from_x},{from_y}) to ({to_x},{to_y})")


# ============================================================================
# LOGGING
# ============================================================================

def log_message(message: str):
    """Log a message to the game log file."""
    if LOGGING_ENABLED:
        try:
            with open(LOG_FILE, 'a') as f:
                f.write(f"{message}\n")
        except:
            pass  # Silently fail if logging fails


def log_search_statistics():
    """Log search statistics to the game log."""
    if LOGGING_ENABLED:
        log_message("\n" + "="*60)
        log_message("BITBOARD SEARCH STATISTICS")
        log_message("="*60)
        log_message(f"Depth reached: {stats['depth_reached']}")
        log_message(f"Nodes searched: {stats['nodes_searched']:,}")
        log_message(f"Quiescence nodes: {stats['quiescence_nodes']:,}")
        log_message(f"Total nodes: {stats['nodes_searched'] + stats['quiescence_nodes']:,}")
        log_message(f"TT hits: {stats['tt_hits']:,}")
        log_message(f"TT stores: {stats['tt_stores']:,}")
        log_message(f"Alpha-beta cutoffs: {stats['cutoffs']:,}")
        if stats['nodes_searched'] > 0:
            tt_hit_rate = (stats['tt_hits'] / stats['nodes_searched']) * 100
            log_message(f"TT hit rate: {tt_hit_rate:.1f}%")
        log_message("="*60 + "\n")


def reset_statistics():
    """Reset search statistics for new move."""
    global stats
    stats = {
        'nodes_searched': 0,
        'quiescence_nodes': 0,
        'tt_hits': 0,
        'tt_stores': 0,
        'cutoffs': 0,
        'depth_reached': 0
    }


# ============================================================================
# MAIN AGENT FUNCTION
# ============================================================================

def agent(board, player, var):
    """
    Main agent function compatible with chessmaker framework.

    This is the entry point called by the game engine. It:
    1. Converts the board to bitboard representation
    2. Runs bitboard-based search
    3. Converts the best move back to framework format

    Args:
        board: Chessmaker board object
        player: Player object (contains name, color, etc.)
        time_remaining: Remaining time in seconds

    Returns:
        Tuple of (piece, move_option) representing the best move
    """
    reset_statistics()

    if LOGGING_ENABLED:
        log_message("\n" + "="*60)
        log_message(f"BITBOARD AGENT MOVE - {player.name}")
        log_message("="*60)

    start_time = time.time()

    # Convert board to bitboard representation
    bb_state = board_to_bitboard(board, player)
    player_is_white = player.name == "white"

    if LOGGING_ENABLED:
        initial_eval = evaluate_bitboard(bb_state, player_is_white)
        log_message(f"Initial evaluation: {initial_eval}")

    # Find best move using iterative deepening
    best_move = find_best_move(bb_state, MAX_DEPTH, TIME_LIMIT, player_is_white)

    if not best_move:
        # No legal moves available (should not happen in normal play)
        if LOGGING_ENABLED:
            log_message("ERROR: No legal moves found!")
        # Fallback: return any legal move from framework
        for piece in board.pieces:
            if piece.color == player.color:
                moves = piece.get_move_options(board)
                if moves:
                    return (piece, moves[0])
        raise ValueError("No legal moves available")

    # Convert to framework format
    piece, move_option = bbmove_to_framework_move(best_move, board)

    # Log results
    search_time = time.time() - start_time
    
    # Calculate TT hit rate
    total_probes = stats['nodes_searched']
    tt_hits = stats['tt_hits']
    tt_hit_rate = (tt_hits / total_probes * 100) if total_probes > 0 else 0
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Total nodes visited: {stats['nodes_searched']}")
    print(f"Max depth reached: {stats['depth_reached']}")
    print(f"TT cache hit rate: {tt_hit_rate:.1f}% ({tt_hits}/{total_probes})")
    print(f"{'='*60}\n")
    
    if LOGGING_ENABLED:
        from_x, from_y = index_to_xy(best_move.from_sq)
        to_x, to_y = index_to_xy(best_move.to_sq)
        log_message(f"\nBest move: {piece} from ({from_x},{from_y}) to ({to_x},{to_y})")
        log_message(f"Search time: {search_time:.2f}s")
        log_message(f"Total nodes visited: {stats['nodes_searched']}")
        log_message(f"Max depth reached: {stats['depth_reached']}")
        log_message(f"TT cache hit rate: {tt_hit_rate:.1f}% ({tt_hits}/{total_probes})")
        log_search_statistics()

    return (piece, move_option)


# ============================================================================
# MODULE TEST
# ============================================================================

if __name__ == "__main__":
    print("agentBitboard.py - Bitboard-based chess agent")
    print("="*60)
    print("This module provides a high-performance chess agent using bitboards.")
    print("\nKey features:")
    print("  - Minimax with alpha-beta pruning")
    print("  - Iterative deepening (depth 1-12)")
    print("  - Transposition table (64MB)")
    print("  - Quiescence search (depth 7)")
    print("  - MVV-LVA move ordering")
    print("\nExpected performance:")
    print("  - 50,000-200,000 nodes/second")
    print("  - Search depth: 8-12 plies")
    print("  - TT hit rate: 30-60%")
    print("="*60)
