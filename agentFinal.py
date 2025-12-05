"""
IMPROVEMENTS LEFT:
- LMR (better pruning)
- Add double pawn move (currently not available)
- Stalemate detection
- Possible 32-Bit architecture --> 64 bit


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
from typing import List, Optional, Tuple, Dict
from helpersFinal import (
    board_to_bitboard, generate_legal_moves, apply_move, evaluate_bitboard,
    is_in_check, static_exchange_eval, BitboardState, BBMove, index_to_xy, square_index,
    PIECE_VALUES, PAWN, KNIGHT, BISHOP, QUEEN, KING, RIGHT
)
from chessmaker.chess.base.board import Position



# ============================================================================
# BITBOARD-NATIVE TRANSPOSITION TABLE
# ============================================================================

class BitboardTranspositionTable:
    """
    Bitboard-optimized Transposition Table for caching position evaluations.

    This implementation is specifically designed to work with BitboardState
    and uses the Zobrist hash already computed in the bitboard state.

    STRUCTURE:
        Dict mapping: zobrist_hash → (depth, score, best_move_tuple)

    REPLACEMENT STRATEGY:
        - Always-replace with depth preference
        - If hash exists: only overwrite if new depth >= old depth
        - If table full: evict oldest entry (FIFO)

    MEMORY EFFICIENCY:
        Each entry: hash (8B) + depth (4B) + score (4B) + move (16B) ≈ 32 bytes
        64MB → ~2 million positions
    """

    def __init__(self, size_mb: int = 64):
        """
        Initialize transposition table.

        Args:
            size_mb: Memory budget in megabytes (default 64MB)
        """
        bytes_per_entry = 32
        self.max_entries = (size_mb * 1024 * 1024) // bytes_per_entry
        self.table: Dict[int, Tuple[int, int, Optional[Tuple[int, int]]]] = {}

        # Statistics
        self.hits = 0
        self.probes = 0
        self.stores = 0
        self.evictions = 0

    def store(self, zobrist_hash: int, depth: int, score: int, bound_type: int,
              best_move_tuple: Optional[Tuple[int, int]] = None):
        """
        Store a position evaluation with bound type.

        Args:
            zobrist_hash: Zobrist hash from BitboardState
            depth: Search depth
            score: Evaluation score
            bound_type: TT_EXACT, TT_LOWER_BOUND, or TT_UPPER_BOUND
            best_move_tuple: Optional (from_sq, to_sq) for move ordering
        """
        # Depth-preferred replacement: only overwrite if deeper or new
        # Also prefer EXACT scores over bound scores at same depth
        if zobrist_hash in self.table:
            old_depth, _, old_bound, _ = self.table[zobrist_hash]
            if depth < old_depth:
                return  # Keep deeper result
            if depth == old_depth and old_bound == 0 and bound_type != 0:
                return  # Keep exact score over bound score at same depth

        # Evict if table full
        if len(self.table) >= self.max_entries and zobrist_hash not in self.table:
            # Remove oldest entry (first item in dict - FIFO)
            self.table.pop(next(iter(self.table)))
            self.evictions += 1

        # Store entry: (depth, score, bound_type, best_move)
        self.table[zobrist_hash] = (depth, score, bound_type, best_move_tuple)
        self.stores += 1

    def probe(self, zobrist_hash: int, depth: int, alpha: int, beta: int) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
        """
        Probe transposition table for cached evaluation with bound-aware logic.

        Args:
            zobrist_hash: Zobrist hash from BitboardState
            depth: Current search depth
            alpha: Current alpha bound
            beta: Current beta bound

        Returns:
            (score, best_move_tuple):
                - score: Cached score if usable (exact or causes cutoff), else None
                - best_move_tuple: Best move for ordering, or None
        """
        self.probes += 1

        if zobrist_hash not in self.table:
            return None, None  # Cache miss

        stored_depth, score, bound_type, best_move_tuple = self.table[zobrist_hash]

        # Only use score if searched at >= current depth
        if stored_depth >= depth:
            # TT_EXACT = 0, TT_LOWER_BOUND = 1, TT_UPPER_BOUND = 2
            if bound_type == 0:  # EXACT
                # Exact score - always usable
                self.hits += 1
                return score, best_move_tuple
            elif bound_type == 1:  # LOWER_BOUND
                # Score is at least 'score' - usable if score >= beta (causes cutoff)
                if score >= beta:
                    self.hits += 1
                    return score, best_move_tuple
            elif bound_type == 2:  # UPPER_BOUND
                # Score is at most 'score' - usable if score <= alpha (causes cutoff)
                if score <= alpha:
                    self.hits += 1
                    return score, best_move_tuple

        # Can't use score, but can still use move for ordering
        return None, best_move_tuple

    def clear(self):
        """Clear the transposition table."""
        self.table.clear()
        self.hits = 0
        self.probes = 0
        self.stores = 0
        self.evictions = 0

    def get_stats(self) -> Dict[str, any]:
        """Get TT statistics."""
        hit_rate = (self.hits / self.probes * 100) if self.probes > 0 else 0
        return {
            'size': len(self.table),
            'max_size': self.max_entries,
            'hits': self.hits,
            'probes': self.probes,
            'stores': self.stores,
            'evictions': self.evictions,
            'hit_rate': hit_rate
        }


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Transposition table (64MB default - adjust based on available memory)
TRANSPOSITION_TABLE = BitboardTranspositionTable(size_mb=64)

# Search configuration
MAX_DEPTH = 50          # Maximum search depth for iterative deepening
QUIESCENCE_MAX_DEPTH = 5  # Maximum quiescence search depth
TIME_LIMIT = 12.5       # Time limit in seconds (leave buffer for move conversion)

# Score constants
CHECKMATE_SCORE = 999_999
STALEMATE_SCORE = 0
DRAW_SCORE = 0

# FIX #16: Precompute infinity constants to avoid repeated float object creation
NEG_INF = float('-inf')
POS_INF = float('inf')

# TT bound type constants
TT_EXACT = 0        # Exact minimax value (no cutoff)
TT_LOWER_BOUND = 1  # Score is at least this good (beta cutoff occurred)
TT_UPPER_BOUND = 2  # Score is at most this good (failed to raise alpha)

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
LOGGING_ENABLED = False


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
        #see_score = static_exchange_eval(bb_state, move)
        see_score = 0

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
    # FIX #13: Use in-place sort to avoid creating new list
    moves.sort(key=lambda m: score_move(m, bb_state, tt_best_move), reverse=True)
    return moves


# ============================================================================
# QUIESCENCE SEARCH
# ============================================================================

def quiescence_search(bb_state: BitboardState, alpha: int, beta: int,
                     depth: int, max_depth: int, start_time: float,
                     is_maximizing: bool, player_is_white: bool) -> int:
    """
    Quiescence search to resolve tactical sequences (captures, plus checkmate detection).

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
        Static evaluation or best capture sequence score, or MATE/STALEMATE score
    """
    global stats
    stats['quiescence_nodes'] += 1

    # Timeout check
    if time.time() - start_time > TIME_LIMIT:
        return evaluate_bitboard(bb_state, player_is_white)

    # --- CHECK FOR TERMINAL STATES (MATE/STALEMATE) ---
    in_check = is_in_check(bb_state, bb_state.side_to_move == 0)

    # If in check, we must generate all moves to confirm if it's mate.
    if in_check:
        # Generate ALL moves (not just captures) to find escapes
        moves = generate_legal_moves(bb_state, captures_only=False)
        
        if not moves:
            # CHECKMATE DETECTED IN QUIESCENCE!
            # The current side_to_move is checkmated.
            # 
            # CRITICAL: Apply fixed 60,000 penalty to ALL quiescence mates to ensure they score
            # BELOW the early termination threshold (949,999), preventing false positives.
            #
            # Score examples:
            # - Quiescence mate (any depth): 999,999 - 60,000 = 939,999 (< 949,999 threshold)
            # - Main search mate at depth 4: -999,999 - 4,000 = -1,003,999 (uses ADDITION in minimax)
            # 
            # This ensures only genuine main-search forced mates trigger early termination.
            mate_score_base = CHECKMATE_SCORE - 60000

            # Determine return score based on whose turn it is
            if bb_state.side_to_move == 0:  # White is mated
                return -mate_score_base if player_is_white else mate_score_base
            else:  # Black is mated
                return mate_score_base if player_is_white else -mate_score_base
        
        # If in check but not mated, we must search the escape moves (which are now the 'captures' set)
        captures = order_moves(moves, bb_state)
        # Skip the stand-pat logic below, as standing still in check is illegal.

    else:
        # --- STANDARD QUIESCENCE LOGIC (Only if NOT in check) ---
        
        # 1. Stand-pat score: current position evaluation
        stand_pat = evaluate_bitboard(bb_state, player_is_white)

        # 2. Update alpha/beta bounds with stand-pat (can always choose not to capture)
        if is_maximizing:
            if stand_pat >= beta:
                return beta
            if stand_pat > alpha:
                alpha = stand_pat
            best_score = stand_pat # Initialize best score
        else:
            if stand_pat <= alpha:
                return alpha
            if stand_pat < beta:
                beta = stand_pat
            best_score = stand_pat # Initialize best score
            
        # 3. Generate captures only for search
        captures = generate_legal_moves(bb_state, captures_only=True)
        captures = order_moves(captures, bb_state)

    # Max depth reached: return static evaluation
    if depth >= max_depth:
        # If we reached max depth, return the most recent static evaluation (stand_pat or current evaluation)
        return evaluate_bitboard(bb_state, player_is_white)
    
    # If not in check, best_score is already initialized to stand_pat.
    # If in check, initialize best_score to alpha/beta bounds to handle the first move.
    if in_check:
        best_score = NEG_INF if is_maximizing else POS_INF


    # No captures/forcing moves available: quiet position
    if not captures and not in_check:
        return stand_pat # Only reached if not in check, and no captures

    # Search forcing moves (captures or check escapes)
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
                return best_score  # Beta cutoff - return actual best score found
        else:
            # Minimizing player
            best_score = min(best_score, score)
            if score < beta:
                beta = score
            if beta <= alpha:
                return best_score  # Alpha cutoff - return actual best score found

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

    # Save original alpha for bound type determination
    original_alpha = alpha

    # Timeout check
    if time.time() - start_time > TIME_LIMIT:
        return evaluate_bitboard(bb_state, player_is_white)

    # Probe transposition table with current alpha/beta bounds
    tt_score, tt_move = TRANSPOSITION_TABLE.probe(bb_state.zobrist_hash, depth, alpha, beta)
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
            
            # --- FIX START ---
            # Use depth directly. Higher depth = closer to root = faster mate = higher score.
            mate_bonus = depth * 1000  
            # --- FIX END ---

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
    # FIX #10: tt_move is already (from_sq, to_sq) as integers - use directly!
    tt_best_move_tuple = tt_move if tt_move else None

    # Order moves
    moves = order_moves(moves, bb_state, tt_best_move_tuple)

    best_score = NEG_INF if is_maximizing else POS_INF
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

    # Determine bound type for TT storage
    # For maximizing player:
    #   - best_score <= original_alpha: UPPER_BOUND (failed to raise alpha)
    #   - best_score >= beta: LOWER_BOUND (beta cutoff)
    #   - otherwise: EXACT
    # For minimizing player:
    #   - best_score >= original_beta would be UPPER_BOUND, but we don't track original_beta
    #   - best_score <= alpha: LOWER_BOUND (alpha cutoff)
    #   - We simplify: if cutoff happened, it's a bound; otherwise exact
    if is_maximizing:
        if best_score <= original_alpha:
            bound_type = TT_UPPER_BOUND
        elif best_score >= beta:
            bound_type = TT_LOWER_BOUND
        else:
            bound_type = TT_EXACT
    else:
        # For minimizing: if we improved beta, check if cutoff
        if best_score >= beta:
            bound_type = TT_UPPER_BOUND
        elif best_score <= alpha:
            bound_type = TT_LOWER_BOUND
        else:
            bound_type = TT_EXACT

    # Store in transposition table
    tt_move_tuple = (best_move.from_sq, best_move.to_sq) if best_move else None
    TRANSPOSITION_TABLE.store(bb_state.zobrist_hash, depth, best_score, bound_type, tt_move_tuple)
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
    best_score = NEG_INF

    # Get all legal moves
    moves = generate_legal_moves(bb_state)
    if not moves:
        return None  # No legal moves (checkmate or stalemate)

    # Check for immediate checkmate in 1 move (mate-in-1 detection)
    # If any move leaves opponent with no legal moves AND in check, play it immediately
    for move in moves:
        new_state = apply_move(bb_state, move)
        opponent_moves = generate_legal_moves(new_state)

        if not opponent_moves:
            # Opponent has no legal moves - check if it's checkmate or stalemate
            opponent_in_check = is_in_check(new_state, new_state.side_to_move == 0)

            if opponent_in_check:
                # CHECKMATE IN 1! Play this move immediately
                if LOGGING_ENABLED:
                    log_message(f"Immediate checkmate detected! Playing winning move.")
                print(f"Immediate checkmate detected! Playing winning move.")
                stats['depth_reached'] = 1
                return move
            # If not in check, it's stalemate - avoid this move, continue searching

    # Move stability tracking for early termination
    # Track last 4 best moves to detect when search has converged
    move_history = []  # List of (from_sq, to_sq) tuples from recent depths

    # Iterative deepening loop
    for depth in range(1, max_depth + 1):
        # Check if we have time for this depth
        elapsed = time.time() - start_time

        depth_start_time = time.time()
        depth_nodes_before = stats['nodes_searched']
        depth_best_move = None
        depth_best_score = NEG_INF

        # Removed verbose depth start logging - only show completion
        if LOGGING_ENABLED:
            log_message(f"\n=== Depth {depth} (Quiescence ON) ===")

        # Search all moves at current depth
        # Root is always maximizing (we're finding our best move)
        alpha = NEG_INF
        beta = POS_INF

        # Probe TT for move ordering hint (pass alpha/beta for bound-aware probing)
        tt_score, tt_move = TRANSPOSITION_TABLE.probe(bb_state.zobrist_hash, depth, alpha, beta)
        # FIX #10: tt_move is already (from_sq, to_sq) as integers - use directly!
        tt_best_move_tuple = tt_move if tt_move else None

        # Order moves for this depth
        ordered_moves = order_moves(moves, bb_state, tt_best_move_tuple)

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
                print(f"Timeout during depth {depth}, keeping previous depth's best move")
                return best_move if best_move else depth_best_move

        # Depth completed successfully - update best move
        best_move = depth_best_move
        best_score = depth_best_score
        stats['depth_reached'] = depth

        depth_time = time.time() - depth_start_time
        depth_nodes = stats['nodes_searched'] - depth_nodes_before

        # Simple depth completion logging
        if best_move:
            from_x, from_y = index_to_xy(best_move.from_sq)
            to_x, to_y = index_to_xy(best_move.to_sq)
            piece_names = ['Pawn', 'Knight', 'Bishop', 'Queen', 'King', 'Right']
            piece_name = piece_names[best_move.piece_type]
            move_str = f"{piece_name} ({from_x},{from_y}) to ({to_x},{to_y})"
        else:
            move_str = "None"
        print(f"Depth {depth} complete, in {depth_time:.2f}s, nodes visited: {depth_nodes}, best move: {move_str}, score: {best_score}")
        if LOGGING_ENABLED:
            log_message(f"Depth {depth} complete, in {depth_time:.2f}s, nodes visited: {depth_nodes}, best move: {move_str}, score: {best_score}")

        # Track move history for stability detection (only from depth 3 onwards)
        if best_move and depth >= 5:
            move_tuple = (best_move.from_sq, best_move.to_sq)
            move_history.append(move_tuple)

            # Check for move stability: same move for 4 consecutive depths (3,4,5,6 or 4,5,6,7, etc.)
            if len(move_history) >= 4:
                # Check if the last 4 moves are all identical
                last_4_moves = move_history[-4:]
                if all(m == last_4_moves[0] for m in last_4_moves):
                    if LOGGING_ENABLED:
                        log_message(f"Move stability detected at depth {depth}: same move for 4 consecutive depths (starting from depth {depth-3}), stopping search")
                    print(f"Move stability detected at depth {depth}: same move for 4 consecutive depths (starting from depth {depth-3}), stopping search")
                    break

        # Early termination: found forced checkmate FOR US (positive score only!)
        # 
        # Requirements:
        # 1. Minimum depth 6 to ensure sufficient search (not just quiescence discoveries)
        # 2. Score >= 949,999 (only main-search mates, not quiescence mates @ 939,999)
        # 3. Positive score only (we're winning, not losing)
        #
        # Score ranges:
        # - Main search mate: 999,999 to ~1,050,000 (uses ADDITION of mate_bonus)
        # - Quiescence mate: 939,999 (fixed penalty, below threshold)
        # - Normal position: -50,000 to +50,000 (typical eval range)
        if depth >= 6 and best_score >= CHECKMATE_SCORE - 50000:
            if LOGGING_ENABLED:
                log_message(f"Forced checkmate found at depth {depth} (score: {best_score}), stopping search")
            print(f"Forced checkmate found at depth {depth} (score: {best_score}), stopping search")
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

    TRANSPOSITION_TABLE.clear()

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
    
    # Get TT statistics
    tt_stats = TRANSPOSITION_TABLE.get_stats()
    tt_hit_rate = tt_stats['hit_rate']

    # Print summary
    print(f"\n{'='*60}")
    print(f"Total nodes visited: {stats['nodes_searched']}")
    print(f"Max depth reached: {stats['depth_reached']}")
    print(f"TT cache hit rate: {tt_hit_rate:.1f}% ({tt_stats['hits']}/{tt_stats['probes']})")
    print(f"TT size: {tt_stats['size']}/{tt_stats['max_size']} entries")
    print(f"{'='*60}\n")
    
    if LOGGING_ENABLED:
        from_x, from_y = index_to_xy(best_move.from_sq)
        to_x, to_y = index_to_xy(best_move.to_sq)
        log_message(f"\nBest move: {piece} from ({from_x},{from_y}) to ({to_x},{to_y})")
        log_message(f"Search time: {search_time:.2f}s")
        log_message(f"Total nodes visited: {stats['nodes_searched']}")
        log_message(f"Max depth reached: {stats['depth_reached']}")
        log_message(f"TT cache hit rate: {tt_hit_rate:.1f}% ({tt_stats['hits']}/{tt_stats['probes']})")
        log_message(f"TT size: {tt_stats['size']}/{tt_stats['max_size']} entries")
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
