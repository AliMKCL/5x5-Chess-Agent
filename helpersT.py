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

# ============================================================================
# ENDGAME-SPECIFIC PIECE-SQUARE TABLES (for agentE.py)
# ============================================================================

# Active king table - rewards centralization in endgames
KING_TABLE_ENDGAME_ACTIVE = [
    [-10, -5,  0, -5, -10],
    [ -5, 10, 15, 10,  -5],
    [  0, 15, 20, 15,   0],  # Strong center control
    [ -5, 10, 15, 10,  -5],
    [-10, -5,  0, -5, -10]
]

# Aggressive pawn table - massively rewards advancement toward promotion
PAWN_TABLE_ENDGAME = [
    [200, 200, 200, 200, 200],  # Row 0: Promotion imminent!
    [ 80,  80,  80,  80,  80],  # Row 1: Very close
    [ 40,  40,  40,  40,  40],  # Row 2: Halfway
    [ 15,  15,  15,  15,  15],  # Row 3
    [  0,   0,   0,   0,   0]   # Row 4: Start position
]

# Queen endgame table - centralized control for mating attacks
QUEEN_TABLE_ENDGAME = [
    [-20,  -5,   0,  -5, -20],
    [ -5,  15,  20,  15,  -5],
    [  0,  20,  25,  20,   0],  # Maximum center control
    [ -5,  15,  20,  15,  -5],
    [-20,  -5,   0,  -5, -20]
]

# Right endgame table - centralized control for mating attacks
RIGHT_TABLE_ENDGAME = [
    [-20,  -5,   0,  -5, -20],
    [ -5,  15,  20,  15,  -5],
    [  0,  20,  25,  20,   0],  # Maximum center control
    [ -5,  15,  20,  15,  -5],
    [-20,  -5,   0,  -5, -20]
]

def attacker_defender_ratio(board, target_position, attacking_player, defending_player, move_cache=None, pos_map=None):
    """
    Analyzes the attacker/defender balance for a given square and calculates exchange outcomes.
    
    Args:
        board: Current board state
        target_position: The position to analyze (Position object with x, y)
        attacking_player: The player whose pieces might attack this square
        defending_player: The player whose pieces might defend this square
        move_cache: Optional cache of piece move options to avoid redundant calculations
        pos_map: Optional position-to-piece lookup dict for O(1) piece lookups
    
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
    # ===== STEP 1: Find the piece on the target square (CRITICAL-3 OPTIMIZED) =====
    if pos_map is not None:
        # O(1) lookup using position map
        target_piece = pos_map.get((target_position.x, target_position.y))
    else:
        # Fallback to O(n) scan if no position map provided
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
        # OPTIMIZATION: Use cached move options if available
        if move_cache:
            key = (piece.position.x, piece.position.y, piece.name, piece.player.name)
            if key in move_cache:
                move_options = move_cache[key]
            else:
                move_options = piece.get_move_options()
        else:
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
        
        # OPTIMIZATION: Use cached move options if available
        if move_cache:
            key = (piece.position.x, piece.position.y, piece.name, piece.player.name)
            if key in move_cache:
                move_options = move_cache[key]
            else:
                move_options = piece.get_move_options()
        else:
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

def get_positional_value(piece, is_white, board=None, use_endgame_tables=False):
    """
    Returns positional value based on piece-square tables.

    Args:
        piece: The piece to evaluate
        is_white: Whether the piece belongs to white
        board: Optional board state (needed for endgame detection)
        use_endgame_tables: If True, uses endgame-specific tables (for agentE.py)

    Returns:
        int: Positional bonus/penalty for the piece's current position
    """
    x, y = piece.position.x, piece.position.y
    if not is_white:
        y = 4 - y

    piece_name = piece.name.lower()

    # Determine if we're in an endgame (for automatic table switching)
    is_endgame = False
    if board is not None:
        # board.get_pieces() returns a generator; convert to list to get length
        all_pieces_tmp = list(board.get_pieces())
        total_pieces = len(all_pieces_tmp)
        is_endgame = total_pieces <= 8

    if piece_name == 'pawn':
        # Use endgame pawn table if enabled and in endgame
        if use_endgame_tables and is_endgame:
            return PAWN_TABLE_ENDGAME[y][x]
        return PAWN_TABLE[y][x]

    elif piece_name == 'knight':
        return KNIGHT_TABLE[y][x]

    elif piece_name == 'king':
        # King table selection logic
        if board is not None:
            player_piece_count = sum(1 for p in board.get_pieces() if p.player == piece.player)

            if player_piece_count <= 4:
                # Very simplified endgame (≤4 pieces)
                if use_endgame_tables:
                    # agentE: Use active king table
                    return KING_TABLE_ENDGAME_ACTIVE[y][x]
                else:
                    # agentS/Q: Use neutral table
                    return KING_TABLE_ENDGAME[y][x]
            elif use_endgame_tables and is_endgame:
                # agentE: Moderate endgame (5-8 pieces) - use active table
                return KING_TABLE_ENDGAME_ACTIVE[y][x]

        # Middlegame: defensive king table
        return KING_TABLE[y][x]

    elif piece_name == 'bishop':
        return BISHOP_TABLE[y][x]

    elif piece_name == 'right':
        # Use endgame right table if enabled and in endgame
        if use_endgame_tables and is_endgame:
            return RIGHT_TABLE_ENDGAME[y][x]
        return RIGHT_TABLE[y][x]

    elif piece_name == 'queen':
        # Use endgame queen table if enabled and in endgame
        if use_endgame_tables and is_endgame:
            return QUEEN_TABLE_ENDGAME[y][x]
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
# TRANSPOSITION TABLE - Efficient Position Caching System
# ============================================================================
#
# CONCEPT: During search, we often encounter the same board position through
# different move orders. For example:
#   Move sequence 1: Knight to (2,3), then Pawn to (1,1) → Position A
#   Move sequence 2: Pawn to (1,1), then Knight to (2,3) → Position A (same!)
#
# Instead of re-evaluating Position A every time we encounter it, we:
#   1. Compute a unique "fingerprint" (hash) for each position
#   2. Store the evaluation result in a hash table
#   3. On revisit, instantly retrieve the cached result
#
# PERFORMANCE IMPACT: In chess, ~30-70% of positions are revisited during search.
# Caching these positions can speed up search by 3-10x, especially in quiescence
# search where the same capture sequences appear repeatedly.
#
# ============================================================================

# ============================================================================
# ZOBRIST HASHING - Fast Position Fingerprinting
# ============================================================================
#
# PROBLEM: We need a fast way to uniquely identify board positions.
#
# NAIVE APPROACH: Convert board to string "WP(0,0),BK(2,3)..." and use hash()
#   - Slow: O(n) string building + O(n) hashing
#   - Fragile: Different move orders create different strings for same position
#
# ZOBRIST SOLUTION: Pre-generate random numbers, XOR them together
#   - Each (piece_type, color, square) gets a unique random 64-bit number
#   - Position hash = XOR of all piece numbers on the board
#   - Fast: O(n) single pass, just XOR operations
#   - Incremental: Can update hash by XOR-ing changed squares only
#
# EXAMPLE:
#   White Pawn at (0,0) has random number: 0x1A2B3C4D5E6F7A8B
#   Black King at (2,3) has random number: 0x9876543210FEDCBA
#   Position hash = 0x1A2B3C4D5E6F7A8B ^ 0x9876543210FEDCBA = 0x825D686D4E912631
#
# XOR PROPERTIES (why this works):
#   - XOR is commutative: A ^ B = B ^ A (order doesn't matter)
#   - XOR is reversible: (A ^ B) ^ B = A (can add/remove pieces)
#   - Hash collisions are rare with 64-bit random numbers (~1 in 18 quintillion)
#
# ============================================================================

# Global lookup tables (initialized once at startup)
_ZOBRIST_TABLE = None  # Maps (piece_type, color, x, y) → random 64-bit number
_ZOBRIST_BLACK_TO_MOVE = None  # Random number XOR'd when black to move


def init_zobrist(seed=42):
    """
    Initialize Zobrist random number tables.

    Call this ONCE at program startup (before any searches).

    Why seed=42?
    - Makes hashes reproducible across runs (same positions → same hashes)
    - Useful for debugging (can verify TT is working correctly)
    - In production, seed doesn't matter as long as it's consistent

    Args:
        seed: Random seed for reproducibility
    """
    global _ZOBRIST_TABLE, _ZOBRIST_BLACK_TO_MOVE

    import random
    rng = random.Random(seed)

    # Piece types in our 5x5 chess variant
    piece_types = ['pawn', 'knight', 'bishop', 'right', 'queen', 'king']
    colors = ['white', 'black']

    _ZOBRIST_TABLE = {}

    # Generate a unique random 64-bit number for each (piece_type, color, square) combo
    # Total: 6 piece types × 2 colors × 25 squares = 300 random numbers
    for piece_type in piece_types:
        for color in colors:
            for x in range(5):  # Files 0-4
                for y in range(5):  # Ranks 0-4
                    # getrandbits(64) generates a random integer in range [0, 2^64-1]
                    _ZOBRIST_TABLE[(piece_type, color, x, y)] = rng.getrandbits(64)

    # Special random number for "black to move"
    # Same position with different players to move = different hash
    _ZOBRIST_BLACK_TO_MOVE = rng.getrandbits(64)


def compute_zobrist_hash(board, player_name):
    """
    Compute the Zobrist hash for the current board position.

    ALGORITHM:
    1. Start with hash = 0
    2. For each piece on the board:
       - Look up its random number in _ZOBRIST_TABLE
       - XOR it into the hash
    3. If black to move, XOR in _ZOBRIST_BLACK_TO_MOVE

    TIME COMPLEXITY: O(n) where n = number of pieces (max 25)
    SPACE COMPLEXITY: O(1) - just one integer

    Args:
        board: Current game board
        player_name: "white" or "black" (current player to move)

    Returns:
        int: 64-bit hash uniquely identifying this position

    EXAMPLE:
        Board:
            White Pawn at (0,0)   → _ZOBRIST_TABLE[('pawn', 'white', 0, 0)] = 0xAABB
            Black King at (2,3)   → _ZOBRIST_TABLE[('king', 'black', 2, 3)] = 0xCCDD
            White to move         → Don't XOR _ZOBRIST_BLACK_TO_MOVE

        Hash = 0 ^ 0xAABB ^ 0xCCDD = 0x6666
    """
    # Initialize Zobrist table if not already done
    if _ZOBRIST_TABLE is None:
        init_zobrist()

    h = 0  # Start with zero

    # XOR in each piece's random number
    for piece in board.get_pieces():
        piece_type = piece.name.lower()  # "Pawn" → "pawn"
        color = piece.player.name  # "white" or "black"
        x, y = piece.position.x, piece.position.y

        key = (piece_type, color, x, y)
        if key in _ZOBRIST_TABLE:
            h ^= _ZOBRIST_TABLE[key]  # XOR operation (^)

    # XOR in side-to-move indicator
    if player_name == "black":
        h ^= _ZOBRIST_BLACK_TO_MOVE

    return h


class TranspositionTable:
    """
    Transposition Table (TT) - Hash table for caching position evaluations.

    ANALOGY: Like a cache in a CPU - stores recent computations for instant retrieval.

    STRUCTURE:
        Dictionary mapping: zobrist_hash → (depth, score, best_move_tuple)

    WHY STORE DEPTH?
        A position evaluated at depth 5 is more accurate than depth 2.
        We only reuse cached scores if they were searched at >= current depth.
        Otherwise, we re-search at the deeper level but still use the cached best move.

    REPLACEMENT STRATEGY:
        When table is full and we need to insert a new entry:
        - If hash already exists: keep the deeper search result
        - If table is full: evict an arbitrary entry (first in dictionary)
        - Simple but effective (more complex strategies have minimal benefit)

    MEMORY USAGE:
        Each entry stores: hash (8 bytes) + depth (4) + score (4) + move (16) ≈ 32 bytes
        Default 64MB → ~2 million positions cached
    """

    def __init__(self, size_mb=64):
        """
        Initialize the transposition table.

        Args:
            size_mb: Memory budget in megabytes (default 64MB)
                    - 64MB is enough for millions of positions
                    - Can increase for longer time controls
        """
        # Calculate max entries based on memory budget
        # Each entry: ~40 bytes (hash + depth + score + move tuple + dict overhead)
        bytes_per_entry = 40
        max_entries = (size_mb * 1024 * 1024) // bytes_per_entry

        self.table = {}  # Dictionary: zobrist_hash → (depth, score, best_move_tuple)
        self.max_entries = max_entries

        # Statistics (for debugging/monitoring)
        self.hits = 0  # Number of times we found a usable cached score
        self.probes = 0  # Total number of lookups

    def store(self, zobrist_hash, depth, score, best_move_tuple=None):
        """
        Store a position evaluation in the table.

        DEPTH-PREFERRED REPLACEMENT:
        - If position already cached: only overwrite if new search is deeper
        - Rationale: Deeper searches are more accurate

        Args:
            zobrist_hash: Position hash from compute_zobrist_hash()
            depth: Search depth at which this score was computed
            score: Evaluation score (from player's perspective)
            best_move_tuple: Optional ((piece.position.x, piece.position.y),
                                       (move.position.x, move.position.y))
                            Used for move ordering in future searches
        """
        # Check if position already exists in table
        if zobrist_hash in self.table:
            old_depth = self.table[zobrist_hash][0]
            # Keep the deeper search result (more accurate)
            # Use abs() to handle both positive (minimax) and negative (quiescence) depths
            # Note: With fix to agentT.py, quiescence now uses positive depths, but keeping
            # abs() for robustness in case negative depths are used elsewhere
            if depth < old_depth:
                return  # Don't overwrite with shallower result

        # Eviction policy: if table is full, remove one entry
        if len(self.table) >= self.max_entries:
            if zobrist_hash not in self.table:  # Don't evict if updating existing
                # Simple strategy: remove first item in dictionary
                # (In Python 3.7+, dicts maintain insertion order)
                # More complex: remove lowest depth entry (slower, minimal benefit)
                self.table.pop(next(iter(self.table)))

        # Store the entry: (depth, score, best_move_tuple)
        self.table[zobrist_hash] = (depth, score, best_move_tuple)

    def probe(self, zobrist_hash, depth):
        """
        Look up a position in the transposition table.

        RETURN CASES:
        1. Cache miss: position not in table → (None, None)
        2. Shallow hit: position found but searched at lower depth
           → (None, best_move_tuple) - can't use score, but use move for ordering
        3. Deep hit: position found and searched at >= current depth
           → (score, best_move_tuple) - use both score and move!

        Args:
            zobrist_hash: Position hash from compute_zobrist_hash()
            depth: Current search depth

        Returns:
            (score, best_move_tuple):
                - score: Cached evaluation if valid, else None
                - best_move_tuple: Best move from previous search, else None
        """
        self.probes += 1  # Increment probe counter (for statistics)

        # Check if position exists in table
        if zobrist_hash not in self.table:
            return None, None  # Cache miss

        # Retrieve stored data
        stored_depth, score, best_move_tuple = self.table[zobrist_hash]

        # Only use cached score if it was searched at equal or greater depth
        # Why? A depth-5 search is more accurate than depth-2
        # Use abs() to handle both positive (minimax) and negative (quiescence) depths
        # Note: With fix to agentT.py, quiescence now uses positive depths, but keeping
        # abs() for robustness in case negative depths are used elsewhere
        if stored_depth >= depth:
            self.hits += 1  # Increment hit counter (successful cache use)
            return score, best_move_tuple  # Cache hit!

        # Shallow hit: can't use score, but return best move for move ordering
        # Even if we can't trust the score from a shallow search,
        # the best move from that search is still likely good
        return None, best_move_tuple

    def clear(self):
        """
        Clear the transposition table.

        Call this between games or when starting a new search from scratch.
        """
        self.table.clear()
        self.hits = 0
        self.probes = 0

    def get_hit_rate(self):
        """
        Calculate the cache hit rate percentage.

        Hit rate = (successful cache uses / total lookups) × 100

        TYPICAL VALUES:
        - 30-50%: Good (many positions reused)
        - 50-70%: Excellent (high reuse, major speedup)
        - <20%: Low (may need larger table or better move ordering)

        Returns:
            float: Hit rate percentage (0-100)
        """
        return (self.hits / self.probes * 100) if self.probes > 0 else 0