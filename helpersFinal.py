"""
STATE FLAG BEFORE ADDING PINS !!!!!!!!!!!!!!!!!!!!!!!!!!
Bitboard Helper Functions for 5x5 Chess Variant

This module provides a complete bitboard implementation for efficient move generation
and position evaluation in the 5x5 chess variant.

BITBOARD REPRESENTATION:
- A 5x5 board has 25 squares, numbered 0-24
- Each square maps to a bit in a 32-bit integer
- Bit position = y * 5 + x (row-major ordering)
- Example: Square (2,3) → bit 17

COORDINATE SYSTEM:
  x: 0 1 2 3 4 (files/columns)
y:
0   0 1 2 3 4
1   5 6 7 8 9
2  10 11 12 13 14
3  15 16 17 18 19
4  20 21 22 23 24

PIECE TYPES (indexed 0-5):
0 = Pawn, 1 = Knight, 2 = Bishop, 3 = Queen, 4 = King, 5 = Right (custom piece)

COLOR:
0 = White, 1 = Black

BITBOARDS USED IN THIS MODULE:

1. PIECE POSITION BITBOARDS (12 total):
   - WP: White pawn positions
   - WN: White knight positions
   - WB: White bishop positions
   - WQ: White queen positions
   - WK: White king position
   - WR: White Right piece positions
   - BP: Black pawn positions
   - BN: Black knight positions
   - BB: Black bishop positions
   - BQ: Black queen positions
   - BK: Black king position
   - BR: Black Right piece positions

2. OCCUPANCY BITBOARDS (3 total):
   - occ_white: All white piece positions (WP | WN | WB | WQ | WK | WR)
   - occ_black: All black piece positions (BP | BN | BB | BQ | BK | BR)
   - occ_all: All occupied squares (occ_white | occ_black)

3. PRECOMPUTED ATTACK BITBOARDS (50 total):
   - KNIGHT_ATTACKS[25]: Knight attack patterns from each square (ignores occupancy)
   - KING_ATTACKS[25]: King attack patterns from each square (ignores occupancy)

4. DYNAMICALLY GENERATED ATTACK BITBOARDS:
   - bishop_attacks: Bishop attack bitboard (considering occupancy via ray-casting)
   - queen_attacks: Queen attack bitboard (rook + bishop moves, considering occupancy)
   - right_attacks: Right piece attack bitboard (rook + knight, considering occupancy)

5. MOVE GENERATION HELPER BITBOARDS:
   - own_occ: Current player's piece occupancy
   - opp_occ: Opponent's piece occupancy
   - attacks: Bitboard of valid destination squares for a piece (after filtering own pieces)

Bitboards represent everything in bits (1-0s). For example, a bb for white cells:
1 0 1 0 1
0 1 0 1 0
1 0 1 0 1
0 1 0 1 0
1 0 1 0 1

Since only 1 and 0s, we need different contexts of bitboard. 
For example, 1 with pawn locations, 1 with knight locations...

How utilized?
- For move generation:
    - Precomputed attack tables for all pieces, where at the start of the program 25 long 
      list of Bitboards are created, showing the possible movements of those pieces at each square.
    - Therefore move generation is a very fast lookup. Legality of the moves are handled later.

To analyze squares and captures etc. you can use bitwise operations accross Bitboards. 
For example if white bishops ^ black knight moves = 1 --> There is a capture.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
from functools import lru_cache

# ============================================================================
# PART 1: CORE INFRASTRUCTURE
# ============================================================================

# Type aliases for code clarity
Bitboard = int  # 32-bit integer representing set of squares
Square = int    # Square index [0-24]
PieceType = int # Piece type index [0-6]
Color = int     # 0=white, 1=black

# Piece type constants
PAWN = 0
KNIGHT = 1
BISHOP = 2
QUEEN = 3
KING = 4
RIGHT = 5  # Custom piece: moves like bishop OR knight

# Piece value constants (for evaluation)
PIECE_VALUES = [100, 330, 320, 900, 20000, 500]  # P, N, B, Q, K, Right

# --- Coordinate Mapping Functions ---

# OPTIMIZED: Inline-friendly single expression functions (Fix #5)
# These are called thousands of times per search, so they need to be as fast as possible
# Python will inline these automatically due to their simplicity

def square_index(x: int, y: int) -> Square:
    """Convert (x,y) to square index. Example: square_index(2,3) → 17"""
    return y * 5 + x

def index_to_xy(sq: Square) -> Tuple[int, int]:
    """Convert square index to (x,y). Example: index_to_xy(17) → (2,3)"""
    return (sq % 5, sq // 5)


# --- Bit Manipulation Helpers ---

def set_bit(bb: Bitboard, sq: Square) -> Bitboard:
    """Set the bit at square index sq (mark square as occupied)."""
    return bb | (1 << sq)


def clear_bit(bb: Bitboard, sq: Square) -> Bitboard:
    """Clear the bit at square index sq (mark square as empty)."""
    return bb & ~(1 << sq)


def test_bit(bb: Bitboard, sq: Square) -> bool:
    """Test if bit at square index sq is set (check if square is occupied)."""
    return (bb & (1 << sq)) != 0


def pop_lsb(bb: Bitboard) -> Square:
    """
    Extract the index of the least significant bit (LSB).

    Uses the bit manipulation trick:
    - lsb = bb & -bb  (isolates lowest set bit)
    - lsb.bit_length() - 1  (converts to index)

    Example:
        bb = 0b10100 (bits 2, 4 set)
        lsb = 0b00100 (bit 2 isolated)
        index = 2

    Returns:
        Square index of the LSB, or -1 if bitboard is empty
    """
    if bb == 0:
        return -1
    lsb = bb & -bb  # Isolate lowest set bit
    return lsb.bit_length() - 1


def iter_bits(bb: Bitboard):
    """
    Generator that yields square indices for all set bits in the bitboard.

    Uses LSB extraction to iterate efficiently:
    1. Extract lowest set bit
    2. Yield its index
    3. Clear that bit
    4. Repeat until bitboard is empty

    Example:
        bb = 0b10100 (bits 2, 4 set)
        yields: 2, then 4
    """
    while bb:
        lsb = bb & -bb  # Isolate lowest set bit
        idx = lsb.bit_length() - 1
        yield idx
        bb ^= lsb  # Remove this bit (XOR to toggle off)


def count_bits(bb: Bitboard) -> int:
    """
    Count number of set bits in bitboard.
    Uses Python's built-in bit_count() which maps to hardware POPCNT instruction.
    """
    return bb.bit_count()


# --- Bitboard State Structure ---

@dataclass
class BitboardState:
    """
    Immutable representation of board state using bitboards.

    Each piece type for each color has its own bitboard (12 total).
    Occupancy masks are derived from piece bitboards for fast lookups.

    Attributes:
        WP, WN, WB, WQ, WK, WR: White pieces (Pawn, Knight, Bishop, Queen, King, Right)
        BP, BN, BB, BQ, BK, BR: Black pieces
        occ_white: Bitboard of all white pieces (WP | WN | WB | ...)
        occ_black: Bitboard of all black pieces (BP | BN | BB | ...)
        occ_all: Bitboard of all pieces (occ_white | occ_black)
        side_to_move: 0=white, 1=black
        zobrist_hash: Hash of this position for transposition table
    """
    # White pieces
    WP: Bitboard  # White pawns
    WN: Bitboard  # White knights
    WB: Bitboard  # White bishops
    WQ: Bitboard  # White queens
    WK: Bitboard  # White king
    WR: Bitboard # White Rights (custom piece)

    # Black pieces
    BP: Bitboard  # Black pawns
    BN: Bitboard  # Black knights
    BB: Bitboard  # Black bishops
    BQ: Bitboard  # Black queens
    BK: Bitboard  # Black king
    BR: Bitboard # Black Rights (custom piece)

    # Derived occupancy masks
    occ_white: Bitboard  # All white pieces
    occ_black: Bitboard  # All black pieces
    occ_all: Bitboard    # All pieces (white | black)

    # Game state
    side_to_move: Color  # 0=white to move, 1=black to move
    zobrist_hash: int    # Zobrist hash for transposition table


# ============================================================================
# PART 2: ATTACK TABLE GENERATION (Precomputed)
# ============================================================================

# These tables are computed once at module import time for efficiency

# Knight move offsets (L-shaped moves)
KNIGHT_DELTAS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                 (1, -2), (1, 2), (2, -1), (2, 1)]

# King move offsets (8 adjacent squares)
KING_DELTAS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
               (0, 1), (1, -1), (1, 0), (1, 1)]


def _generate_knight_attacks() -> List[Bitboard]:
    """
    Precompute knight attack bitboards for all 25 squares.

    For each square, generate a bitboard showing all squares a knight
    can move to from that square (ignoring occupancy).

    Returns:
        List of 25 bitboards, indexed by square number
    """
    attacks = []
    for sq in range(25):
        # Inline index_to_xy for performance (Fix #5)
        x = sq % 5
        y = sq // 5
        attack_bb = 0

        # Try all 8 L-shaped knight moves
        for dx, dy in KNIGHT_DELTAS:
            nx, ny = x + dx, y + dy
            # Check if destination is on the board
            if 0 <= nx < 5 and 0 <= ny < 5:
                # Inline square_index for performance (Fix #5)
                dest_sq = ny * 5 + nx
                attack_bb |= (1 << dest_sq)  # Inline set_bit

        attacks.append(attack_bb)

    return attacks


def _generate_king_attacks() -> List[Bitboard]:
    """
    Precompute king attack bitboards for all 25 squares.

    For each square, generate a bitboard showing all squares a king
    can move to from that square (ignoring occupancy).

    Returns:
        List of 25 bitboards, indexed by square number
    """
    attacks = []
    for sq in range(25):
        # Inline index_to_xy for performance (Fix #5)
        x = sq % 5
        y = sq // 5
        attack_bb = 0

        # Try all 8 adjacent squares
        for dx, dy in KING_DELTAS:
            nx, ny = x + dx, y + dy
            # Check if destination is on the board
            if 0 <= nx < 5 and 0 <= ny < 5:
                # Inline square_index for performance (Fix #5)
                dest_sq = ny * 5 + nx
                attack_bb |= (1 << dest_sq)  # Inline set_bit

        attacks.append(attack_bb)

    return attacks


# Precompute attack tables (done once at module import)
KNIGHT_ATTACKS = _generate_knight_attacks()
KING_ATTACKS = _generate_king_attacks()


# --- Precompute Sliding Piece Ray Masks (Optimization Fix #6) ---
def _generate_rook_rays() -> List[Bitboard]:
    """
    Precompute rook ray masks for all 25 squares (ignoring occupancy).
    These are used as a quick mask before computing actual attacks.
    
    Returns:
        List of 25 bitboards, one for each square
    """
    rays = []
    for sq in range(25):
        x = sq % 5
        y = sq // 5
        attacks = 0
        
        # Cast rays in 4 directions without considering occupancy
        directions = ((0, 1), (0, -1), (1, 0), (-1, 0))
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while 0 <= nx < 5 and 0 <= ny < 5:
                dest_sq = ny * 5 + nx
                attacks |= (1 << dest_sq)
                nx += dx
                ny += dy
        
        rays.append(attacks)
    return rays


def _generate_bishop_rays() -> List[Bitboard]:
    """
    Precompute bishop ray masks for all 25 squares (ignoring occupancy).
    
    Returns:
        List of 25 bitboards, one for each square
    """
    rays = []
    for sq in range(25):
        x = sq % 5
        y = sq // 5
        attacks = 0
        
        # Cast rays in 4 diagonal directions
        directions = ((1, 1), (1, -1), (-1, 1), (-1, -1))
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while 0 <= nx < 5 and 0 <= ny < 5:
                dest_sq = ny * 5 + nx
                attacks |= (1 << dest_sq)
                nx += dx
                ny += dy
        
        rays.append(attacks)
    return rays


# Precompute ray masks for quick checks
ROOK_RAYS = _generate_rook_rays()
BISHOP_RAYS = _generate_bishop_rays()


# --- Sliding Piece Attacks (Rook, Bishop, Queen, Right) ---
# NOTE: There is no ROOK piece in this variant, but _get_rook_attacks() is a
#       helper function used to generate horizontal/vertical moves for Queen and Right

def _get_rook_attacks(sq: Square, occupancy: Bitboard) -> Bitboard:
    """
    Generate rook attack bitboard from square sq considering occupancy.
    
    OPTIMIZED: Uses precomputed ROOK_RAYS for early exit when no blockers present.

    A rook attacks along ranks (rows) and files (columns) until blocked.

    NOTE: This is a helper function - there is no rook piece in this variant.
          Used by Queen (rook+bishop) and Right (rook+knight) pieces.

    Args:
        sq: Source square
        occupancy: Bitboard of all occupied squares (blocks rays)

    Returns:
        Bitboard of all squares the rook can attack

    Algorithm:
        1. Check if precomputed ray has any blockers
        2. If no blockers, return full ray (fast path)
        3. Otherwise, cast rays per direction until hitting blocker (slow path)
    """
    # Fast path: if no occupancy along rays, return full ray mask
    ray_mask = ROOK_RAYS[sq]
    if not (ray_mask & occupancy):
        return ray_mask
    
    # Slow path: there are blockers, need to compute exact attacks
    x = sq % 5
    y = sq // 5
    attacks = 0

    # Direction vectors: (dx, dy)
    directions = ((0, 1), (0, -1), (1, 0), (-1, 0))  # up, down, right, left

    for dx, dy in directions:
        # Cast ray in this direction
        nx, ny = x + dx, y + dy
        while 0 <= nx < 5 and 0 <= ny < 5:
            dest_sq = ny * 5 + nx
            attacks |= (1 << dest_sq)

            # Stop if we hit an occupied square (include the blocker)
            if occupancy & (1 << dest_sq):
                break

            nx += dx
            ny += dy

    return attacks

def _get_bishop_attacks(sq: Square, occupancy: Bitboard) -> Bitboard:
    """
    Generate bishop attack bitboard from square sq considering occupancy.
    
    OPTIMIZED: Uses precomputed BISHOP_RAYS for early exit when no blockers present.

    A bishop attacks along diagonals until blocked.

    Args:
        sq: Source square
        occupancy: Bitboard of all occupied squares

    Returns:
        Bitboard of all squares the bishop can attack
        
    Algorithm:
        1. Check if precomputed ray has any blockers
        2. If no blockers, return full ray (fast path)
        3. Otherwise, cast rays per direction until hitting blocker (slow path)
    """
    # Fast path: if no occupancy along rays, return full ray mask
    ray_mask = BISHOP_RAYS[sq]
    if not (ray_mask & occupancy):
        return ray_mask
    
    # Slow path: there are blockers, need to compute exact attacks
    x = sq % 5
    y = sq // 5
    attacks = 0

    # Diagonal directions: (dx, dy)
    directions = ((1, 1), (1, -1), (-1, 1), (-1, -1))  # NE, SE, NW, SW

    for dx, dy in directions:
        # Cast ray in this direction
        nx, ny = x + dx, y + dy
        while 0 <= nx < 5 and 0 <= ny < 5:
            dest_sq = ny * 5 + nx
            attacks |= (1 << dest_sq)

            # Stop if we hit an occupied square
            if occupancy & (1 << dest_sq):
                break

            nx += dx
            ny += dy

    return attacks


def _get_queen_attacks(sq: Square, occupancy: Bitboard) -> Bitboard:
    """
    Generate queen attack bitboard (combines rook + bishop attacks).

    Queen moves in all 8 directions: horizontal, vertical, and diagonal.
    """
    return _get_rook_attacks(sq, occupancy) | _get_bishop_attacks(sq, occupancy)


def _get_right_attacks(sq: Square, occupancy: Bitboard) -> Bitboard:
    """
    Generate Right piece attack bitboard (combines rook + knight attacks).

    The "Right" piece is a custom piece that moves like a rook OR knight.
    It can both slide like a rook (horizontal/vertical) AND jump like a knight.
    """
    return _get_rook_attacks(sq, occupancy) | KNIGHT_ATTACKS[sq]


# ============================================================================
# PART 3: ZOBRIST HASHING
# ============================================================================

class ZobristHasher:
    """
    Zobrist hashing system for fast position fingerprinting.

    Precomputes random 64-bit numbers for each (piece_type, color, square) combination.
    Position hash = XOR of all piece keys + side-to-move key.

    Benefits:
    - O(1) incremental updates (just XOR changed squares)
    - Collision probability ~1 / 2^64 (negligible)
    - Enables transposition table lookups
    """

    def __init__(self, seed: int = 42):
        """
        Initialize Zobrist hash tables with deterministic random numbers.

        Args:
            seed: Random seed for reproducibility (same seed = same hashes)
        """
        rng = random.Random(seed)

        # Generate random 64-bit keys for each (piece_type, color, square) combination
        # 6 piece types × 2 colors × 25 squares = 300 keys
        self.piece_keys = {}
        for piece_type in range(6):  # 0-5: P, N, B, Q, K, Right
            for color in range(2):   # 0-1: white, black
                for sq in range(25): # 0-24: all squares
                    key = rng.getrandbits(64)
                    self.piece_keys[(piece_type, color, sq)] = key

        # Special key for "black to move"
        self.black_to_move_key = rng.getrandbits(64)

    def compute_hash(self, bb_state: BitboardState) -> int:
        """
        Compute full Zobrist hash from scratch for a bitboard state.

        Algorithm:
        1. Start with hash = 0
        2. XOR in key for each piece on the board
        3. XOR in black_to_move_key if black to move

        Args:
            bb_state: Bitboard representation of position

        Returns:
            64-bit Zobrist hash
        """
        h = 0

        # Get list of all piece bitboards in order
        piece_bitboards = [
            (bb_state.WP, PAWN, 0),   (bb_state.WN, KNIGHT, 0), (bb_state.WB, BISHOP, 0),
            (bb_state.WQ, QUEEN, 0),  (bb_state.WK, KING, 0),   (bb_state.WR, RIGHT, 0),
            (bb_state.BP, PAWN, 1),   (bb_state.BN, KNIGHT, 1), (bb_state.BB, BISHOP, 1),
            (bb_state.BQ, QUEEN, 1),  (bb_state.BK, KING, 1),   (bb_state.BR, RIGHT, 1)
        ]

        # XOR in key for each piece
        for bitboard, piece_type, color in piece_bitboards:
            for sq in iter_bits(bitboard):
                h ^= self.piece_keys[(piece_type, color, sq)]

        # XOR in side-to-move
        if bb_state.side_to_move == 1:  # Black to move
            h ^= self.black_to_move_key

        return h


# Global zobrist hasher instance (initialized once)
_ZOBRIST = ZobristHasher(seed=42)

# ============================================================================
# PART 4: CHECK DETECTION
# ============================================================================

def is_in_check(bb_state: BitboardState, check_white: bool) -> bool:
    """
    Determine if the specified side's king is in check.
    
    FIX #12: Now uses cached implementation for performance.
    
    Uses "reverse attack generation": From the king's position, generate attacks
    as if the king were each piece type, then check if opponent has that piece
    type on those squares.

    This is faster than generating all opponent moves!

    Args:
        bb_state: Current board state
        check_white: True to check if white king in check, False for black

    Returns:
        True if the king is in check, False otherwise

    Algorithm:
        1. Find king position
        2. Generate knight attacks from king → check if opponent knights/rights there
        3. Check king adjacency (kings can't be adjacent)
        4. Generate rook attacks from king → check if opponent queens/rights there
        5. Generate bishop attacks from king → check if opponent bishops/queens there
        6. Check pawn attacks (special case - asymmetric)
    """
    # FIX #12: Delegate to cached version with individual bitboards as args
    return _is_in_check_cached(
        check_white,
        bb_state.WK, bb_state.BK,
        bb_state.WP, bb_state.BP,
        bb_state.WN, bb_state.BN,
        bb_state.WB, bb_state.BB,
        bb_state.WQ, bb_state.BQ,
        bb_state.WR, bb_state.BR,
        bb_state.occ_all
    )


# FIX #12: Cached version of is_in_check for performance
@lru_cache(maxsize=10000)
def _is_in_check_cached(
    check_white: bool,
    WK: int, BK: int, WP: int, BP: int, WN: int, BN: int,
    WB: int, BB: int, WQ: int, BQ: int, WR: int, BR: int,
    occ_all: int
) -> bool:
    """
    Cached implementation of is_in_check using individual bitboards as cache key.
    All arguments are ints (bitboards) which are hashable for lru_cache.
    """
    if check_white:
        king_bb = WK
        opp_knights = BN
        opp_bishops = BB
        opp_queens = BQ
        opp_rights = BR
        opp_king = BK
        opp_pawns = BP
        king_is_white = True
    else:
        king_bb = BK
        opp_knights = WN
        opp_bishops = WB
        opp_queens = WQ
        opp_rights = WR
        opp_king = WK
        opp_pawns = WP
        king_is_white = False

    if king_bb == 0:
        return False

    king_sq = pop_lsb(king_bb)

    # 1. Check for knight/Right attacks
    knight_attacks = KNIGHT_ATTACKS[king_sq]
    if knight_attacks & (opp_knights | opp_rights):
        return True

    # 2. Check for king adjacency
    if KING_ATTACKS[king_sq] & opp_king:
        return True

    # 3. Check for rook/queen/Right attacks
    rook_attacks = _get_rook_attacks(king_sq, occ_all)
    if rook_attacks & (opp_queens | opp_rights):
        return True

    # 4. Check for bishop/queen attacks
    bishop_attacks = _get_bishop_attacks(king_sq, occ_all)
    if bishop_attacks & (opp_bishops | opp_queens):
        return True

    # 5. Check for pawn attacks
    kx, ky = index_to_xy(king_sq)

    if king_is_white:
        pawn_attack_y = ky - 1
    else:
        pawn_attack_y = ky + 1

    if 0 <= pawn_attack_y < 5:
        for pawn_attack_x in [kx - 1, kx + 1]:
            if 0 <= pawn_attack_x < 5:
                pawn_sq = square_index(pawn_attack_x, pawn_attack_y)
                if test_bit(opp_pawns, pawn_sq):
                    return True

    return False


def is_square_attacked(bb_state: BitboardState, sq: Square, by_white: bool) -> bool:
    """
    Determine if a square is attacked by the specified side.

    Similar to is_in_check but checks any square, not just king position.
    Uses reverse attack generation for efficiency.

    Args:
        bb_state: Current board state
        sq: Square index to check
        by_white: True if checking for white attacks, False for black

    Returns:
        True if the square is attacked by the specified side
    """
    occ = bb_state.occ_all

    if by_white:
        # Check if white attacks this square
        atk_knights = bb_state.WN
        atk_bishops = bb_state.WB
        atk_queens = bb_state.WQ
        atk_rights = bb_state.WR
        atk_king = bb_state.WK
        atk_pawns = bb_state.WP
        attacker_is_white = True
    else:
        # Check if black attacks this square
        atk_knights = bb_state.BN
        atk_bishops = bb_state.BB
        atk_queens = bb_state.BQ
        atk_rights = bb_state.BR
        atk_king = bb_state.BK
        atk_pawns = bb_state.BP
        attacker_is_white = False

    # 1. Check for knight/Right attacks
    if KNIGHT_ATTACKS[sq] & (atk_knights | atk_rights):
        return True

    # 2. Check for king attacks
    if KING_ATTACKS[sq] & atk_king:
        return True

    # 3. Check for rook/queen/Right attacks (sliding horizontal/vertical)
    rook_attacks = _get_rook_attacks(sq, occ)
    if rook_attacks & (atk_queens | atk_rights):
        return True

    # 4. Check for bishop/queen attacks (sliding diagonal)
    bishop_attacks = _get_bishop_attacks(sq, occ)
    if bishop_attacks & (atk_bishops | atk_queens):
        return True

    # 5. Check for pawn attacks (SPECIAL CASE - asymmetric)
    x, y = index_to_xy(sq)

    if attacker_is_white:
        # White pawns attack diagonally upward (from white's perspective)
        # White pawn at (x±1, y+1) would attack square at (x, y)
        pawn_y = y + 1
    else:
        # Black pawns attack diagonally downward (from white's perspective)
        # Black pawn at (x±1, y-1) would attack square at (x, y)
        pawn_y = y - 1

    if 0 <= pawn_y < 5:
        for pawn_x in [x - 1, x + 1]:
            if 0 <= pawn_x < 5:
                pawn_sq = square_index(pawn_x, pawn_y)
                if test_bit(atk_pawns, pawn_sq):
                    return True

    return False


def static_exchange_eval(bb_state: BitboardState, move: 'BBMove') -> int:
    """
    Static Exchange Evaluation (SEE) for a capture move.

    Estimates the material outcome of a capture sequence on the target square.
    Used for move ordering to prioritize safe/profitable captures.

    Args:
        bb_state: Current board state
        move: The capture move to evaluate (must be a capture, move.captured_type != -1)

    Returns:
        Net material outcome in centipawns (positive = good, negative = bad)

    Algorithm:
        1. Find all pieces attacking the target square for both sides
        2. Simulate exchange sequence using cheapest pieces first
        3. Return net material gain/loss

    Example:
        Pawn captures Queen defended by Pawn:
        1. We capture Queen: +900
        2. They recapture with Pawn: -100
        Net: +800 (excellent trade!)
    """
    to_sq = move.to_sq
    attacker_is_white = (bb_state.side_to_move == 0)

    # Get occupancy for attack generation
    occ = bb_state.occ_all

    # Collect attackers for both sides (by piece type and value)
    # We'll track lists of (piece_type, value) sorted by value
    white_attackers = []
    black_attackers = []

    # Helper to add attackers of a specific type
    def add_attackers_of_type(piece_bb, piece_type, is_white, attack_bb):
        """Add all pieces of given type that attack to_sq"""
        for sq in iter_bits(piece_bb & attack_bb):
            if is_white:
                white_attackers.append((piece_type, PIECE_VALUES[piece_type], sq))
            else:
                black_attackers.append((piece_type, PIECE_VALUES[piece_type], sq))

    # 1. Find all knight attackers
    knight_attacks_to = KNIGHT_ATTACKS[to_sq]
    add_attackers_of_type(bb_state.WN, KNIGHT, True, knight_attacks_to)
    add_attackers_of_type(bb_state.BN, KNIGHT, False, knight_attacks_to)
    # Rights also attack like knights
    add_attackers_of_type(bb_state.WR, RIGHT, True, knight_attacks_to)
    add_attackers_of_type(bb_state.BR, RIGHT, False, knight_attacks_to)

    # 2. Find all king attackers
    king_attacks_to = KING_ATTACKS[to_sq]
    add_attackers_of_type(bb_state.WK, KING, True, king_attacks_to)
    add_attackers_of_type(bb_state.BK, KING, False, king_attacks_to)

    # 3. Find all sliding piece attackers (bishop, rook, queen, right)
    rook_attacks_to = _get_rook_attacks(to_sq, occ)
    bishop_attacks_to = _get_bishop_attacks(to_sq, occ)

    # Bishops attack diagonally
    add_attackers_of_type(bb_state.WB, BISHOP, True, bishop_attacks_to)
    add_attackers_of_type(bb_state.BB, BISHOP, False, bishop_attacks_to)

    # Queens attack both ways
    queen_attacks_to = rook_attacks_to | bishop_attacks_to
    add_attackers_of_type(bb_state.WQ, QUEEN, True, queen_attacks_to)
    add_attackers_of_type(bb_state.BQ, QUEEN, False, queen_attacks_to)

    # Rights attack rook-ways (already added knight-ways above)
    add_attackers_of_type(bb_state.WR, RIGHT, True, rook_attacks_to)
    add_attackers_of_type(bb_state.BR, RIGHT, False, rook_attacks_to)

    # 4. Find pawn attackers (special case - asymmetric)
    tx, ty = index_to_xy(to_sq)

    # White pawns attack from below (y+1)
    if ty + 1 < 5:
        for px in [tx - 1, tx + 1]:
            if 0 <= px < 5:
                pawn_sq = square_index(px, ty + 1)
                if test_bit(bb_state.WP, pawn_sq):
                    white_attackers.append((PAWN, PIECE_VALUES[PAWN], pawn_sq))

    # Black pawns attack from above (y-1)
    if ty - 1 >= 0:
        for px in [tx - 1, tx + 1]:
            if 0 <= px < 5:
                pawn_sq = square_index(px, ty - 1)
                if test_bit(bb_state.BP, pawn_sq):
                    black_attackers.append((PAWN, PIECE_VALUES[PAWN], pawn_sq))

    # Sort by value (cheapest first for optimal exchange sequence)
    white_attackers.sort(key=lambda x: x[1])
    black_attackers.sort(key=lambda x: x[1])

    # Remove the initial attacker (move.from_sq) from appropriate list
    # since it's already making the first capture
    from_sq = move.from_sq
    if attacker_is_white:
        white_attackers = [(pt, pv, sq) for pt, pv, sq in white_attackers if sq != from_sq]
    else:
        black_attackers = [(pt, pv, sq) for pt, pv, sq in black_attackers if sq != from_sq]

    # Simulate exchange sequence
    gain = [PIECE_VALUES[move.captured_type]]  # Initial capture value
    current_attacker_value = PIECE_VALUES[move.piece_type]  # Value of piece making initial capture

    # Alternate between sides, using cheapest piece each time
    white_turn = not attacker_is_white  # After our move, opponent responds

    while True:
        if white_turn:
            if not white_attackers:
                break  # No more white attackers
            piece_type, piece_value, sq = white_attackers.pop(0)  # Use cheapest
            gain.append(current_attacker_value)  # Capture previous attacker
            current_attacker_value = piece_value
            white_turn = False
        else:
            if not black_attackers:
                break  # No more black attackers
            piece_type, piece_value, sq = black_attackers.pop(0)  # Use cheapest
            gain.append(current_attacker_value)  # Capture previous attacker
            current_attacker_value = piece_value
            white_turn = True

    # Now compute final score using minimax on gain array
    # gain[0] = initial capture, gain[1] = first recapture, etc.
    # We want to maximize: gain[0] - gain[1] + gain[2] - gain[3] + ...

    # Work backwards through gain array
    score = 0
    for i in range(len(gain) - 1, -1, -1):
        if i % 2 == 0:  # Even index = our gain
            score = max(score, gain[i] - score)
        else:  # Odd index = opponent's gain
            score = min(score, gain[i] - score)

    # Return from current side's perspective
    if not attacker_is_white:
        score = -score

    return score


# ============================================================================
# PART 5: MOVE REPRESENTATION & GENERATION
# ============================================================================

@dataclass
class BBMove:
    """
    Compact move representation for bitboard engine.

    Attributes:
        from_sq: Source square index [0-24]
        to_sq: Destination square index [0-24]
        piece_type: Moving piece type [0-5]
        captured_type: Captured piece type, or -1 if no capture
        promo: Promotion piece type (3=Queen), or 0 if no promotion
    """
    from_sq: Square
    to_sq: Square
    piece_type: PieceType
    captured_type: int  # -1 if no capture
    promo: int  # 0 if no promotion, 4 if promote to queen


def generate_legal_moves(bb_state: BitboardState, captures_only: bool = False) -> List[BBMove]:
    """
    Generate all legal moves for the side to move.

    Algorithm:
    1. Generate pseudo-legal moves (moves that look legal ignoring check)
    2. For each move:
       a. Apply the move to get child state
       b. Check if our king is in check in child state
       c. If not in check, move is legal → add to list

    Args:
        bb_state: Current board state
        captures_only: If True, only generate captures (for quiescence search)

    Returns:
        List of legal BBMove objects
    """
    legal_moves = []
    stm_white = (bb_state.side_to_move == 0)

    # Get piece bitboards for side to move
    if stm_white:
        own_pieces = [
            (bb_state.WP, PAWN), (bb_state.WN, KNIGHT), (bb_state.WB, BISHOP),
            (bb_state.WQ, QUEEN), (bb_state.WK, KING), (bb_state.WR, RIGHT)
        ]
        own_occ = bb_state.occ_white
        opp_occ = bb_state.occ_black
    else:
        own_pieces = [
            (bb_state.BP, PAWN), (bb_state.BN, KNIGHT), (bb_state.BB, BISHOP),
            (bb_state.BQ, QUEEN), (bb_state.BK, KING), (bb_state.BR, RIGHT)
        ]
        own_occ = bb_state.occ_black
        opp_occ = bb_state.occ_white

    occ_all = bb_state.occ_all

    # --- Generate moves for each piece type ---

    for piece_bb, piece_type in own_pieces:
        if piece_bb == 0:
            continue  # No pieces of this type

        if piece_type == PAWN:
            # Pawn moves are special (direction-dependent, promotion)
            legal_moves.extend(_generate_pawn_moves(
                piece_bb, stm_white, own_occ, opp_occ, occ_all, bb_state, captures_only
            ))

        elif piece_type == KNIGHT:
            # Knight moves from precomputed table
            for from_sq in iter_bits(piece_bb):
                attacks = KNIGHT_ATTACKS[from_sq] & ~own_occ  # Can't capture own pieces
                if captures_only:
                    attacks &= opp_occ  # Only captures

                for to_sq in iter_bits(attacks):
                    captured = _get_captured_piece_type(bb_state, to_sq, not stm_white)
                    move = BBMove(from_sq, to_sq, KNIGHT, captured, 0)

                    # Test legality: apply move and check if king in check
                    child = apply_move(bb_state, move)
                    if not is_in_check(child, stm_white):
                        legal_moves.append(move)

        elif piece_type == BISHOP:
            # Bishop sliding moves
            for from_sq in iter_bits(piece_bb):
                attacks = _get_bishop_attacks(from_sq, occ_all) & ~own_occ
                if captures_only:
                    attacks &= opp_occ

                for to_sq in iter_bits(attacks):
                    captured = _get_captured_piece_type(bb_state, to_sq, not stm_white)
                    move = BBMove(from_sq, to_sq, BISHOP, captured, 0)

                    child = apply_move(bb_state, move)
                    if not is_in_check(child, stm_white):
                        legal_moves.append(move)

        elif piece_type == QUEEN:
            # Queen sliding moves (bishop only)
            for from_sq in iter_bits(piece_bb):
                attacks = _get_queen_attacks(from_sq, occ_all) & ~own_occ
                if captures_only:
                    attacks &= opp_occ

                for to_sq in iter_bits(attacks):
                    captured = _get_captured_piece_type(bb_state, to_sq, not stm_white)
                    move = BBMove(from_sq, to_sq, QUEEN, captured, 0)

                    child = apply_move(bb_state, move)
                    if not is_in_check(child, stm_white):
                        legal_moves.append(move)

        elif piece_type == KING:
            # King moves from precomputed table
            for from_sq in iter_bits(piece_bb):
                attacks = KING_ATTACKS[from_sq] & ~own_occ
                if captures_only:
                    attacks &= opp_occ

                for to_sq in iter_bits(attacks):
                    captured = _get_captured_piece_type(bb_state, to_sq, not stm_white)
                    move = BBMove(from_sq, to_sq, KING, captured, 0)

                    child = apply_move(bb_state, move)
                    if not is_in_check(child, stm_white):
                        legal_moves.append(move)

        elif piece_type == RIGHT:
            # Right piece (bishop + knight hybrid)
            for from_sq in iter_bits(piece_bb):
                attacks = _get_right_attacks(from_sq, occ_all) & ~own_occ
                if captures_only:
                    attacks &= opp_occ

                for to_sq in iter_bits(attacks):
                    captured = _get_captured_piece_type(bb_state, to_sq, not stm_white)
                    move = BBMove(from_sq, to_sq, RIGHT, captured, 0)

                    child = apply_move(bb_state, move)
                    if not is_in_check(child, stm_white):
                        legal_moves.append(move)

    return legal_moves


def _generate_pawn_moves(pawn_bb: Bitboard, is_white: bool, own_occ: Bitboard,
                         opp_occ: Bitboard, occ_all: Bitboard,
                         bb_state: BitboardState, captures_only: bool) -> List[BBMove]:
    """
    Generate pawn moves (special case due to direction, promotion, captures).

    Pawns:
    - Move forward one square if unoccupied
    - Capture diagonally forward
    - Promote to queen when reaching back rank (y=0 for white, y=4 for black)

    Args:
        pawn_bb: Bitboard of pawns to generate moves for
        is_white: True if white pawns, False if black
        own_occ, opp_occ, occ_all: Occupancy masks
        bb_state: Full board state (for legality checking)
        captures_only: If True, only generate captures

    Returns:
        List of legal pawn moves
    """
    moves = []
    dir_y = -1 if is_white else 1  # White pawns move up (y decreases), black down

    for from_sq in iter_bits(pawn_bb):
        x, y = index_to_xy(from_sq)

        # --- Forward move ---
        if not captures_only:
            to_y = y + dir_y
            if 0 <= to_y < 5:
                to_sq = square_index(x, to_y)
                # Can only move if square is unoccupied
                if not test_bit(occ_all, to_sq):
                    # Check for promotion (reaching back rank)
                    promo = QUEEN if (to_y == 0 or to_y == 4) else 0
                    move = BBMove(from_sq, to_sq, PAWN, -1, promo)

                    # Test legality
                    child = apply_move(bb_state, move)
                    if not is_in_check(child, is_white):
                        moves.append(move)

        # --- Diagonal captures ---
        to_y = y + dir_y
        if 0 <= to_y < 5:
            for to_x in [x - 1, x + 1]:  # Left and right diagonals
                if 0 <= to_x < 5:
                    to_sq = square_index(to_x, to_y)
                    # Can only capture if opponent piece present
                    if test_bit(opp_occ, to_sq):
                        captured = _get_captured_piece_type(bb_state, to_sq, not is_white)
                        promo = QUEEN if (to_y == 0 or to_y == 4) else 0
                        move = BBMove(from_sq, to_sq, PAWN, captured, promo)

                        # Test legality
                        child = apply_move(bb_state, move)
                        if not is_in_check(child, is_white):
                            moves.append(move)

    return moves


def _get_captured_piece_type(bb_state: BitboardState, sq: Square, is_white: bool) -> int:
    """
    Determine what piece type (if any) is on square sq for the given color.

    Args:
        bb_state: Board state
        sq: Square to check
        is_white: True to check white pieces, False for black

    Returns:
        Piece type [0-6] if piece found, -1 if square empty
    """
    mask = 1 << sq

    if is_white:
        pieces = [
            (bb_state.WP, PAWN), (bb_state.WN, KNIGHT), (bb_state.WB, BISHOP),
            (bb_state.WQ, QUEEN), (bb_state.WK, KING), (bb_state.WR, RIGHT)
        ]
    else:
        pieces = [
            (bb_state.BP, PAWN), (bb_state.BN, KNIGHT), (bb_state.BB, BISHOP),
            (bb_state.BQ, QUEEN), (bb_state.BK, KING), (bb_state.BR, RIGHT)
        ]

    for piece_bb, piece_type in pieces:
        if piece_bb & mask:
            return piece_type

    return -1  # No piece on this square


# ============================================================================
# PART 6: MOVE APPLICATION
# ============================================================================

def apply_move(bb_state: BitboardState, move: BBMove) -> BitboardState:
    """
    Apply a move to a bitboard state, returning a NEW state (immutable).

    Algorithm:
    1. Copy all piece bitboards to a list
    2. Remove piece from origin square
    3. If capture, remove captured piece from destination
    4. Add piece to destination (or promoted piece if promotion)
    5. Rebuild occupancy masks
    6. Toggle side to move
    7. Compute new Zobrist hash incrementally

    Args:
        bb_state: Current state
        move: Move to apply

    Returns:
        New BitboardState after move is applied
    """
    # Copy piece bitboards to list for modification
    pieces = [
        bb_state.WP, bb_state.WN, bb_state.WB, bb_state.WQ, bb_state.WK, bb_state.WR,
        bb_state.BP, bb_state.BN, bb_state.BB, bb_state.BQ, bb_state.BK, bb_state.BR
    ]

    stm_white = (bb_state.side_to_move == 0)

    # Calculate piece indices (white=0-5, black=6-11)
    color_offset = 0 if stm_white else 6
    piece_idx = color_offset + move.piece_type

    # Start with current hash
    new_hash = bb_state.zobrist_hash

    # 1. Remove piece from origin square
    from_mask = 1 << move.from_sq
    pieces[piece_idx] &= ~from_mask
    # Update hash: XOR out piece at origin
    new_hash ^= _ZOBRIST.piece_keys[(move.piece_type, bb_state.side_to_move, move.from_sq)]

    # 2. Handle capture (remove opponent piece from destination)
    if move.captured_type != -1:
        opp_color_offset = 6 if stm_white else 0
        captured_idx = opp_color_offset + move.captured_type
        to_mask = 1 << move.to_sq
        pieces[captured_idx] &= ~to_mask
        # Update hash: XOR out captured piece
        opp_color = 1 if stm_white else 0
        new_hash ^= _ZOBRIST.piece_keys[(move.captured_type, opp_color, move.to_sq)]

    # 3. Add piece to destination (or promoted piece)
    to_mask = 1 << move.to_sq
    if move.promo != 0:
        # Promotion: add queen instead of pawn
        queen_idx = color_offset + QUEEN
        pieces[queen_idx] |= to_mask
        # Update hash: XOR in queen at destination
        new_hash ^= _ZOBRIST.piece_keys[(QUEEN, bb_state.side_to_move, move.to_sq)]
    else:
        # Normal move: add same piece type
        pieces[piece_idx] |= to_mask
        # Update hash: XOR in piece at destination
        new_hash ^= _ZOBRIST.piece_keys[(move.piece_type, bb_state.side_to_move, move.to_sq)]

    # 4. Rebuild occupancy masks
    new_occ_white = pieces[0] | pieces[1] | pieces[2] | pieces[3] | pieces[4] | pieces[5]
    new_occ_black = pieces[6] | pieces[7] | pieces[8] | pieces[9] | pieces[10] | pieces[11]
    new_occ_all = new_occ_white | new_occ_black

    # 5. Toggle side to move
    new_side_to_move = 1 - bb_state.side_to_move
    # Update hash: toggle side-to-move (XOR is self-inverse, so this flips the bit)
    new_hash ^= _ZOBRIST.black_to_move_key

    # 6. Create new state
    return BitboardState(
        WP=pieces[0], WN=pieces[1], WB=pieces[2], WQ=pieces[3], WK=pieces[4], WR=pieces[5],
        BP=pieces[6], BN=pieces[7], BB=pieces[8], BQ=pieces[9], BK=pieces[10], BR=pieces[11],
        occ_white=new_occ_white,
        occ_black=new_occ_black,
        occ_all=new_occ_all,
        side_to_move=new_side_to_move,
        zobrist_hash=new_hash
    )


# ============================================================================
# PART 7: BOARD CONVERSION (Bridge from chessmaker framework)
# ============================================================================

def board_to_bitboard(board, player) -> BitboardState:
    """
    Convert chessmaker Board object to BitboardState.

    This is the bridge function that allows us to use bitboards
    with the existing game framework.

    Args:
        board: chessmaker Board object
        player: Current player object

    Returns:
        BitboardState representation
    """
    # Initialize all bitboards to 0
    piece_bitboards = {
        'WP': 0, 'WN': 0, 'WB': 0, 'WQ': 0, 'WK': 0, 'WR': 0,
        'BP': 0, 'BN': 0, 'BB': 0, 'BQ': 0, 'BK': 0, 'BR': 0
    }

    # Iterate all pieces on board
    for piece in board.get_pieces():
        piece_name = piece.name.lower()
        color_prefix = 'W' if piece.player.name == "white" else 'B'

        # Map piece name to bitboard key
        if piece_name == 'pawn':
            key = color_prefix + 'P'
        elif piece_name == 'knight':
            key = color_prefix + 'N'
        elif piece_name == 'bishop':
            key = color_prefix + 'B'
        elif piece_name == 'queen':
            key = color_prefix + 'Q'
        elif piece_name == 'king':
            key = color_prefix + 'K'
        elif piece_name == 'right':
            key = color_prefix + 'R'
        else:
            continue  # Unknown piece type

        # Set bit for this piece's position
        sq = square_index(piece.position.x, piece.position.y)
        piece_bitboards[key] = set_bit(piece_bitboards[key], sq)

    # Build occupancy masks
    occ_white = (piece_bitboards['WP'] | piece_bitboards['WN'] | piece_bitboards['WB'] |
                 piece_bitboards['WQ'] | piece_bitboards['WK'] | piece_bitboards['WR'])
    occ_black = (piece_bitboards['BP'] | piece_bitboards['BN'] | piece_bitboards['BB'] |
                 piece_bitboards['BQ'] | piece_bitboards['BK'] | piece_bitboards['BR'])
    occ_all = occ_white | occ_black

    # Determine side to move
    side_to_move = 0 if player.name == "white" else 1

    # Create state
    bb_state = BitboardState(
        WP=piece_bitboards['WP'], WN=piece_bitboards['WN'], WB=piece_bitboards['WB'],
        WQ=piece_bitboards['WQ'], WK=piece_bitboards['WK'], WR=piece_bitboards['WR'],
        BP=piece_bitboards['BP'], BN=piece_bitboards['BN'], BB=piece_bitboards['BB'],
        BQ=piece_bitboards['BQ'], BK=piece_bitboards['BK'], BR=piece_bitboards['BR'],
        occ_white=occ_white,
        occ_black=occ_black,
        occ_all=occ_all,
        side_to_move=side_to_move,
        zobrist_hash=0  # Will be computed below
    )

    # Compute Zobrist hash
    bb_state = BitboardState(
        WP=bb_state.WP, WN=bb_state.WN, WB=bb_state.WB, WQ=bb_state.WQ,
        WK=bb_state.WK, WR=bb_state.WR,
        BP=bb_state.BP, BN=bb_state.BN, BB=bb_state.BB, BQ=bb_state.BQ,
        BK=bb_state.BK, BR=bb_state.BR,
        occ_white=bb_state.occ_white,
        occ_black=bb_state.occ_black,
        occ_all=bb_state.occ_all,
        side_to_move=bb_state.side_to_move,
        zobrist_hash=_ZOBRIST.compute_hash(bb_state)
    )

    return bb_state


# ============================================================================
# TESTING / DEBUGGING UTILITIES
# ============================================================================

def print_bitboard(bb: Bitboard, label: str = ""):
    """
    Pretty-print a bitboard for debugging.

    Args:
        bb: Bitboard to print
        label: Optional label to print before the board
    """
    if label:
        print(f"\n{label}:")

    print("  0 1 2 3 4")
    for y in range(5):
        print(f"{y} ", end="")
        for x in range(5):
            sq = square_index(x, y)
            print("X " if test_bit(bb, sq) else ". ", end="")
        print()
    print()


def print_board_state(bb_state: BitboardState):
    """
    Pretty-print entire board state for debugging.

    Shows piece placement similar to chess notation.
    """
    print("\nBoard State:")
    print("  0 1 2 3 4")

    for y in range(5):
        print(f"{y} ", end="")
        for x in range(5):
            sq = square_index(x, y)
            piece_char = '.'

            # Check each piece type
            if test_bit(bb_state.WP, sq): piece_char = 'P'
            elif test_bit(bb_state.WN, sq): piece_char = 'N'
            elif test_bit(bb_state.WB, sq): piece_char = 'B'
            elif test_bit(bb_state.WQ, sq): piece_char = 'Q'
            elif test_bit(bb_state.WK, sq): piece_char = 'K'
            elif test_bit(bb_state.WR, sq): piece_char = 'R'  # R for Right (custom piece)
            elif test_bit(bb_state.BP, sq): piece_char = 'p'
            elif test_bit(bb_state.BN, sq): piece_char = 'n'
            elif test_bit(bb_state.BB, sq): piece_char = 'b'
            elif test_bit(bb_state.BQ, sq): piece_char = 'q'
            elif test_bit(bb_state.BK, sq): piece_char = 'k'
            elif test_bit(bb_state.BR, sq): piece_char = 'r'

            print(f"{piece_char} ", end="")
        print()

    print(f"Side to move: {'White' if bb_state.side_to_move == 0 else 'Black'}")
    print(f"Zobrist hash: {bb_state.zobrist_hash}")
    print()


# ============================================================================
# PART 8: EVALUATION FUNCTION
# ============================================================================

# Piece-square tables (from white's perspective, flip for black)
# Values encourage good piece placement

PAWN_TABLE = [
    [10, 10, 10, 10, 10],   # Promotion rank (y=0 for white)
    [ 5,  5,  5,  5,  5],
    [ 5,  5,  5,  5,  5],
    [0, 0, 0,  0,  0],
    [ 0,  0,  0,  0,  0]    # Starting rank (y=4 for white)
]

KNIGHT_TABLE = [
    [-5, -5, -5, -5, -5],
    [-5,  0,  0,  0, -5],
    [-5,  0,  0,  0, -5],   # Center is best
    [-5,  0,  0,  0, -5],
    [-5, -5, -5, -5, -5]
]

BISHOP_TABLE = [
    [-10, -5, -5, -5, -10],
    [ -5,  0,  0,  0,  -5],
    [ -5,  0,  5,  0,  -5],  # Center peak
    [ -5,  0,  0,  0,  -5],
    [-10, -5, -5, -5, -10]
]

RIGHT_TABLE = [
    [-5,  5,  5,  5, -5],   # Back rank advantage
    [ 0,  5,  5,  5,  0],
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0],
    [-5,  0,  0,  0, -5]
]

QUEEN_TABLE = [
    [-5,  0,  0,  0, -5],
    [ 0,  5,  5,  5,  0],
    [ 0,  5,  5,  5,  0],   # Strong center presence
    [ 0,  5,  5,  5,  0],
    [-5,  0,  0,  0, -5]
]

KING_TABLE = [
    [-20, -20, -20, -20, -20],  # Heavily penalized exposure
    [-15, -15, -15, -15, -15],
    [-10, -10, -10, -10, -10],
    [ -5,  -5,  -5,  -5,  -5],
    [  5,   5,   5,   5,   5]   # Safe home rank
]


def evaluate_bitboard(bb_state: BitboardState, player_is_white: bool) -> int:
    """
    Evaluate board position from player's perspective.

    Uses material count + piece-square tables + king safety for positional evaluation.
    Positive score = good for player, negative = bad for player.

    Args:
        bb_state: Current board state
        player_is_white: True if evaluating from white's perspective

    Returns:
        Evaluation score in centipawns (e.g., +100 = up a pawn)

    Algorithm:
        1. Calculate material balance (sum of piece values)
        2. Add positional bonuses from piece-square tables
        3. Add king safety evaluation (middlegame only)
        4. Return score from player's perspective
    """
    score = 0

    # Define piece lists for iteration
    white_pieces = [
        (bb_state.WP, PAWN, PAWN_TABLE),
        (bb_state.WN, KNIGHT, KNIGHT_TABLE),
        (bb_state.WB, BISHOP, BISHOP_TABLE),
        (bb_state.WQ, QUEEN, QUEEN_TABLE),
        (bb_state.WK, KING, KING_TABLE),
        (bb_state.WR, RIGHT, RIGHT_TABLE)
    ]

    black_pieces = [
        (bb_state.BP, PAWN, PAWN_TABLE),
        (bb_state.BN, KNIGHT, KNIGHT_TABLE),
        (bb_state.BB, BISHOP, BISHOP_TABLE),
        (bb_state.BQ, QUEEN, QUEEN_TABLE),
        (bb_state.BK, KING, KING_TABLE),
        (bb_state.BR, RIGHT, RIGHT_TABLE)
    ]

    # Track king positions for king safety evaluation
    white_king_sq = -1
    black_king_sq = -1

    # Evaluate white pieces
    for piece_bb, piece_type, pst in white_pieces:
        for sq in iter_bits(piece_bb):
            # Material value
            score += PIECE_VALUES[piece_type]

            # Positional value (from piece-square table)
            if pst is not None:
                x, y = index_to_xy(sq)
                # Tables are from white's perspective (y=0 is promotion rank)
                score += pst[y][x]

            # Track king position
            if piece_type == KING:
                white_king_sq = sq

    # Evaluate black pieces
    for piece_bb, piece_type, pst in black_pieces:
        for sq in iter_bits(piece_bb):
            # Material value (negative for opponent)
            score -= PIECE_VALUES[piece_type]

            # Positional value (flip table vertically for black)
            if pst is not None:
                x, y = index_to_xy(sq)
                # Flip y coordinate: black's promotion rank is y=4 → index 0
                flipped_y = 4 - y
                score -= pst[flipped_y][x]

            # Track king position
            if piece_type == KING:
                black_king_sq = sq

    # --- KING SAFETY EVALUATION (Middlegame only) ---
    # Count total pieces to determine if we're in middlegame
    total_pieces = count_bits(bb_state.occ_all)

    if total_pieces > 8:  # Middlegame
        # Evaluate white king safety
        if white_king_sq != -1:
            # FIX #14: Use precomputed KING_ATTACKS for neighbor squares
            nearby_allies = count_bits(KING_ATTACKS[white_king_sq] & bb_state.occ_white)

            # Apply king safety bonus/penalty
            if nearby_allies >= 2:
                score += 50
            elif nearby_allies == 1:
                score += 20
            else:
                score -= 50  # Isolated king penalty

        # Evaluate black king safety (subtract from score since it's opponent)
        if black_king_sq != -1:
            # FIX #14: Use precomputed KING_ATTACKS for neighbor squares
            nearby_allies = count_bits(KING_ATTACKS[black_king_sq] & bb_state.occ_black)

            # Apply king safety bonus/penalty (inverted for black)
            if nearby_allies >= 2:
                score -= 50
            elif nearby_allies == 1:
                score -= 20
            else:
                score += 50  # Opponent's isolated king is good for us

    # Return from player's perspective
    if player_is_white:
        return score
    else:
        return -score  # Flip score if evaluating from black's perspective
