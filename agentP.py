import random
import math
import time
import struct
from typing import Dict, Tuple, Optional, List, Any, NamedTuple
from dataclasses import dataclass, field
from extension.board_utils import list_legal_moves_for

# Logging/diagnostics flags
DEBUG = False
ENABLE_TREE_LOGGING = False
DEBUG_LEGAL = False

# =====================================================================
# === Logging Infrastructure
# =====================================================================


def log_move_str(piece, move_opt):
    """
    Converts a chess move to a human-readable string format for logging purposes.
    
    Args:
        piece: The chess piece object making the move. Must have a `.name` attribute
            and a `.position` attribute with `.x` and `.y` coordinates.
        move_opt: The move option object representing the destination. Must have a
            `.position` attribute with `.x` and `.y` coordinates.
    
    Returns:
        A string in the format "PieceName (sx,sy) -> (dx,dy)" where sx,sy are the
        source coordinates and dx,dy are the destination coordinates. Returns "None"
        if either piece or move_opt is missing or lacks the required attributes.
    
    Example:
        >>> log_move_str(piece, move)
        "Knight (1,2) -> (2,4)"
    """
    if piece and move_opt and hasattr(move_opt, 'position'):
        return f"{piece.name} ({piece.position.x},{piece.position.y}) -> ({move_opt.position.x},{move_opt.position.y})"
    return "None"


# =====================================================================
# === Constants
# =====================================================================

INF = float("inf")
START_SEARCH_DEPTH = 1
MAX_SEARCH_DEPTH = 30
# Position caching removed; we no longer index the board via Position
MATE_VALUE = 1_000_000
WIN_VALUE = 900_000
PHASE_VALUES = {
    "Pawn": 0,
    "Knight": 1,
    "Bishop": 1,
    "Rook": 2,
    "Right": 2,  # Custom piece
    "Queen": 4,
    "King": 0,
}

MAX_PHASE = 16

EVAL_BISHOP_PAIR = 30
EVAL_CENTER_CONTROL = 5
EVAL_PASSED_PAWN_MG = [0, 15, 30, 55, 80]
EVAL_PASSED_PAWN_EG = [0, 20, 40, 70, 100]
EVAL_DOUBLED_PAWN = -12
EVAL_ISOLATED_PAWN = -10
EVAL_ROOK_OPEN_FILE = 15
EVAL_ROOK_SEMIOPEN = 8
EVAL_RIGHT_OPEN_FILE = 12
EVAL_RIGHT_SEMIOPEN = 6
EVAL_KING_SHIELD_WEAK = 10
EVAL_KING_SHIELD_GONE = 20
EVAL_KING_STORM = 6

# Razoring and Futility margins (tunable)
RAZOR_MARGIN = {1: 150, 2: 225}
FUTILITY_MARGIN = {3: 200, 4: 300, 5: 400}
# Continuation history bonus scaler
CH_BONUS = 1
# Reverse Futility Pruning margins (depth -> margin)
RFP_MARGIN = {1: 120, 2: 160, 3: 210, 4: 260, 5: 320}

# Endgame drive weights (5x5 tuned, modest to avoid overpowering MG)
EVAL_EG_OPP_KING_TO_EDGE_R = 14  # KRK: drive to edge
EVAL_EG_OPP_KING_TO_CORNER_R = 12  # KRK: then corner
EVAL_EG_OPP_KING_TO_EDGE_Q = 10  # KQK: edge is OK
EVAL_EG_OPP_KING_TO_CORNER_Q = 14  # KQK: prefer corner
EVAL_EG_OWN_KING_PROXIMITY = 8  # Bring our king closer in KRK/KQK
EVAL_EG_ROOK_CUTOFF = 10  # Rook/right on same rank/file as enemy king
EVAL_EG_QUEEN_ADJ_PENALTY = -20  # Avoid stalemate-y queen adjacency if our king is far

# Advanced evaluation feature weights (initial guesses; tune via Texel/SPSA)
EVAL_ROOK_ON_7TH = 22
EVAL_ROOKS_ON_7TH = 38
EVAL_ROOK_ON_7TH_VS_KING = 8

EVAL_KNIGHT_OUTPOST = 18
EVAL_KNIGHT_OUTPOST_CENTER_BONUS = 6

EVAL_CONNECTED_PASSERS = 18
EVAL_BACKWARD_PAWN = -10

EVAL_BAD_BISHOP_PENALTY_PER_PAWN = -5

# King safety: attacker piece weights and file pressure/open-file penalties
EVAL_KING_ATTACK_WEIGHTS = {
    "Knight": 2,
    "Bishop": 2,
    "Rook": 3,
    "Right": 3,
    "Queen": 5,
}
EVAL_KING_ATTACK_SCALE = 2
EVAL_KING_FILE_PRESSURE = 2
EVAL_OPEN_FILE_TO_KING = 18
EVAL_SEMIOPEN_FILE_TO_KING = 10

IDX_TO_SQ_TABLE = [(i % 5, i // 5) for i in range(25)]
# SQ_TO_IDX_TABLE: 2D lookup table mapping [y][x] -> index. Some code
# historically treated this as a 2D table, so provide the 2D shape to
# remain compatible with older callers.
SQ_TO_IDX_TABLE = [[y * 5 + x for x in range(5)] for y in range(5)]

# Mobility by attacked squares (very fast proxy, avoids full legal move gen)
EVAL_MOBILITY_MG = 1
EVAL_MOBILITY_EG = 1
MOBILITY_WEIGHTS = {
    "Pawn": 1,
    "Knight": 2,
    "Bishop": 2,
    "Rook": 3,
    "Right": 3,
    "Queen": 4,
    "King": 1,
}


PIECE_VALUES_MG = {
    "Pawn": 120,
    "Knight": 350,
    "Bishop": 330,
    "Rook": 500,
    "Right": 800,
    "Queen": 900,
    "King": 20000,
}
PIECE_VALUES_EG = {
    "Pawn": 140,
    "Knight": 350,
    "Bishop": 330,
    "Rook": 500,
    "Right": 800,
    "Queen": 900,
    "King": 20000,
}


def mirror_pst(pst):
    """
    Returns a vertically mirrored copy of a 5x5 piece-square table (PST).

    This helper is used to derive black PSTs from white PSTs by flipping
    rank order.

    Args:
        pst: A list of length 5 where each element is a list of 5 numbers
            representing the PST rows from White's perspective (rank 0 at
            the bottom).

    Returns:
        A new PST list with the row order reversed (top <-> bottom).
    """
    return pst[::-1]

_ZERO_PST = [[0, 0, 0, 0, 0] for _ in range(5)]
# Non-zero PSTs (rows are y=0..4 from White's perspective; x=0..4)
PAWN_PST_MG = [
    [0, 0, 0, 0, 0],
    [2, 3, 3, 3, 2],
    [3, 4, 5, 4, 3],
    [5, 6, 7, 6, 5],
    [0, 0, 0, 0, 0],
]
PAWN_PST_EG = [
    [0, 0, 0, 0, 0],
    [1, 2, 2, 2, 1],
    [2, 3, 4, 3, 2],
    [3, 3, 4, 3, 3],
    [0, 0, 0, 0, 0],
]
KNIGHT_PST_MG = [
    [-10, -5, -3, -5, -10],
    [ -5,  0,  5,  0,  -5],
    [ -3,  5, 10,  5,  -3],
    [ -5,  0,  5,  0,  -5],
    [-10, -5, -3, -5, -10],
]
KNIGHT_PST_EG = [
    [-8, -4, -2, -4, -8],
    [ -4,  0,  4,  0, -4],
    [ -2,  4,  8,  4, -2],
    [ -4,  0,  4,  0, -4],
    [ -8, -4, -2, -4, -8],
]
BISHOP_PST_MG = [
    [ -5, -2, -2, -2, -5],
    [ -2,  3,  4,  3, -2],
    [ -2,  4,  6,  4, -2],
    [ -2,  3,  4,  3, -2],
    [ -5, -2, -2, -2, -5],
]
BISHOP_PST_EG = [
    [ -3, -1, -1, -1, -3],
    [ -1,  3,  4,  3, -1],
    [ -1,  4,  6,  4, -1],
    [ -1,  3,  4,  3, -1],
    [ -3, -1, -1, -1, -3],
]
ROOK_PST_MG = [
    [0, 0, 1, 0, 0],
    [1, 2, 3, 2, 1],
    [1, 2, 3, 2, 1],
    [1, 2, 3, 2, 1],
    [0, 0, 1, 0, 0],
]
ROOK_PST_EG = [
    [0, 0, 1, 0, 0],
    [0, 1, 2, 1, 0],
    [0, 1, 2, 1, 0],
    [0, 1, 2, 1, 0],
    [0, 0, 1, 0, 0],
]
RIGHT_PST_MG = [
    [0, 1, 2, 1, 0],
    [1, 3, 5, 3, 1],
    [2, 5, 8, 5, 2],
    [1, 3, 5, 3, 1],
    [0, 1, 2, 1, 0],
]
RIGHT_PST_EG = [
    [0, 1, 2, 1, 0],
    [1, 2, 4, 2, 1],
    [2, 4, 6, 4, 2],
    [1, 2, 4, 2, 1],
    [0, 1, 2, 1, 0],
]
QUEEN_PST_MG = [
    [ -4, -2, -1, -2, -4],
    [ -2,  0,  1,  0, -2],
    [ -1,  1,  2,  1, -1],
    [ -2,  0,  1,  0, -2],
    [ -4, -2, -1, -2, -4],
]
QUEEN_PST_EG = [
    [ -3, -1,  0, -1, -3],
    [ -1,  1,  2,  1, -1],
    [  0,  2,  3,  2,  0],
    [ -1,  1,  2,  1, -1],
    [ -3, -1,  0, -1, -3],
]
KING_PST_MG = [
    [ 2,  1,  0,  1,  2],
    [ 1,  0, -1,  0,  1],
    [ 0, -1, -2, -1,  0],
    [ 1,  0, -1,  0,  1],
    [ 2,  1,  0,  1,  2],
]
KING_PST_EG = [
    [ -2, -1,  0, -1, -2],
    [ -1,  1,  2,  1, -1],
    [  0,  2,  4,  2,  0],
    [ -1,  1,  2,  1, -1],
    [ -2, -1,  0, -1, -2],
]

PSTS_MG = {
    "white": {
        "Pawn": PAWN_PST_MG,
        "Knight": KNIGHT_PST_MG,
        "Bishop": BISHOP_PST_MG,
        "Rook": ROOK_PST_MG,
        "Right": RIGHT_PST_MG,
        "Queen": QUEEN_PST_MG,
        "King": KING_PST_MG,
    },
    "black": {
        "Pawn": mirror_pst(PAWN_PST_MG),
        "Knight": mirror_pst(KNIGHT_PST_MG),
        "Bishop": mirror_pst(BISHOP_PST_MG),
        "Rook": mirror_pst(ROOK_PST_MG),
        "Right": mirror_pst(RIGHT_PST_MG),
        "Queen": mirror_pst(QUEEN_PST_MG),
        "King": mirror_pst(KING_PST_MG),
    },
}
PSTS_EG = {
    "white": {
        "Pawn": PAWN_PST_EG,
        "Knight": KNIGHT_PST_EG,
        "Bishop": BISHOP_PST_EG,
        "Rook": ROOK_PST_EG,
        "Right": RIGHT_PST_EG,
        "Queen": QUEEN_PST_EG,
        "King": KING_PST_EG,
    },
    "black": {
        "Pawn": mirror_pst(PAWN_PST_EG),
        "Knight": mirror_pst(KNIGHT_PST_EG),
        "Bishop": mirror_pst(BISHOP_PST_EG),
        "Rook": mirror_pst(ROOK_PST_EG),
        "Right": mirror_pst(RIGHT_PST_EG),
        "Queen": mirror_pst(QUEEN_PST_EG),
        "King": mirror_pst(KING_PST_EG),
    },
}
PIECE_VALUES = {"mg": PIECE_VALUES_MG, "eg": PIECE_VALUES_EG}
PSTS = {"mg": PSTS_MG, "eg": PSTS_EG}

CENTER_SQUARES: List[Tuple[int, int]] = [
    (2, 2),
    (2, 1),
    (2, 3),
    (1, 2),
    (3, 2),
]

_LAST_SEARCH_INFO: Dict[str, Any] = {}
_PERSISTENT_VAR: Dict[str, Any] = {}


# =====================================================================
# === Search Tree Instrumentation (Browser Viewer Support)
# =====================================================================

def get_legal_moves_cached(board, player, var, board_hash):
    """
    Retrieves legal moves from cache or computes them if not cached.
    
    This function implements a caching mechanism for legal move generation to
    avoid redundant computations. It uses a combination of board hash and player
    name as the cache key, storing results in the provided variable dictionary.
    
    Args:
        board: The current board state (used for move generation if cache miss).
        player: The player whose legal moves are being requested. Must have a
            `.name` attribute ("white" or "black").
        var: A mutable dictionary used to store the cache. The cache is stored
            under the key '_legal_moves_cache' as a dictionary mapping
            (board_hash, player_name) tuples to lists of legal moves.
        board_hash: A hash value representing the current board state. Used as
            part of the cache key along with player name.
    
    Returns:
        A list of legal moves for the specified player. Each move is represented
        as a tuple (piece, move_option) as returned by list_legal_moves_for().
    
    Notes:
        - Cache is stored in var['_legal_moves_cache'] as a dictionary
        - Cache key is (board_hash, player.name) tuple
        - On cache miss, calls list_legal_moves_for() and stores the result
        - Helps reduce redundant move generation during search
    """
    cache_key = (board_hash, player.name)
    if cache_key not in var.setdefault('_legal_moves_cache', {}):
        var['_legal_moves_cache'][cache_key] = list_legal_moves_for(board, player)
    return var['_legal_moves_cache'][cache_key]

# =====================================================================
# === Bitboard Utilities (Replaces bitboard_utils import)
# =====================================================================


_BB_VALUES_LIST = [
    120,
    350,
    330,
    500,
    900,
    20000,
    120,
    350,
    330,
    500,
    900,
    20000,
]
_BB_INDEX_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def square_index(x: int, y: int) -> int:
    """
    Converts 2D board coordinates into a packed 0..24 index.

    The 5x5 board is laid out row-major with 0 at (0,0) and 24 at (4,4).

    Args:
        x: File/column, 0..4.
        y: Rank/row, 0..4.

    Returns:
        The linear index y * 5 + x in the range [0, 24].
    """
    # SQ_TO_IDX_TABLE is a flat list of length 25 (y * 5 + x).
    # Compute the index directly to avoid indexing a 1D list as 2D.
    return y * 5 + x


def index_to_sq(idx: int) -> Tuple[int, int]:
    """
    Converts a packed 0..24 index back to (x, y) board coordinates.

    Args:
        idx: Linear index in [0, 24].

    Returns:
        A tuple (x, y) with x,y in [0, 4].
    """
    return IDX_TO_SQ_TABLE[idx]


# --- 5x5 board constants and masks ---
# ALL25: all 25 squares set
ALL25: int = (1 << 25) - 1

# File and rank masks
FILE_0_MASK: int = sum(1 << square_index(0, y) for y in range(5))
FILE_4_MASK: int = sum(1 << square_index(4, y) for y in range(5))
RANK_0_MASK: int = sum(1 << square_index(x, 0) for x in range(5))
RANK_4_MASK: int = sum(1 << square_index(x, 4) for x in range(5))

# Convenience NOT masks (within 25-bit board)
NOT_FILE_0: int = ALL25 ^ FILE_0_MASK
NOT_FILE_4: int = ALL25 ^ FILE_4_MASK


# --- BETWEEN_RAYS table: squares strictly between two collinear squares (0 if not collinear) ---
def _gen_between_rays() -> list[list[int]]:
    """
    Precomputes a lookup table of squares strictly between two collinear squares.
    
    For a 5x5 board, this generates a 25x25 table where table[a][b] contains a
    bitboard mask of all squares that lie strictly between square indices a and b
    along the same line (rank, file, or diagonal). If squares a and b are not
    collinear, the entry is 0.
    
    Returns:
        A 25x25 list of lists, where each entry is an integer bitboard mask.
        The mask has bits set for squares between the two given squares (exclusive
        of the endpoints) along the connecting ray.
    
    Notes:
        - Uses 8 directions: 4 orthogonals (N, S, E, W) and 4 diagonals (NE, NW, SE, SW)
        - Only includes squares strictly between a and b (not including a or b)
        - Returns 0 if squares are not collinear or are adjacent
    """
    table: list[list[int]] = [[0 for _ in range(25)] for __ in range(25)]
    # Directions: 8 rays (orthogonals + diagonals)
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (1, 1), (-1, 1), (1, -1),
    ]
    for a in range(25):
        ax, ay = index_to_sq(a)
        for b in range(25):
            if a == b:
                table[a][b] = 0
                continue
            bx, by = index_to_sq(b)
            dx = bx - ax
            dy = by - ay
            step = None
            for sx, sy in directions:
                # Check if (dx,dy) is a multiple of (sx,sy)
                if sx == 0 and dx != 0:
                    continue
                if sy == 0 and dy != 0:
                    continue
                # For diagonal, absolute dx == absolute dy along same sign
                if sx != 0 and sy != 0:
                    if abs(dx) != abs(dy):
                        continue
                    # Normalize step direction
                    sgnx = 1 if dx > 0 else -1
                    sgny = 1 if dy > 0 else -1
                    if sgnx != sx or sgny != sy:
                        continue
                    step = (sx, sy)
                    break
                # For orthogonals
                if sx == 0 and dx == 0 and (sy == 1 and dy > 0 or sy == -1 and dy < 0):
                    step = (sx, 1 if dy > 0 else -1)
                    break
                if sy == 0 and dy == 0 and (sx == 1 and dx > 0 or sx == -1 and dx < 0):
                    step = (1 if dx > 0 else -1, sy)
                    break
            if step is None:
                table[a][b] = 0
                continue
            # Walk from a towards b, collecting strictly between squares
            sx, sy = step
            x, y = ax + sx, ay + sy
            mask = 0
            while 0 <= x < 5 and 0 <= y < 5:
                if (x, y) == (bx, by):
                    break
                mask |= 1 << square_index(x, y)
                x += sx
                y += sy
            # If (b) was not reached in line, clear mask
            if not (x == bx and y == by):
                mask = 0
            table[a][b] = mask
    return table


BETWEEN_RAYS: list[list[int]] = _gen_between_rays()


def _pop_lsb_njit(bb: int) -> int:
    """
    Returns the index of the least significant set bit in a bitboard.

    Args:
        bb: Bitboard integer where bit i corresponds to square i.

    Returns:
        The index [0, 63+] of the least significant set bit; -1 if bb == 0.

    Notes:
        Uses the identity x & -x to isolate the LSB and bit_length to get
        its index without loops or branches.
    """
    if bb == 0:
        return -1
    return (bb & -bb).bit_length() - 1


# --- Bitboard Pre-generation -----------------------------------------


def _gen_knight_moves() -> list[int]:
    """
    Precomputes knight attack bitboards for each of the 25 squares.

    Returns:
        A length-25 list where entry i is a bitboard of knight attacks from i.

    How it works:
        For each square (r,c) we consider the 8 L-shaped offsets, check they
        are on-board, and set the corresponding bit in the bitboard.
    """
    moves = [0] * 25
    for r in range(5):
        for c in range(5):
            sq = r * 5 + c
            bb = 0
            for dr, dc in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 5 and 0 <= nc < 5:
                    bb |= 1 << (nr * 5 + nc)
            moves[sq] = bb
    return moves


def _gen_king_moves() -> list[int]:
    """
    Precomputes king attack bitboards for each of the 25 squares.

    Returns:
        A length-25 list where entry i is a bitboard of king attacks from i.

    How it works:
        For each square (r,c) we consider all 8 neighboring deltas and set
        their bits if they are on-board.
    """
    moves = [0] * 25
    for r in range(5):
        for c in range(5):
            sq = r * 5 + c
            bb = 0
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < 5 and 0 <= nc < 5:
                        bb |= 1 << (nr * 5 + nc)
            moves[sq] = bb
    return moves


_KNIGHT_MOVES = _gen_knight_moves()
_KING_MOVES = _gen_king_moves()

FILE_MASKS = [sum(1 << square_index(f, y) for y in range(5)) for f in range(5)]
RANK_MASKS = [sum(1 << square_index(x, r) for x in range(5)) for r in range(5)]
CENTER_MASK = sum(1 << square_index(x, y) for (x, y) in CENTER_SQUARES)

RING1_MASKS = [0] * 25
for sq in range(25):
    kx, ky = index_to_sq(sq)
    m = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            tx, ty = kx + dx, ky + dy
            if 0 <= tx < 5 and 0 <= ty < 5:
                m |= 1 << square_index(tx, ty)
    RING1_MASKS[sq] = m

LIGHT_MASK = sum(1 << square_index(x, y) for y in range(5) for x in range(5) if ((x + y) & 1) == 0)
DARK_MASK = ALL25 ^ LIGHT_MASK

WHITE_PAWN_ATTACKS = [0]*25
BLACK_PAWN_ATTACKS = [0]*25
for sq in range(25):
    x, y = index_to_sq(sq)
    m = 0
    ny = y - 1
    if ny >= 0:
        if x - 1 >= 0: m |= 1 << square_index(x - 1, ny)
        if x + 1 < 5:  m |= 1 << square_index(x + 1, ny)
    WHITE_PAWN_ATTACKS[sq] = m
    m = 0
    ny = y + 1
    if ny < 5:
        if x - 1 >= 0: m |= 1 << square_index(x - 1, ny)
        if x + 1 < 5:  m |= 1 << square_index(x + 1, ny)
    BLACK_PAWN_ATTACKS[sq] = m

# =====================================================================
# === Precomputed Line Definitions and Ray Attack Tables (5x5)
# =====================================================================
#
# We precompute rank/file/diagonal line layouts and constant-time attack
# tables for sliding pieces. Each table indexes by:
#   - line id
#   - position index within the line
#   - a compact occupancy mask for the line (bit i corresponds to line[i])
#
# For a given square and global occupancy, we map to the line and build the
# line-occupancy mask (with the moving square cleared) and then look up the
# precomputed bitboard of attacked squares.

# Line layouts (lists of lists of global square indices)
RANK_LINES: list[list[int]] = [[(r * 5 + c) for c in range(5)] for r in range(5)]
FILE_LINES: list[list[int]] = [[(r * 5 + c) for r in range(5)] for c in range(5)]

# Diagonals: NW-SE has constant (x - y), shifted to [0..8] by +4.
#            NE-SW has constant (x + y) in [0..8].
DIAG_A_LINES: list[list[int]] = []  # NW-SE
for d in range(-4, 5):
    line = []
    y_start = max(0, -d)
    y_end = min(4, 4 - d)
    for y in range(y_start, y_end + 1):
        x = y + d
        line.append(y * 5 + x)
    DIAG_A_LINES.append(line)

DIAG_B_LINES: list[list[int]] = []  # NE-SW
for s in range(0, 9):
    line = []
    y_start = max(0, s - 4)
    y_end = min(4, s)
    for y in range(y_start, y_end + 1):
        x = s - y
        line.append(y * 5 + x)
    DIAG_B_LINES.append(line)

# Per-square mappings: (line_id, pos_in_line)
SQ_TO_RANK: list[tuple[int, int]] = [(-1, -1)] * 25
SQ_TO_FILE: list[tuple[int, int]] = [(-1, -1)] * 25
SQ_TO_DA: list[tuple[int, int]] = [(-1, -1)] * 25
SQ_TO_DB: list[tuple[int, int]] = [(-1, -1)] * 25

for r in range(5):
    for i, sq in enumerate(RANK_LINES[r]):
        SQ_TO_RANK[sq] = (r, i)
for f in range(5):
    for i, sq in enumerate(FILE_LINES[f]):
        SQ_TO_FILE[sq] = (f, i)
for d in range(9):
    for i, sq in enumerate(DIAG_A_LINES[d]):
        SQ_TO_DA[sq] = (d, i)
for d in range(9):
    for i, sq in enumerate(DIAG_B_LINES[d]):
        SQ_TO_DB[sq] = (d, i)


def _build_line_attacks_table(line: list[int]) -> list[list[int]]:
    """
    For a single line (list of global square indices), build a table:
      table[pos][occ_mask] -> attack bitboard
    where occ_mask is an integer in [0, 2^L) with bit i set if line[i] is occupied.
    Attacks include the first blocker square in each direction.
    """
    L = len(line)
    size = 1 << L
    table = [[0 for _ in range(size)] for __ in range(L)]
    for pos in range(L):
        for occ in range(size):
            bb = 0
            # Left side (towards smaller indices)
            for i in range(pos - 1, -1, -1):
                sqi = line[i]
                bb |= (1 << sqi)
                if (occ >> i) & 1:
                    break
            # Right side (towards larger indices)
            for i in range(pos + 1, L):
                sqi = line[i]
                bb |= (1 << sqi)
                if (occ >> i) & 1:
                    break
            table[pos][occ] = bb
    return table


# Build attack tables for all ranks/files/diagonals
RANK_ATTACKS: list[list[list[int]]] = [_build_line_attacks_table(RANK_LINES[r]) for r in range(5)]
FILE_ATTACKS: list[list[list[int]]] = [_build_line_attacks_table(FILE_LINES[f]) for f in range(5)]
DIAG_A_ATTACKS: list[list[list[int]]] = [_build_line_attacks_table(DIAG_A_LINES[d]) for d in range(9)]
DIAG_B_ATTACKS: list[list[list[int]]] = [_build_line_attacks_table(DIAG_B_LINES[d]) for d in range(9)]

_RAY_TABLES_READY = True


def _line_occ_from_global(occ: int, line: list[int]) -> int:
    """
    Extracts a compact occupancy mask for a specific line from the global occupancy bitboard.
    
    This function maps the global 25-bit occupancy bitboard to a smaller bitmask
    representing only the squares that belong to a specific line (rank, file, or diagonal).
    The resulting mask uses a compact representation where bit i corresponds to line[i].
    
    Args:
        occ: Global 25-bit occupancy bitboard where bit i (0-24) represents square i.
        line: List of square indices (0-24) that form a line (rank, file, or diagonal).
            The order of indices in the list determines the bit positions in the result.
    
    Returns:
        An integer bitmask where bit i is set if and only if the square at line[i]
        is occupied in the global occupancy bitboard. The mask length equals len(line).
    
    Example:
        For a rank line [10, 11, 12, 13, 14] and occ having bits set at 10 and 13,
        returns a 5-bit mask with bits 0 and 3 set.
    """
    mask = 0
    for i, sq in enumerate(line):
        if occ & (1 << sq):
            mask |= (1 << i)
    return mask


def _get_rook_attacks(sq: int, occ: int) -> int:
    """
    Computes rook attack bitboard from a square using precomputed lookup tables.
    
    This function provides constant-time rook attack generation by using precomputed
    rank and file attack tables. The attacks include all squares a rook can reach
    from the given square, stopping at the first blocker in each direction.
    
    Args:
        sq: Source square index in the range [0, 24] (5x5 board).
        occ: Occupancy bitboard representing all pieces on the board. Rays stop
            at the first occupied square in each direction, including that blocker.
    
    Returns:
        A 25-bit bitboard where each set bit represents a square attacked by a rook
        from sq. Includes squares along ranks and files, stopping at blockers.
    
    Notes:
        - Uses precomputed RANK_ATTACKS and FILE_ATTACKS tables for efficiency
        - Falls back to scanning method if tables are unavailable
        - Includes the first blocker square in each direction
    """
    if not _RAY_TABLES_READY:
        return _get_rook_attacks_scan(sq, occ)
    r_id, r_pos = SQ_TO_RANK[sq]
    f_id, f_pos = SQ_TO_FILE[sq]
    # Rank
    r_line = RANK_LINES[r_id]
    r_occ = _line_occ_from_global(occ, r_line) & ~(1 << r_pos)
    r_bb = RANK_ATTACKS[r_id][r_pos][r_occ]
    # File
    f_line = FILE_LINES[f_id]
    f_occ = _line_occ_from_global(occ, f_line) & ~(1 << f_pos)
    f_bb = FILE_ATTACKS[f_id][f_pos][f_occ]
    return r_bb | f_bb


def _get_bishop_attacks(sq: int, occ: int) -> int:
    """
    Computes bishop attack bitboard from a square using precomputed lookup tables.
    
    This function provides constant-time bishop attack generation by using precomputed
    diagonal attack tables. The attacks include all squares a bishop can reach
    from the given square along both diagonals, stopping at the first blocker.
    
    Args:
        sq: Source square index in the range [0, 24] (5x5 board).
        occ: Occupancy bitboard representing all pieces on the board. Rays stop
            at the first occupied square in each diagonal direction, including that blocker.
    
    Returns:
        A 25-bit bitboard where each set bit represents a square attacked by a bishop
        from sq. Includes squares along both diagonals (NW-SE and NE-SW), stopping at blockers.
    
    Notes:
        - Uses precomputed DIAG_A_ATTACKS and DIAG_B_ATTACKS tables for efficiency
        - Falls back to scanning method if tables are unavailable
        - Includes the first blocker square in each direction
    """
    if not _RAY_TABLES_READY:
        return _get_bishop_attacks_scan(sq, occ)
    da_id, da_pos = SQ_TO_DA[sq]
    db_id, db_pos = SQ_TO_DB[sq]
    # Diagonal A (NW-SE)
    da_line = DIAG_A_LINES[da_id]
    da_occ = _line_occ_from_global(occ, da_line) & ~(1 << da_pos)
    da_bb = DIAG_A_ATTACKS[da_id][da_pos][da_occ]
    # Diagonal B (NE-SW)
    db_line = DIAG_B_LINES[db_id]
    db_occ = _line_occ_from_global(occ, db_line) & ~(1 << db_pos)
    db_bb = DIAG_B_ATTACKS[db_id][db_pos][db_occ]
    return da_bb | db_bb

def _get_rook_attacks_scan(sq: int, occ: int) -> int:
    """
    Computes rook ray attacks from a square on a 5x5 board by scanning.

    Args:
        sq: Source square index [0, 24].
        occ: Occupancy bitboard; rays stop at the first blocker.

    Returns:
        Bitboard of all squares attacked by a rook from sq.

    How it works:
        Cast rays in four cardinal directions until the board edge or the
        first occupied square, including that blocker square.
    """
    bb = 0
    r, c = sq // 5, sq % 5
    # Directions: (N, S, E, W)
    for dr, dc in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
        for i in range(1, 5):
            nr, nc = r + dr * i, c + dc * i
            if not (0 <= nr < 5 and 0 <= nc < 5):
                break
            sq_bit = 1 << (nr * 5 + nc)
            bb |= sq_bit
            if occ & sq_bit:  # Stop at the first blocker
                break
    return bb


def _get_bishop_attacks_scan(sq: int, occ: int) -> int:
    """
    Computes bishop diagonal ray attacks from a square on a 5x5 board by scanning.

    Args:
        sq: Source square index [0, 24].
        occ: Occupancy bitboard; rays stop at the first blocker.

    Returns:
        Bitboard of all squares attacked by a bishop from sq.

    How it works:
        Cast rays along the four diagonals until the board edge or a blocker,
        including that blocker square.
    """
    bb = 0
    r, c = sq // 5, sq % 5
    for dr, dc in [(-1, 1), (-1, -1), (1, 1), (1, -1)]:
        for i in range(1, 5):
            nr, nc = r + dr * i, c + dc * i
            if not (0 <= nr < 5 and 0 <= nc < 5):
                break
            sq_bit = 1 << (nr * 5 + nc)
            bb |= sq_bit
            if occ & sq_bit:
                break
    return bb


class BBPos(NamedTuple):
    """Holds all bitboards for a position. (UPDATED FOR 'Right' PIECE)"""


    WP: int
    WN: int
    WB: int
    WR: int
    WQ: int
    WK: int
    WRi: int
    BP: int
    BN: int
    BB: int
    BR: int
    BQ: int
    BK: int
    BRi: int
    occ_white: int
    occ_black: int
    occ_all: int


_PIECE_TYPE_TO_IDX = {
    "Pawn": 0,
    "Knight": 1,
    "Bishop": 2,
    "Rook": 3,
    "Queen": 4,
    "King": 5,
    "Right": 6,
}


_COLOR_OFFSET = {"white": 0, "black": 7}


 
 
@dataclass(slots=True)
class BitboardState:
    """
    Compact internal engine state made of per-piece bitboards, aggregate occupancy,
    side-to-move (0=white, 1=black), and a Zobrist hash key.
    """
    WP: int
    WN: int
    WB: int
    WR: int
    WQ: int
    WK: int
    WRi: int
    BP: int
    BN: int
    BB: int
    BR: int
    BQ: int
    BK: int
    BRi: int
    occ_white: int
    occ_black: int
    occ_all: int
    side_to_move: int  # 0 = white, 1 = black
    zkey: int
    mg_mat_pst: int
    eg_mat_pst: int


def convert_board_to_bb_state(board, player, zobrist: "Zobrist") -> BitboardState:
    """
    Converts a ChessMaker board representation to the internal BitboardState format.
    
    This function serves as a bridge between the external ChessMaker board API
    and the internal bitboard-based engine representation. It extracts all
    piece positions, computes occupancy masks, determines side-to-move, and
    generates a Zobrist hash for the position.
    
    Args:
        board: A ChessMaker-compatible board object that supports iteration over
            pieces via `board.get_pieces()`. Each piece must have `.name`,
            `.position` (with `.x` and `.y`), and `.player.name` attributes.
        player: The player object whose turn it is. Must have a `.name` attribute
            that is either "white" or "black".
        zobrist: A Zobrist hashing instance used to compute the position hash key.
            Must implement `compute_full_hash(board, player_name)` method.
    
    Returns:
        A BitboardState dataclass containing:
        - Individual piece bitboards for all 7 piece types and 2 colors (14 bitboards)
        - Aggregate occupancy masks (occ_white, occ_black, occ_all)
        - side_to_move: 0 for white, 1 for black
        - zkey: 64-bit Zobrist hash of the position
    
    Raises:
        Exception: If Zobrist hash computation fails, zkey defaults to 0.
    
    Notes:
        - This is a one-time conversion; subsequent moves use incremental updates
        - Handles the custom "Right" piece type (rook|knight combination)
    """
    bbpos = bb_from_board(board)
    stm = 0 if getattr(player, "name", "white") == "white" else 1
    try:
        zkey = zobrist.compute_full_hash(board, player.name)
    except Exception:
        zkey = 0
    # Compute initial Material+PST differential (MG/EG) from bb position
    def _compute_mat_pst_from_bbpos(bbpos: BBPos) -> tuple[int, int]:
        mg = 0
        eg = 0
        # White pieces
        white_list = [
            ("Pawn", bbpos.WP),
            ("Knight", bbpos.WN),
            ("Bishop", bbpos.WB),
            ("Rook", bbpos.WR),
            ("Right", bbpos.WRi),
            ("Queen", bbpos.WQ),
            ("King", bbpos.WK),
        ]
        for name, bbmask in white_list:
            val_mg = 0 if name == "King" else PIECE_VALUES_MG.get(name, 0)
            val_eg = 0 if name == "King" else PIECE_VALUES_EG.get(name, 0)
            pst_mg = PSTS_MG["white"][name]
            pst_eg = PSTS_EG["white"][name]
            for sq in _iter_set_bits(bbmask):
                x, y = index_to_sq(sq)
                mg += val_mg + pst_mg[y][x]
                eg += val_eg + pst_eg[y][x]
        # Black pieces (subtract)
        black_list = [
            ("Pawn", bbpos.BP),
            ("Knight", bbpos.BN),
            ("Bishop", bbpos.BB),
            ("Rook", bbpos.BR),
            ("Right", bbpos.BRi),
            ("Queen", bbpos.BQ),
            ("King", bbpos.BK),
        ]
        for name, bbmask in black_list:
            val_mg = 0 if name == "King" else PIECE_VALUES_MG.get(name, 0)
            val_eg = 0 if name == "King" else PIECE_VALUES_EG.get(name, 0)
            pst_mg = PSTS_MG["black"][name]
            pst_eg = PSTS_EG["black"][name]
            for sq in _iter_set_bits(bbmask):
                x, y = index_to_sq(sq)
                mg -= (val_mg + pst_mg[y][x])
                eg -= (val_eg + pst_eg[y][x])
        return int(mg), int(eg)
    mg0, eg0 = _compute_mat_pst_from_bbpos(bbpos)
    return BitboardState(
        WP=bbpos.WP,
        WN=bbpos.WN,
        WB=bbpos.WB,
        WR=bbpos.WR,
        WQ=bbpos.WQ,
        WK=bbpos.WK,
        WRi=bbpos.WRi,
        BP=bbpos.BP,
        BN=bbpos.BN,
        BB=bbpos.BB,
        BR=bbpos.BR,
        BQ=bbpos.BQ,
        BK=bbpos.BK,
        BRi=bbpos.BRi,
        occ_white=bbpos.occ_white,
        occ_black=bbpos.occ_black,
        occ_all=bbpos.occ_all,
        side_to_move=stm,
        zkey=zkey,
        mg_mat_pst=mg0,
        eg_mat_pst=eg0,
    )


@dataclass(frozen=True, slots=True)
class BBMove:
    """
    Internal compact move:
      from_sq, to_sq in [0..24]
      promo: 0 = none, 4 = promote to Queen (matches PIECE idx for convenience)
      piece_type: 0..6 (Pawn..Right)
      captured_type: -1 if none else 0..6
    """
    from_sq: int
    to_sq: int
    promo: int
    piece_type: int
    captured_type: int


def pack_move(from_sq: int, to_sq: int, piece_type: int, captured_type: int, promo: int) -> int:
    """
    Packs a move into a 17-bit integer:
      bits 0..4  : from_sq (0-24)
      bits 5..9  : to_sq   (0-24)
      bits 10..12: piece_type (0-6)
      bits 13..15: captured_type+1 (0=None, 1..7 piece)
      bit  16    : promo flag (1 if promo==4 else 0)
    """
    cap_val = (captured_type + 1) & 7
    promo_val = 1 if promo == 4 else 0
    return (from_sq & 31) | ((to_sq & 31) << 5) | ((piece_type & 7) << 10) | (cap_val << 13) | (promo_val << 16)


def unpack_move(move_int: int) -> tuple[int, int, int, int, int]:
    """
    Unpacks a 17-bit packed move integer into (from_sq, to_sq, piece_type, captured_type, promo).
    """
    from_sq = move_int & 31
    to_sq = (move_int >> 5) & 31
    piece_type = (move_int >> 10) & 7
    captured_type = ((move_int >> 13) & 7) - 1
    promo = 4 if ((move_int >> 16) & 1) else 0
    return from_sq, to_sq, piece_type, captured_type, promo


def bbmove_to_tuple_xy(m: "BBMove") -> Tuple[int, int, int, int]:
    """
    Converts an internal BBMove object to a coordinate tuple representation.
    
    This function extracts the source and destination coordinates from a BBMove
    and converts them from linear square indices to (x, y) coordinate pairs.
    The resulting tuple is compatible with external move matching functions.
    
    Args:
        m: A BBMove dataclass containing:
            - from_sq: Source square index [0, 24]
            - to_sq: Destination square index [0, 24]
            - Other fields (promo, piece_type, captured_type) are ignored
    
    Returns:
        A tuple (sx, sy, dx, dy) where:
        - sx, sy: Source square coordinates (x, y) in [0, 4]
        - dx, dy: Destination square coordinates (x, y) in [0, 4]
    
    Example:
        >>> m = BBMove(from_sq=6, to_sq=11, ...)
        >>> bbmove_to_tuple_xy(m)
        (1, 1, 1, 2)  # Square 6 = (1,1), Square 11 = (1,2)
    """
    # Accept both packed-int moves and BBMove dataclass for compatibility
    if isinstance(m, int):
        from_sq, to_sq, _pt, _ct, _pr = unpack_move(m)
        sx, sy = index_to_sq(from_sq)
        dx, dy = index_to_sq(to_sq)
    else:
        sx, sy = index_to_sq(m.from_sq)
        dx, dy = index_to_sq(m.to_sq)
    return sx, sy, dx, dy


# --- Helpers for bit iteration and capture typing ---
def _iter_set_bits(bb: int):
    """
    Iterates over all set bits in a bitboard, yielding their square indices.
    
    This generator function efficiently extracts all set bits from a bitboard
    using bit manipulation tricks. It uses the identity `x & -x` to isolate the
    least significant bit, then clears it before finding the next one.
    
    Args:
        bb: An integer bitboard where bit i represents square i. Can be any
            non-negative integer representing a 25-bit (or larger) bitboard.
    
    Yields:
        Integer square indices (0-24 for 5x5 board) corresponding to each
        set bit in the bitboard, in order from least to most significant bit.
    
    Example:
        >>> bb = 0b10110  # Bits 1, 2, and 4 are set
        >>> list(_iter_set_bits(bb))
        [1, 2, 4]
    
    Notes:
        - Uses bit_length() to find the index of the LSB
        - Clears each bit after yielding to avoid infinite loops
        - Order is from LSB to MSB (not necessarily board order)
    """
    while bb:
        lsb = bb & -bb
        idx = lsb.bit_length() - 1
        yield idx
        bb ^= lsb


def _get_piece_type_at_square(bb: "BitboardState", color_white: bool, sq: int) -> int:
    """
    Identifies the piece type at a specific square for a given color.
    
    This function checks all piece bitboards for the specified color to determine
    which piece (if any) occupies the given square. The piece type is returned as
    an integer index matching the internal piece type encoding.
    
    Args:
        bb: BitboardState containing all piece bitboards and occupancy information.
        color_white: True to check white pieces, False to check black pieces.
        sq: Square index in the range [0, 24] to check for a piece.
    
    Returns:
        An integer piece type index:
        - 0: Pawn
        - 1: Knight
        - 2: Bishop
        - 3: Rook
        - 4: Queen
        - 5: King
        - 6: Right (custom piece)
        - -1: No piece of the specified color at this square
    
    Notes:
        - Only checks pieces of the specified color
        - Returns -1 if the square is empty or occupied by the opposite color
        - Checks piece bitboards in order: Pawn, Knight, Bishop, Rook, Queen, King, Right
    """
    mask = 1 << sq
    if color_white:
        if bb.WP & mask: return 0
        if bb.WN & mask: return 1
        if bb.WB & mask: return 2
        if bb.WR & mask: return 3
        if bb.WQ & mask: return 4
        if bb.WK & mask: return 5
        if bb.WRi & mask: return 6
    else:
        if bb.BP & mask: return 0
        if bb.BN & mask: return 1
        if bb.BB & mask: return 2
        if bb.BR & mask: return 3
        if bb.BQ & mask: return 4
        if bb.BK & mask: return 5
        if bb.BRi & mask: return 6
    return -1


def attackers_to_square(
    bb: "BitboardState",
    sq: int,
    by_white: bool,
    occ_override: Optional[int] = None,
) -> int:
    """
    Returns a bitboard of all pieces of the given color that attack 'sq'.
    Uses 'occ_override' for sliding attacks if provided.
    
    Coordinate convention on 5x5:
      - Linear index idx = y*5 + x with x,y in [0..4].
      - White pawns move towards smaller y (push uses >>5), so a white pawn
        that attacks target (r,c) must be located at (r+1, c±1).
      - Black pawns move towards larger y (push uses <<5), so a black pawn
        that attacks target (r,c) must be located at (r-1, c±1).
    """
    occ = bb.occ_all if occ_override is None else int(occ_override)
    r, c = (sq // 5), (sq % 5)
    attackers = 0

    if by_white:
        # Pawns: white pawns attack up-left/up-right -> come from (r+1,c±1)
        if r + 1 < 5:
            if c - 1 >= 0:
                from_sq = (r + 1) * 5 + (c - 1)
                if bb.WP & (1 << from_sq):
                    attackers |= (1 << from_sq)
            if c + 1 < 5:
                from_sq = (r + 1) * 5 + (c + 1)
                if bb.WP & (1 << from_sq):
                    attackers |= (1 << from_sq)
        # Knights and Right (knight component)
        attackers |= _KNIGHT_MOVES[sq] & (bb.WN | bb.WRi)
        # Bishop/Queen
        attackers |= _get_bishop_attacks(sq, occ) & (bb.WB | bb.WQ)
        # Rook/Queen/Right (rook component)
        attackers |= _get_rook_attacks(sq, occ) & (bb.WR | bb.WQ | bb.WRi)
        # King
        attackers |= _KING_MOVES[sq] & bb.WK
    else:
        # Pawns: black pawns attack down-left/down-right -> come from (r-1,c±1)
        if r - 1 >= 0:
            if c - 1 >= 0:
                from_sq = (r - 1) * 5 + (c - 1)
                if bb.BP & (1 << from_sq):
                    attackers |= (1 << from_sq)
            if c + 1 < 5:
                from_sq = (r - 1) * 5 + (c + 1)
                if bb.BP & (1 << from_sq):
                    attackers |= (1 << from_sq)
        # Knights and Right (knight component)
        attackers |= _KNIGHT_MOVES[sq] & (bb.BN | bb.BRi)
        # Bishop/Queen
        attackers |= _get_bishop_attacks(sq, occ) & (bb.BB | bb.BQ)
        # Rook/Queen/Right (rook component)
        attackers |= _get_rook_attacks(sq, occ) & (bb.BR | bb.BQ | bb.BRi)
        # King
        attackers |= _KING_MOVES[sq] & bb.BK

    return attackers


def calculate_legality_masks(
    bb: "BitboardState",
    ksq: int,
    stm_white: bool,
) -> Tuple[int, int, Dict[int, int]]:
    """
    Computes legality helpers for the side-to-move:
      - checkers_bb: enemy pieces currently giving check to our king
      - pin_mask_absolute: friendly pieces absolutely pinned to our king
      - pin_ray_map: allowed ray squares for each pinned piece (inclusive of attacker)
    """
    opp_white = (not stm_white)
    # Checkers to our king square
    checkers_bb = attackers_to_square(bb, ksq, by_white=opp_white)

    own_occ = bb.occ_white if stm_white else bb.occ_black
    pin_mask_absolute = 0
    pin_ray_map: Dict[int, int] = {}

    # Ray directions: orthogonals + diagonals
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (1, 1), (-1, 1), (1, -1),
    ]

    kx, ky = index_to_sq(ksq)
    for sx, sy in directions:
        x, y = kx + sx, ky + sy
        first_sq = -1
        first_is_own = False
        # Step until first piece or edge
        while 0 <= x < 5 and 0 <= y < 5:
            s = square_index(x, y)
            if bb.occ_all & (1 << s):
                first_sq = s
                first_is_own = bool(own_occ & (1 << s))
                break
            x += sx
            y += sy
        if first_sq < 0:
            continue
        # Look for a potential pin: first piece must be ours
        if not first_is_own:
            continue
        # Continue to find the next piece in the same ray
        x, y = (first_sq % 5) + sx, (first_sq // 5) + sy
        while 0 <= x < 5 and 0 <= y < 5:
            s = square_index(x, y)
            if bb.occ_all & (1 << s):
                # Check if this enemy piece is a compatible slider along this ray
                # Enemy piece must be identified using opponent color
                enemy_piece_type = _get_piece_type_at_square(bb, color_white=(not stm_white), sq=s)
                is_orth = (sx == 0 or sy == 0)
                is_diag = (sx != 0 and sy != 0)
                slider_ok = False
                if enemy_piece_type == 4:  # Queen
                    slider_ok = True
                elif enemy_piece_type == 3:  # Rook
                    slider_ok = is_orth
                elif enemy_piece_type == 2:  # Bishop
                    slider_ok = is_diag
                elif enemy_piece_type == 6:  # Right (rook|knight)
                    slider_ok = is_orth
                if (bb.occ_all & (1 << s)) and not (own_occ & (1 << s)) and slider_ok:
                    pin_mask_absolute |= (1 << first_sq)
                    # Allowed squares: anywhere on the ray segment between king and attacker, plus attacker
                    pin_ray = BETWEEN_RAYS[ksq][s] | (1 << s) | BETWEEN_RAYS[first_sq][ksq]
                    pin_ray_map[first_sq] = pin_ray
                break
            x += sx
            y += sy

    return checkers_bb, pin_mask_absolute, pin_ray_map


def format_legality_debug(bb: "BitboardState", stm_white: bool) -> Dict[str, Any]:
    """
    Returns a diagnostics dict for legality at side-to-move:
      - king square
      - checkers bitboard
      - pin mask and per-piece pin rays
    Safe to call from tests or debug tooling.
    """
    kbb = bb.WK if stm_white else bb.BK
    ksq = _pop_lsb_njit(kbb) if kbb else -1
    info: Dict[str, Any] = {"ksq": ksq, "checkers_bb": 0, "pin_mask": 0, "pin_ray_map": {}}
    if ksq >= 0:
        c, p, m = calculate_legality_masks(bb, ksq, stm_white)
        info["checkers_bb"] = c
        info["pin_mask"] = p
        info["pin_ray_map"] = m
    if DEBUG_LEGAL:
        try:
            print(f"[LEGAL] ksq={info['ksq']} checkers={bin(info['checkers_bb'])} pins={bin(info['pin_mask'])}")
        except Exception:
            pass
    return info

def _update_occ_from_parts(bbvals: List[int]) -> Tuple[int, int, int]:
    """
    Computes aggregate occupancy bitboards from individual piece bitboards.
    
    This helper function combines individual piece bitboards into aggregate
    occupancy masks. It separates white and black pieces, then combines them
    into a total occupancy mask.
    
    Args:
        bbvals: A list of exactly 14 integers representing piece bitboards in order:
            [WP, WN, WB, WR, WQ, WK, WRi, BP, BN, BB, BR, BQ, BK, BRi]
            where indices 0-6 are white pieces and 7-13 are black pieces.
    
    Returns:
        A tuple (occ_white, occ_black, occ_all) where:
        - occ_white: Bitboard with all white pieces (OR of indices 0-6)
        - occ_black: Bitboard with all black pieces (OR of indices 7-13)
        - occ_all: Combined occupancy of both colors (occ_white | occ_black)
    
    Notes:
        - Assumes the input list has exactly 14 elements
        - Used internally during move application to update occupancy after piece moves
    """
    occ_white = bbvals[0] | bbvals[1] | bbvals[2] | bbvals[3] | bbvals[4] | bbvals[5] | bbvals[6]
    occ_black = bbvals[7] | bbvals[8] | bbvals[9] | bbvals[10] | bbvals[11] | bbvals[12] | bbvals[13]
    return occ_white, occ_black, (occ_white | occ_black)


def apply_bb_move(bb: "BitboardState", m, zobrist: Optional["Zobrist"]) -> "BitboardState":
    """
    Applies a move to the current position, producing a new BitboardState.
    
    This function performs all necessary updates to apply a move: removing the
    moving piece from its source square, placing it (or a promoted piece) on the
    destination, removing any captured piece, updating occupancy masks, flipping
    side-to-move, and incrementally updating the Zobrist hash.
    
    Args:
        bb: The current BitboardState before the move is applied.
        m: A BBMove object describing the move:
            - from_sq: Source square index [0, 24]
            - to_sq: Destination square index [0, 24]
            - piece_type: Type of moving piece (0-6)
            - captured_type: Type of captured piece (-1 if none, 0-6 if capture)
            - promo: Promotion type (0 if none, 4 for queen promotion)
        zobrist: Optional Zobrist hasher for incremental hash updates. If None,
            hash updates are skipped and zkey remains unchanged.
    
    Returns:
        A new BitboardState representing the position after the move:
        - All piece bitboards updated (source cleared, destination set)
        - Captured piece removed if applicable
        - Promotion handled (pawn replaced with promoted piece)
        - Occupancy masks recalculated
        - side_to_move flipped (0 <-> 1)
        - zkey updated incrementally if zobrist is provided
    
    Notes:
        - Does not modify the input BitboardState (creates a new one)
        - Handles pawn promotion by replacing pawn with queen at destination
        - Incremental Zobrist updates toggle pieces at source/destination and side-to-move
    """
    stm_white = (bb.side_to_move == 0)
    # Copy all piece bitboards into list for easier updates
    parts = [
        bb.WP, bb.WN, bb.WB, bb.WR, bb.WQ, bb.WK, bb.WRi,
        bb.BP, bb.BN, bb.BB, bb.BR, bb.BQ, bb.BK, bb.BRi,
    ]
    if isinstance(m, int):
        from_sq, to_sq, piece_type, captured_type, promo = unpack_move(m)
    else:
        from_sq, to_sq = m.from_sq, m.to_sq
        piece_type, captured_type, promo = m.piece_type, m.captured_type, m.promo
    from_mask = 1 << from_sq
    to_mask = 1 << to_sq
    color_idx = 0 if stm_white else 1
    zkey = bb.zkey
    # Incremental Material+PST delta
    mg_delta = 0
    eg_delta = 0
    mover_color = "white" if stm_white else "black"
    opp_color = "black" if stm_white else "white"
    s = 1 if stm_white else -1
    mover_name = _IDX_TO_NAME.get(piece_type, "Pawn")
    def _pst_mg(name: str, sq: int, color: str) -> int:
        x, y = index_to_sq(sq)
        return PSTS_MG[color][name][y][x]
    def _pst_eg(name: str, sq: int, color: str) -> int:
        x, y = index_to_sq(sq)
        return PSTS_EG[color][name][y][x]

    # Remove captured piece if any
    if captured_type >= 0:
        opp_offset = 7 if stm_white else 0
        cap_idx = opp_offset + captured_type
        if parts[cap_idx] & to_mask:
            parts[cap_idx] ^= to_mask
            if zobrist:
                zkey = zobrist.toggle_by_indices(zkey, captured_type, 1 - color_idx, to_sq)
            cap_name = _IDX_TO_NAME.get(captured_type, "Pawn")
            cap_sign = 1 if stm_white else -1
            mg_delta += cap_sign * (PIECE_VALUES_MG.get(cap_name, 0) + _pst_mg(cap_name, to_sq, opp_color))
            eg_delta += cap_sign * (PIECE_VALUES_EG.get(cap_name, 0) + _pst_eg(cap_name, to_sq, opp_color))

    # Move the piece (and handle promotion replacement)
    my_offset = 0 if stm_white else 7
    src_idx = my_offset + piece_type
    # Remove from source
    parts[src_idx] ^= from_mask
    if zobrist:
        zkey = zobrist.toggle_by_indices(zkey, piece_type, color_idx, from_sq)
    # Eval: remove mover from source
    if mover_name != "King":
        mg_delta -= s * PIECE_VALUES_MG.get(mover_name, 0)
        eg_delta -= s * PIECE_VALUES_EG.get(mover_name, 0)
    mg_delta -= s * _pst_mg(mover_name, from_sq, mover_color)
    eg_delta -= s * _pst_eg(mover_name, from_sq, mover_color)
    # Destination: either same piece or promotion replacement
    if promo == 4 and piece_type == 0:
        # Promotion to Queen replaces pawn with queen at destination
        dst_idx = my_offset + 4
        parts[dst_idx] |= to_mask
        if zobrist:
            zkey = zobrist.toggle_by_indices(zkey, 4, color_idx, to_sq)
        # Eval: add promoted queen at destination
        mg_delta += s * (PIECE_VALUES_MG["Queen"] + _pst_mg("Queen", to_sq, mover_color))
        eg_delta += s * (PIECE_VALUES_EG["Queen"] + _pst_eg("Queen", to_sq, mover_color))
    else:
        parts[src_idx] |= to_mask
        if zobrist:
            zkey = zobrist.toggle_by_indices(zkey, piece_type, color_idx, to_sq)
        # Eval: add moved piece at destination
        if mover_name != "King":
            mg_delta += s * PIECE_VALUES_MG.get(mover_name, 0)
            eg_delta += s * PIECE_VALUES_EG.get(mover_name, 0)
        mg_delta += s * _pst_mg(mover_name, to_sq, mover_color)
        eg_delta += s * _pst_eg(mover_name, to_sq, mover_color)

    occ_w, occ_b, occ_all = _update_occ_from_parts(parts)
    # Flip side-to-move
    if zobrist:
        zkey = zobrist.toggle_black_to_move(zkey)

    return BitboardState(
        WP=parts[0], WN=parts[1], WB=parts[2], WR=parts[3], WQ=parts[4], WK=parts[5], WRi=parts[6],
        BP=parts[7], BN=parts[8], BB=parts[9], BR=parts[10], BQ=parts[11], BK=parts[12], BRi=parts[13],
        occ_white=occ_w, occ_black=occ_b, occ_all=occ_all,
        side_to_move=(1 if stm_white else 0),
        zkey=zkey,
        mg_mat_pst=bb.mg_mat_pst + mg_delta,
        eg_mat_pst=bb.eg_mat_pst + eg_delta,
    )


@dataclass(frozen=True, slots=True)
class UndoInfoInplace:
    """
    Minimal undo payload for in-place make/unmake.
    Restores occupancy, pieces, side-to-move and zobrist hash.
    """
    from_mask: int
    to_mask: int
    moved_type: int
    captured_type: int  # -1 if none
    promo: int          # 0 or 4 (queen)
    prev_side_to_move: int
    prev_zkey: int
    prev_mg_mat_pst: int
    prev_eg_mat_pst: int


_PIECE_FIELDS_WHITE = ("WP", "WN", "WB", "WR", "WQ", "WK", "WRi")
_PIECE_FIELDS_BLACK = ("BP", "BN", "BB", "BR", "BQ", "BK", "BRi")


def make_move_inplace(bb: "BitboardState", m, zobrist: Optional["Zobrist"]) -> "UndoInfoInplace":
    """
    Applies a move by mutating bb in-place. Returns an undo object to restore state.
    Mirrors apply_bb_move semantics but without allocations.
    """
    stm_white = (bb.side_to_move == 0)
    if isinstance(m, int):
        from_sq, to_sq, piece_type, captured_type, promo = unpack_move(m)
    else:
        from_sq, to_sq = m.from_sq, m.to_sq
        piece_type, captured_type, promo = m.piece_type, m.captured_type, m.promo
    from_mask = 1 << from_sq
    to_mask = 1 << to_sq
    color_idx = 0 if stm_white else 1
    prev_side = bb.side_to_move
    prev_zkey = bb.zkey
    prev_mg = bb.mg_mat_pst
    prev_eg = bb.eg_mat_pst
    # Eval delta
    mg_delta = 0
    eg_delta = 0
    mover_color = "white" if stm_white else "black"
    opp_color = "black" if stm_white else "white"
    s = 1 if stm_white else -1
    mover_name = _IDX_TO_NAME.get(piece_type, "Pawn")
    def _pst_mg(name: str, sq: int, color: str) -> int:
        x, y = index_to_sq(sq)
        return PSTS_MG[color][name][y][x]
    def _pst_eg(name: str, sq: int, color: str) -> int:
        x, y = index_to_sq(sq)
        return PSTS_EG[color][name][y][x]

    # Remove captured piece (if any)
    if captured_type >= 0:
        opp_field = _PIECE_FIELDS_BLACK[captured_type] if stm_white else _PIECE_FIELDS_WHITE[captured_type]
        setattr(bb, opp_field, getattr(bb, opp_field) ^ to_mask)
        if stm_white:
            bb.occ_black ^= to_mask
        else:
            bb.occ_white ^= to_mask
        if zobrist:
            bb.zkey = zobrist.toggle_by_indices(bb.zkey, captured_type, 1 - color_idx, to_sq)
        cap_name = _IDX_TO_NAME.get(captured_type, "Pawn")
        cap_sign = 1 if stm_white else -1
        mg_delta += cap_sign * (PIECE_VALUES_MG.get(cap_name, 0) + _pst_mg(cap_name, to_sq, opp_color))
        eg_delta += cap_sign * (PIECE_VALUES_EG.get(cap_name, 0) + _pst_eg(cap_name, to_sq, opp_color))

    # Move the piece
    mover_field = _PIECE_FIELDS_WHITE[piece_type] if stm_white else _PIECE_FIELDS_BLACK[piece_type]
    # Remove from source
    setattr(bb, mover_field, getattr(bb, mover_field) ^ from_mask)
    if zobrist:
        bb.zkey = zobrist.toggle_by_indices(bb.zkey, piece_type, color_idx, from_sq)
    # Eval removal from source
    if mover_name != "King":
        mg_delta -= s * PIECE_VALUES_MG.get(mover_name, 0)
        eg_delta -= s * PIECE_VALUES_EG.get(mover_name, 0)
    mg_delta -= s * _pst_mg(mover_name, from_sq, mover_color)
    eg_delta -= s * _pst_eg(mover_name, from_sq, mover_color)
    # Place at destination (promotion replaces pawn with queen at destination)
    if promo == 4 and piece_type == 0:
        dst_field = _PIECE_FIELDS_WHITE[4] if stm_white else _PIECE_FIELDS_BLACK[4]
        setattr(bb, dst_field, getattr(bb, dst_field) | to_mask)
        if zobrist:
            bb.zkey = zobrist.toggle_by_indices(bb.zkey, 4, color_idx, to_sq)
        mg_delta += s * (PIECE_VALUES_MG["Queen"] + _pst_mg("Queen", to_sq, mover_color))
        eg_delta += s * (PIECE_VALUES_EG["Queen"] + _pst_eg("Queen", to_sq, mover_color))
    else:
        setattr(bb, mover_field, getattr(bb, mover_field) | to_mask)
        if zobrist:
            bb.zkey = zobrist.toggle_by_indices(bb.zkey, piece_type, color_idx, to_sq)
        if mover_name != "King":
            mg_delta += s * PIECE_VALUES_MG.get(mover_name, 0)
            eg_delta += s * PIECE_VALUES_EG.get(mover_name, 0)
        mg_delta += s * _pst_mg(mover_name, to_sq, mover_color)
        eg_delta += s * _pst_eg(mover_name, to_sq, mover_color)

    # Update occupancy
    if stm_white:
        bb.occ_white ^= from_mask
        bb.occ_white |= to_mask
    else:
        bb.occ_black ^= from_mask
        bb.occ_black |= to_mask
    bb.occ_all = bb.occ_white | bb.occ_black
    # Update incremental eval
    bb.mg_mat_pst = prev_mg + mg_delta
    bb.eg_mat_pst = prev_eg + eg_delta

    # Flip side to move
    bb.side_to_move = 1 if stm_white else 0
    if zobrist:
        bb.zkey = zobrist.toggle_black_to_move(bb.zkey)

    return UndoInfoInplace(
        from_mask=from_mask,
        to_mask=to_mask,
        moved_type=piece_type,
        captured_type=captured_type,
        promo=promo,
        prev_side_to_move=prev_side,
        prev_zkey=prev_zkey,
        prev_mg_mat_pst=prev_mg,
        prev_eg_mat_pst=prev_eg,
    )


def unmake_move_inplace(bb: "BitboardState", undo: "UndoInfoInplace", zobrist: Optional["Zobrist"]) -> None:
    """
    Restores bb to the exact state before the corresponding make_move_inplace call.
    """
    # Restore side and hash first (hash covers all piece toggles)
    bb.side_to_move = undo.prev_side_to_move
    bb.zkey = undo.prev_zkey
    # Restore eval
    bb.mg_mat_pst = undo.prev_mg_mat_pst
    bb.eg_mat_pst = undo.prev_eg_mat_pst

    stm_white_prev = (undo.prev_side_to_move == 0)
    # Restore mover piece
    if undo.promo == 4 and undo.moved_type == 0:
        # Remove promoted queen from destination; restore pawn at source
        q_field = _PIECE_FIELDS_WHITE[4] if stm_white_prev else _PIECE_FIELDS_BLACK[4]
        setattr(bb, q_field, getattr(bb, q_field) ^ undo.to_mask)
        pawn_field = _PIECE_FIELDS_WHITE[0] if stm_white_prev else _PIECE_FIELDS_BLACK[0]
        setattr(bb, pawn_field, getattr(bb, pawn_field) | undo.from_mask)
    else:
        mover_field = _PIECE_FIELDS_WHITE[undo.moved_type] if stm_white_prev else _PIECE_FIELDS_BLACK[undo.moved_type]
        setattr(bb, mover_field, getattr(bb, mover_field) ^ undo.to_mask)
        setattr(bb, mover_field, getattr(bb, mover_field) | undo.from_mask)

    # Restore captured piece (if any)
    if undo.captured_type >= 0:
        opp_field = _PIECE_FIELDS_BLACK[undo.captured_type] if stm_white_prev else _PIECE_FIELDS_WHITE[undo.captured_type]
        setattr(bb, opp_field, getattr(bb, opp_field) | undo.to_mask)

    # Restore occupancy
    if stm_white_prev:
        bb.occ_white ^= undo.to_mask
        bb.occ_white |= undo.from_mask
        if undo.captured_type >= 0:
            bb.occ_black |= undo.to_mask
    else:
        bb.occ_black ^= undo.to_mask
        bb.occ_black |= undo.from_mask
        if undo.captured_type >= 0:
            bb.occ_white |= undo.to_mask
    bb.occ_all = bb.occ_white | bb.occ_black


@dataclass(frozen=True, slots=True)
class UndoInfo:
    parent: BitboardState


def make_move(bb: "BitboardState", m: "BBMove", zobrist: Optional["Zobrist"]) -> Tuple["BitboardState", "UndoInfo"]:
    """
    Applies a move and returns the new state with undo information for restoration.
    
    This function is the main entry point for making moves in the search tree.
    It applies the move to create a child node and stores the parent state in
    an UndoInfo object, allowing fast restoration via unmake_move().
    
    Args:
        bb: The current BitboardState before the move.
        m: The BBMove to apply.
        zobrist: Optional Zobrist hasher for hash updates.
    
    Returns:
        A tuple (child_state, undo_info) where:
        - child_state: New BitboardState after applying the move
        - undo_info: UndoInfo object containing the parent state for restoration
    
    Notes:
        - Used during search tree traversal
        - The undo mechanism allows efficient move unmaking without copying entire state
        - Parent state is stored in UndoInfo for O(1) restoration
    """
    child = apply_bb_move(bb, m, zobrist)
    return child, UndoInfo(parent=bb)


def unmake_move(_child: "BitboardState", undo: "UndoInfo") -> "BitboardState":
    """
    Restores the parent state from before a move was applied.
    
    This function implements the undo mechanism for search tree traversal.
    It returns the parent state that was stored when the move was made,
    effectively "unmaking" the move without needing to reverse all bitboard operations.
    
    Args:
        _child: The current BitboardState after the move (unused, kept for API consistency).
        undo: UndoInfo object containing the parent state stored during make_move().
    
    Returns:
        The BitboardState from before the move was applied (the parent state).
    
    Notes:
        - O(1) operation - just returns the stored parent state
        - Used during alpha-beta search to restore position after exploring a move
        - The _child parameter is unused but kept for API consistency
    """
    return undo.parent


def _gen_sliding_moves(from_sq: int, occ_all: int, own_occ: int, attack_fn) -> Tuple[int, int]:
    """
    Generates quiet and capture move masks for a sliding piece (rook or bishop).
    
    This helper function computes the legal moves for a sliding piece by first
    getting all attacked squares, then filtering out squares occupied by friendly
    pieces, and finally separating quiet moves (empty squares) from captures
    (opponent-occupied squares).
    
    Args:
        from_sq: Source square index [0, 24] where the sliding piece is located.
        occ_all: Bitboard of all occupied squares (both colors).
        own_occ: Bitboard of squares occupied by friendly pieces (to exclude).
        attack_fn: Function to compute attacks, either _get_rook_attacks or
            _get_bishop_attacks. Must accept (sq, occ) and return attack bitboard.
    
    Returns:
        A tuple (quiet_mask, capture_mask) where:
        - quiet_mask: Bitboard of empty squares the piece can move to
        - capture_mask: Bitboard of opponent-occupied squares the piece can capture
    
    Notes:
        - Attacks are computed with respect to occ_all (includes blockers)
        - Friendly squares are excluded from both masks
        - Used internally by generate_legal_moves() for rooks, bishops, and queens
    """
    attacks = attack_fn(from_sq, occ_all)
    legal = attacks & (~own_occ)
    # Quiet squares are those not occupied
    quiet = legal & (~occ_all)
    captures = legal & occ_all
    return quiet, captures


def generate_legal_moves(bb: "BitboardState", captures_only: bool = False, legality_info: Optional[Tuple] = None) -> List[int]:
    """
    Generates all legal moves for the side-to-move in the current position.
    
    This is the main move generation function that produces a complete list of
    legal moves, respecting check, pins, and other legality constraints. It uses
    efficient bitboard operations and precomputed legality masks to filter moves.
    
    Args:
        bb: The current BitboardState representing the position.
        captures_only: If True, only generates capture moves (used for quiescence
            search). If False, generates all legal moves including quiet moves.
        legality_info: Optional tuple (checkers_bb, pin_mask, pin_ray_map) from
            calculate_legality_masks. If provided, skips recalculating legality.
    
    Returns:
        A list of packed-int moves. Each move packs:
        - from_sq, to_sq, piece_type, captured_type, promo(0 or 4)
    
    Notes:
        - Handles check by restricting moves to those that escape or block check
        - Handles double-check by only allowing king moves
        - Respects pins by restricting pinned pieces to move along the pin ray
        - Generates moves for all piece types: pawns, knights, bishops, rooks,
          queens, kings, and the custom "Right" piece
        - Uses bitboard operations for efficient move generation
        - King moves are validated to ensure they don't move into check
    """
    moves: List[int] = []
    stm_white = (bb.side_to_move == 0)
    own_occ = bb.occ_white if stm_white else bb.occ_black
    opp_occ = bb.occ_black if stm_white else bb.occ_white
    occ_all = bb.occ_all

    kbb = bb.WK if stm_white else bb.BK
    if kbb == 0:
        return moves
    ksq = _pop_lsb_njit(kbb)

    if legality_info:
        checkers_bb, pin_mask, pin_ray_map = legality_info
    elif ksq >= 0:
        checkers_bb, pin_mask, pin_ray_map = calculate_legality_masks(bb, ksq, stm_white)
    else:
        checkers_bb, pin_mask, pin_ray_map = 0, 0, {}  # Should not happen if kbb != 0
    is_in_check = (checkers_bb != 0)
    is_double_check = (_count_bits(checkers_bb) > 1)

    # Determine target mask for non-king moves
    target_mask = ALL25
    if is_in_check:
        if is_double_check:
            target_mask = 0
        else:
            checker_sq = _pop_lsb_njit(checkers_bb)
            target_mask = (1 << checker_sq)
            # If checker is a slider along the line, add blocking squares
            ctype = _get_piece_type_at_square(bb, color_white=(not stm_white), sq=checker_sq)
            kx, ky = index_to_sq(ksq)
            cx, cy = index_to_sq(checker_sq)
            same_file = (kx == cx)
            same_rank = (ky == cy)
            same_diag = (abs(kx - cx) == abs(ky - cy))
            if ctype == 4:
                target_mask |= BETWEEN_RAYS[ksq][checker_sq]
            elif ctype == 3 and (same_file or same_rank):
                target_mask |= BETWEEN_RAYS[ksq][checker_sq]
            elif ctype == 2 and same_diag:
                target_mask |= BETWEEN_RAYS[ksq][checker_sq]
            elif ctype == 6 and (same_file or same_rank):
                target_mask |= BETWEEN_RAYS[ksq][checker_sq]

    # King moves (special case; only legal option under double-check)
    king_targets = _KING_MOVES[ksq] & (~own_occ)
    for to_sq in _iter_set_bits(king_targets):
        to_mask = 1 << to_sq
        # New occupancy after moving king (and capturing if applicable)
        occ_prime = (occ_all & ~ (1 << ksq) & ~to_mask) | to_mask
        if attackers_to_square(bb, to_sq, by_white=(not stm_white), occ_override=occ_prime):
            continue
        if captures_only and not (opp_occ & to_mask):
            continue
        cap_type = _get_piece_type_at_square(bb, not stm_white, to_sq) if (opp_occ & to_mask) else -1
        moves.append(pack_move(ksq, to_sq, 5, cap_type, 0))

    if is_double_check:
        return moves

    # Pre-fetch opponent piece list for fast capture-typing
    opp_bbs = (
        (bb.BP, bb.BN, bb.BB, bb.BR, bb.BQ, bb.BK, bb.BRi)
        if stm_white
        else (bb.WP, bb.WN, bb.WB, bb.WR, bb.WQ, bb.WK, bb.WRi)
    )

    # Non-king moves
    # Pawns using bit-shifts
    P = bb.WP if stm_white else bb.BP
    empty = (~occ_all) & ALL25
    if stm_white:
        push = (P >> 5) & empty
        capL = ((P & NOT_FILE_0) >> 6) & opp_occ
        capR = ((P & NOT_FILE_4) >> 4) & opp_occ
    else:
        push = ((P << 5) & ALL25) & empty
        capL = ((P & NOT_FILE_4) << 6) & ALL25 & opp_occ
        capR = ((P & NOT_FILE_0) << 4) & ALL25 & opp_occ
    if captures_only:
        push = 0
    push &= target_mask
    capL &= target_mask
    capR &= target_mask

    # Emit pawn pushes
    pp = push
    while pp:
        to_sq = (pp & -pp).bit_length() - 1
        pp ^= (1 << to_sq)
        from_sq = (to_sq + 5) if stm_white else (to_sq - 5)
        if (pin_mask & (1 << from_sq)) and not (pin_ray_map.get(from_sq, 0) & (1 << to_sq)):
            continue
        promo = 4 if ((RANK_0_MASK & (1 << to_sq)) if stm_white else (RANK_4_MASK & (1 << to_sq))) else 0
        moves.append(pack_move(from_sq, to_sq, 0, -1, promo))

    # Pawn captures (left)
    cl = capL
    while cl:
        to_sq = (cl & -cl).bit_length() - 1
        cl ^= (1 << to_sq)
        from_sq = (to_sq + 6) if stm_white else (to_sq - 6)
        if (pin_mask & (1 << from_sq)) and not (pin_ray_map.get(from_sq, 0) & (1 << to_sq)):
            continue
        promo = 4 if ((RANK_0_MASK & (1 << to_sq)) if stm_white else (RANK_4_MASK & (1 << to_sq))) else 0
        cap_type = _get_piece_type_at_square(bb, not stm_white, to_sq)
        moves.append(pack_move(from_sq, to_sq, 0, cap_type, promo))

    # Pawn captures (right)
    cr = capR
    while cr:
        to_sq = (cr & -cr).bit_length() - 1
        cr ^= (1 << to_sq)
        from_sq = (to_sq + 4) if stm_white else (to_sq - 4)
        if (pin_mask & (1 << from_sq)) and not (pin_ray_map.get(from_sq, 0) & (1 << to_sq)):
            continue
        promo = 4 if ((RANK_0_MASK & (1 << to_sq)) if stm_white else (RANK_4_MASK & (1 << to_sq))) else 0
        cap_type = _get_piece_type_at_square(bb, not stm_white, to_sq)
        moves.append(pack_move(from_sq, to_sq, 0, cap_type, promo))

    # Knights
    N = bb.WN if stm_white else bb.BN
    for from_sq in _iter_set_bits(N):
        targets = _KNIGHT_MOVES[from_sq] & (~own_occ) & target_mask
        if pin_mask & (1 << from_sq):
            targets &= pin_ray_map.get(from_sq, 0)
        # Generate captures first
        for cap_type, opp_bb in enumerate(opp_bbs):
            cap_mask = targets & opp_bb
            for to_sq in _iter_set_bits(cap_mask):
                moves.append(pack_move(from_sq, to_sq, 1, cap_type, 0))
        # Generate quiet moves
        if not captures_only:
            quiet_mask = targets & (~occ_all)
            for to_sq in _iter_set_bits(quiet_mask):
                moves.append(pack_move(from_sq, to_sq, 1, -1, 0))

    # Bishops
    B = bb.WB if stm_white else bb.BB
    for from_sq in _iter_set_bits(B):
        targets = _get_bishop_attacks(from_sq, occ_all) & (~own_occ) & target_mask
        if pin_mask & (1 << from_sq):
            targets &= pin_ray_map.get(from_sq, 0)
        # Generate captures first
        for cap_type, opp_bb in enumerate(opp_bbs):
            cap_mask = targets & opp_bb
            for to_sq in _iter_set_bits(cap_mask):
                moves.append(pack_move(from_sq, to_sq, 2, cap_type, 0))
        # Generate quiet moves
        if not captures_only:
            quiet_mask = targets & (~occ_all)
            for to_sq in _iter_set_bits(quiet_mask):
                moves.append(pack_move(from_sq, to_sq, 2, -1, 0))

    # Rooks
    R = bb.WR if stm_white else bb.BR
    for from_sq in _iter_set_bits(R):
        targets = _get_rook_attacks(from_sq, occ_all) & (~own_occ) & target_mask
        if pin_mask & (1 << from_sq):
            targets &= pin_ray_map.get(from_sq, 0)
        # Generate captures first
        for cap_type, opp_bb in enumerate(opp_bbs):
            cap_mask = targets & opp_bb
            for to_sq in _iter_set_bits(cap_mask):
                moves.append(pack_move(from_sq, to_sq, 3, cap_type, 0))
        # Generate quiet moves
        if not captures_only:
            quiet_mask = targets & (~occ_all)
            for to_sq in _iter_set_bits(quiet_mask):
                moves.append(pack_move(from_sq, to_sq, 3, -1, 0))

    # Queens
    Q = bb.WQ if stm_white else bb.BQ
    for from_sq in _iter_set_bits(Q):
        targets = (_get_rook_attacks(from_sq, occ_all) | _get_bishop_attacks(from_sq, occ_all)) & (~own_occ) & target_mask
        if pin_mask & (1 << from_sq):
            targets &= pin_ray_map.get(from_sq, 0)
        # Generate captures first
        for cap_type, opp_bb in enumerate(opp_bbs):
            cap_mask = targets & opp_bb
            for to_sq in _iter_set_bits(cap_mask):
                moves.append(pack_move(from_sq, to_sq, 4, cap_type, 0))
        # Generate quiet moves
        if not captures_only:
            quiet_mask = targets & (~occ_all)
            for to_sq in _iter_set_bits(quiet_mask):
                moves.append(pack_move(from_sq, to_sq, 4, -1, 0))

    # Right (rook | knight)
    Ri = bb.WRi if stm_white else bb.BRi
    for from_sq in _iter_set_bits(Ri):
        r_att = _get_rook_attacks(from_sq, occ_all)
        n_att = _KNIGHT_MOVES[from_sq]
        targets = (r_att | n_att) & (~own_occ) & target_mask
        if pin_mask & (1 << from_sq):
            targets &= pin_ray_map.get(from_sq, 0)
        # Generate captures first
        for cap_type, opp_bb in enumerate(opp_bbs):
            cap_mask = targets & opp_bb
            for to_sq in _iter_set_bits(cap_mask):
                moves.append(pack_move(from_sq, to_sq, 6, cap_type, 0))
        # Generate quiet moves
        if not captures_only:
            quiet_mask = targets & (~occ_all)
            for to_sq in _iter_set_bits(quiet_mask):
                moves.append(pack_move(from_sq, to_sq, 6, -1, 0))

    return moves

# (obsolete generate_all_moves removed in favor of generate_legal_moves)


def _count_bits(x: int) -> int:
    """
    Counts the number of set bits (population count) in an integer bitboard.
    
    This function uses Python's built-in bit_count() method to efficiently
    count the number of 1-bits in a bitboard, which corresponds to counting
    the number of pieces or squares represented by the bitboard.
    
    Args:
        x: An integer bitboard where each set bit represents a piece or square.
    
    Returns:
        The number of set bits in x (population count). For a 5x5 board bitboard,
        this ranges from 0 to 25.
    
    Example:
        >>> _count_bits(0b10110)
        3  # Three bits are set
    
    Notes:
        - Uses Python 3.10+ bit_count() method for optimal performance
        - Equivalent to counting pieces on a bitboard or squares in a mask
    """
    return int(x.bit_count())


def _piece_on_square_color(bb: "BitboardState", sq: int) -> int:
    """
    Determines the color of the piece (if any) occupying a specific square.
    
    This function checks both white and black occupancy bitboards to determine
    which color (if any) has a piece on the given square. It returns a simple
    integer code indicating the result.
    
    Args:
        bb: BitboardState containing all piece bitboards and occupancy information.
        sq: Square index in the range [0, 24] to check.
    
    Returns:
        An integer code:
        - 1: Square is occupied by a white piece
        - -1: Square is occupied by a black piece
        - 0: Square is empty (no piece of either color)
    
    Notes:
        - Checks occ_white and occ_black bitboards for efficiency
        - Used in evaluation functions to determine piece colors on squares
    """
    mask = 1 << sq
    if bb.occ_white & mask:
        return 1
    if bb.occ_black & mask:
        return -1
    return 0


def _phase_from_bb(bb: "BitboardState") -> int:
    """
    Calculates the game phase value based on remaining pieces on the board.
    
    The game phase is a measure of how far the game has progressed from opening
    to endgame. It's computed by summing phase values for all non-king pieces.
    Higher phase values indicate more pieces (middlegame), lower values indicate
    fewer pieces (endgame). This is used to interpolate between middlegame and
    endgame evaluation terms.
    
    Args:
        bb: BitboardState containing all piece bitboards.
    
    Returns:
        An integer phase value in the range [1, MAX_PHASE] (typically 1-16).
        - Higher values (closer to MAX_PHASE): More pieces, middlegame phase
        - Lower values (closer to 1): Fewer pieces, endgame phase
    
    Notes:
        - Phase values per piece: Pawn=0, Knight=1, Bishop=1, Rook=2, Right=2, Queen=4, King=0
        - Clamped to [1, MAX_PHASE] to avoid division by zero and overflow
        - Used to blend middlegame and endgame evaluation scores
    """
    total = 0
    # White
    total += PHASE_VALUES["Pawn"] * bb.WP.bit_count()
    total += PHASE_VALUES["Knight"] * bb.WN.bit_count()
    total += PHASE_VALUES["Bishop"] * bb.WB.bit_count()
    total += PHASE_VALUES["Rook"] * bb.WR.bit_count()
    total += PHASE_VALUES["Right"] * bb.WRi.bit_count()
    total += PHASE_VALUES["Queen"] * bb.WQ.bit_count()
    # Black
    total += PHASE_VALUES["Pawn"] * bb.BP.bit_count()
    total += PHASE_VALUES["Knight"] * bb.BN.bit_count()
    total += PHASE_VALUES["Bishop"] * bb.BB.bit_count()
    total += PHASE_VALUES["Rook"] * bb.BR.bit_count()
    total += PHASE_VALUES["Right"] * bb.BRi.bit_count()
    total += PHASE_VALUES["Queen"] * bb.BQ.bit_count()
    return max(1, min(total, MAX_PHASE))

def _pawn_attacks_mask(is_white: bool, pawns_bb: int) -> int:
    """
    Computes a bitboard of all squares attacked by pawns of a given color.
    
    This function generates the attack mask for all pawns of the specified color.
    White pawns attack diagonally forward (up-left and up-right), while black
    pawns attack diagonally backward (down-left and down-right).
    
    Args:
        is_white: True for white pawns, False for black pawns.
        pawns_bb: Bitboard where each set bit represents a pawn of the specified color.
            Square indices are in the range [0, 24] for a 5x5 board.
    
    Returns:
        A 25-bit bitboard where each set bit represents a square attacked by
        at least one pawn of the specified color. The mask includes all squares
        that are diagonally adjacent to any pawn in the forward direction.
    
    Notes:
        - White pawns at (x, y) attack (x-1, y-1) and (x+1, y-1) if in bounds
        - Black pawns at (x, y) attack (x-1, y+1) and (x+1, y+1) if in bounds
        - Used for mobility calculations and attack pattern evaluation
    """
    mask = 0
    table = WHITE_PAWN_ATTACKS if is_white else BLACK_PAWN_ATTACKS
    bb = pawns_bb
    while bb:
        lsb = bb & -bb
        idx = lsb.bit_length() - 1
        mask |= table[idx]
        bb ^= lsb
    return mask

def evaluate_bb_state(bb: "BitboardState") -> float:
    """
    Evaluates a chess position and returns a score from the side-to-move's perspective.
    
    This is the main evaluation function that computes a comprehensive score for a
    position. It evaluates material, piece-square tables, mobility, pawn structure,
    king safety, and various positional features. The score is returned from the
    perspective of the side-to-move (positive favors side-to-move, negative favors opponent).
    
    Args:
        bb: BitboardState representing the position to evaluate.
    
    Returns:
        A float score in centipawns (1/100 of a pawn) from side-to-move's perspective:
        - Positive values: Position favors the side-to-move
        - Negative values: Position favors the opponent
        - Large positive values (>900000): Mate in favor of side-to-move
        - Large negative values (<-900000): Mate against side-to-move
    
    Notes:
        Evaluation components (in order of computation):
        1. Material: Piece values (MG and EG scales)
        2. Piece-Square Tables: Positional bonuses based on square location
        3. Mobility: Number of squares pieces can move to
        4. Bishop pair: Bonus for having two bishops
        5. Center control: Bonus for controlling central squares
        6. Pawn structure: Doubled, isolated, passed, connected passers, backward pawns
        7. Open/semi-open files: For rooks and right pieces
        8. King safety: Attack patterns, pawn shield, open files to king
        9. Endgame drives: King proximity, driving opponent king to edge/corner (KRK, KQK)
        10. Rooks on 7th rank: Bonus for rooks on opponent's back rank
        11. Knight outposts: Protected knights that can't be attacked by pawns
        12. Bad bishops: Penalties for bishops blocked by own pawns
        
        The final score blends middlegame and endgame evaluations based on game phase.
        Phase is calculated from remaining pieces (more pieces = middlegame, fewer = endgame).
    """
    # --- Cache all BB attributes to local variables ---
    WP, WN, WB, WR, WQ, WK, WRi = bb.WP, bb.WN, bb.WB, bb.WR, bb.WQ, bb.WK, bb.WRi
    BP, BN, BB, BR, BQ, BK, BRi = bb.BP, bb.BN, bb.BB, bb.BR, bb.BQ, bb.BK, bb.BRi
    occ_all = bb.occ_all
    occ_white = bb.occ_white
    occ_black = bb.occ_black
    # --- End cache ---
    
    # === 1. Start with Incremental Material + PST scores ===
    # These are already calculated and updated during make/unmake
    mg = bb.mg_mat_pst
    eg = bb.eg_mat_pst
    
    # === 2. Single Pass: Add Mobility & Other Features ===
    # We still loop over pieces, but *only* for mobility,
    # king safety features, etc. (NOT material or PSTs)
    
    # Piece counts (still needed for bishop pair, etc.)
    wp_count = _count_bits(WP)
    wn_count = _count_bits(WN)
    wb_count = _count_bits(WB)
    wr_count = _count_bits(WR)
    wri_count = _count_bits(WRi)
    wq_count = _count_bits(WQ)
    
    bp_count = _count_bits(BP)
    bn_count = _count_bits(BN)
    bb_count = _count_bits(BB)
    br_count = _count_bits(BR)
    bri_count = _count_bits(BRi)
    bq_count = _count_bits(BQ)
    
    w_mob = 0
    b_mob = 0

    # Pre-calculate masks and shared data
    occ = occ_all
    center_set = set(CENTER_SQUARES)
    w_seventh_mask = RANK_MASKS[1]
    b_seventh_mask = RANK_MASKS[3]
    
    light_mask = 0
    dark_mask = 0
    for ry in range(5):
        for rx in range(5):
            sqi = square_index(rx, ry)
            if ((rx + ry) & 1) == 0:
                light_mask |= 1 << sqi
            else:
                dark_mask |= 1 << sqi
    
    # For "Bad Bishop" calculations
    wp_light_count = _count_bits(WP & light_mask)
    wp_dark_count = _count_bits(WP & dark_mask)
    bp_light_count = _count_bits(BP & light_mask)
    bp_dark_count = _count_bits(BP & dark_mask)
    
    w_bishops_light = 0
    w_bishops_dark = 0
    b_bishops_light = 0
    b_bishops_dark = 0
    
    w_rooks_on_7th = 0
    b_rooks_on_7th = 0

    # --- White Pieces Single Pass ---
    # (Pawns: no mobility loop needed, handled separately)
    
    for sq in _iter_set_bits(WN):
        x, y = index_to_sq(sq)
        w_mob += MOBILITY_WEIGHTS["Knight"] * _count_bits(_KNIGHT_MOVES[sq] & ~occ_white)
        # Knight Outpost
        if is_white_knight_outpost(bb, sq):
            bonus = EVAL_KNIGHT_OUTPOST
            if (x, y) in center_set:
                bonus += EVAL_KNIGHT_OUTPOST_CENTER_BONUS
            mg += bonus
            eg += bonus

    for sq in _iter_set_bits(WB):
        x, y = index_to_sq(sq)
        w_mob += MOBILITY_WEIGHTS["Bishop"] * _count_bits(_get_bishop_attacks(sq, occ) & ~occ_white)
        # Bad Bishop tracker
        if light_mask & (1 << sq):
            w_bishops_light += 1
        else:
            w_bishops_dark += 1

    for sq in _iter_set_bits(WR):
        x, y = index_to_sq(sq)
        w_mob += MOBILITY_WEIGHTS["Rook"] * _count_bits(_get_rook_attacks(sq, occ) & ~occ_white)
        # Rook on 7th
        if w_seventh_mask & (1 << sq):
            w_rooks_on_7th += 1
            
    for sq in _iter_set_bits(WRi):
        x, y = index_to_sq(sq)
        right_att = _get_rook_attacks(sq, occ) | _KNIGHT_MOVES[sq]
        w_mob += MOBILITY_WEIGHTS["Right"] * _count_bits(right_att & ~occ_white)
        # Rook on 7th
        if w_seventh_mask & (1 << sq):
            w_rooks_on_7th += 1

    for sq in _iter_set_bits(WQ):
        x, y = index_to_sq(sq)
        w_mob += MOBILITY_WEIGHTS["Queen"] * _count_bits(
            (_get_bishop_attacks(sq, occ) | _get_rook_attacks(sq, occ)) & ~occ_white
        )

    for sq in _iter_set_bits(WK):
        x, y = index_to_sq(sq)
        w_mob += MOBILITY_WEIGHTS["King"] * _count_bits(_KING_MOVES[sq] & ~occ_white)

    # --- Black Pieces Single Pass ---
    # (Pawns: no mobility loop needed, handled separately)
        
    for sq in _iter_set_bits(BN):
        x, y = index_to_sq(sq)
        b_mob += MOBILITY_WEIGHTS["Knight"] * _count_bits(_KNIGHT_MOVES[sq] & ~occ_black)
        # Knight Outpost
        if is_black_knight_outpost(bb, sq):
            bonus = EVAL_KNIGHT_OUTPOST
            if (x, y) in center_set:
                bonus += EVAL_KNIGHT_OUTPOST_CENTER_BONUS
            mg -= bonus  # Subtract bonus for black
            eg -= bonus
            
    for sq in _iter_set_bits(BB):
        x, y = index_to_sq(sq)
        b_mob += MOBILITY_WEIGHTS["Bishop"] * _count_bits(_get_bishop_attacks(sq, occ) & ~occ_black)
        # Bad Bishop tracker
        if light_mask & (1 << sq):
            b_bishops_light += 1
        else:
            b_bishops_dark += 1

    for sq in _iter_set_bits(BR):
        x, y = index_to_sq(sq)
        b_mob += MOBILITY_WEIGHTS["Rook"] * _count_bits(_get_rook_attacks(sq, occ) & ~occ_black)
        # Rook on 7th
        if b_seventh_mask & (1 << sq):
            b_rooks_on_7th += 1
            
    for sq in _iter_set_bits(BRi):
        x, y = index_to_sq(sq)
        right_att = _get_rook_attacks(sq, occ) | _KNIGHT_MOVES[sq]
        b_mob += MOBILITY_WEIGHTS["Right"] * _count_bits(right_att & ~occ_black)
        # Rook on 7th
        if b_seventh_mask & (1 << sq):
            b_rooks_on_7th += 1

    for sq in _iter_set_bits(BQ):
        x, y = index_to_sq(sq)
        b_mob += MOBILITY_WEIGHTS["Queen"] * _count_bits(
            (_get_bishop_attacks(sq, occ) | _get_rook_attacks(sq, occ)) & ~occ_black
        )

    for sq in _iter_set_bits(BK):
        x, y = index_to_sq(sq)
        b_mob += MOBILITY_WEIGHTS["King"] * _count_bits(_KING_MOVES[sq] & ~occ_black)
        
    # --- Pawn Mobility (handled last) ---
    w_mob += MOBILITY_WEIGHTS["Pawn"] * _count_bits(_pawn_attacks_mask(True, WP) & ~occ_white)
    b_mob += MOBILITY_WEIGHTS["Pawn"] * _count_bits(_pawn_attacks_mask(False, BP) & ~occ_black)

    # --- 3. Apply Accumulated Scores & Other Features ---
    
    # Add Mobility scores
    mob_diff = w_mob - b_mob
    mg += EVAL_MOBILITY_MG * mob_diff
    eg += EVAL_MOBILITY_EG * mob_diff

    # Bishop pair (uses pre-calculated counts)
    if wb_count >= 2:
        mg += EVAL_BISHOP_PAIR
        eg += EVAL_BISHOP_PAIR
    if bb_count >= 2:
        mg -= EVAL_BISHOP_PAIR
        eg -= EVAL_BISHOP_PAIR

    # Center control (fast, no change needed)
    for (cx, cy) in CENTER_SQUARES:
        sq = square_index(cx, cy)
        col = _piece_on_square_color(bb, sq)
        if col == 1:
            mg += EVAL_CENTER_CONTROL
            eg += EVAL_CENTER_CONTROL
        elif col == -1:
            mg -= EVAL_CENTER_CONTROL
            eg -= EVAL_CENTER_CONTROL

    # Pawn structure (cached, no change needed)
    pe_mg, pe_eg = pawn_eval(bb)
    mg += pe_mg
    eg += pe_eg

    # Open/semi-open files (uses pre-calculated counts)
    w_has_rook = (wr_count + wri_count) > 0
    b_has_rook = (br_count + bri_count) > 0
    for f in range(5):
        fmask = file_mask(f)
        w_pawns_on_file = bool(WP & fmask)
        b_pawns_on_file = bool(BP & fmask)
        if w_has_rook:
            if (not w_pawns_on_file) and (not b_pawns_on_file):
                mg += EVAL_ROOK_OPEN_FILE
                eg += EVAL_ROOK_OPEN_FILE
            elif not w_pawns_on_file:
                mg += EVAL_ROOK_SEMIOPEN
                eg += EVAL_ROOK_SEMIOPEN
        if b_has_rook:
            if (not w_pawns_on_file) and (not b_pawns_on_file):
                mg -= EVAL_ROOK_OPEN_FILE
                eg -= EVAL_ROOK_OPEN_FILE
            elif not b_pawns_on_file:
                mg -= EVAL_ROOK_SEMIOPEN
                eg -= EVAL_ROOK_SEMIOPEN
        # (Right logic omitted for brevity, but it's the same pattern)
        w_has_right = wri_count > 0
        b_has_right = bri_count > 0
        if w_has_right:
            if (not w_pawns_on_file) and (not b_pawns_on_file):
                mg += EVAL_RIGHT_OPEN_FILE
                eg += EVAL_RIGHT_OPEN_FILE
            elif not w_pawns_on_file:
                mg += EVAL_RIGHT_SEMIOPEN
                eg += EVAL_RIGHT_SEMIOPEN
        if b_has_right:
            if (not w_pawns_on_file) and (not b_pawns_on_file):
                mg -= EVAL_RIGHT_OPEN_FILE
                eg -= EVAL_RIGHT_OPEN_FILE
            elif not b_pawns_on_file:
                mg -= EVAL_RIGHT_SEMIOPEN
                eg -= EVAL_RIGHT_SEMIOPEN


    # Advanced King Safety (fast, no change needed)
    w_k_sq = _pop_lsb_njit(WK) if WK else -1
    b_k_sq = _pop_lsb_njit(BK) if BK else -1
    w_king_score = 0
    b_king_score = 0
    if w_k_sq >= 0:
        kx, ky = index_to_sq(w_k_sq)
        zone = ring1_mask(kx, ky)
        counts = count_attackers_to_zone(bb, zone, white_attacking=False)
        for p, c in counts.items():
            w_king_score += EVAL_KING_ATTACK_WEIGHTS.get(p, 0) * c
        w_king_score += king_shield_penalty(bb, kx, ky, True)
        w_king_score += open_file_to_king_penalty(bb, kx, ky, True)
        w_king_score += same_file_pressure(bb, kx, True)
    if b_k_sq >= 0:
        kx, ky = index_to_sq(b_k_sq)
        zone = ring1_mask(kx, ky)
        counts = count_attackers_to_zone(bb, zone, white_attacking=True)
        for p, c in counts.items():
            b_king_score += EVAL_KING_ATTACK_WEIGHTS.get(p, 0) * c
        b_king_score += king_shield_penalty(bb, kx, ky, False)
        b_king_score += open_file_to_king_penalty(bb, kx, ky, False)
        b_king_score += same_file_pressure(bb, kx, False)

    mg += EVAL_KING_ATTACK_SCALE * (b_king_score - w_king_score)
    eg += EVAL_KING_ATTACK_SCALE * (b_king_score - w_king_score)

    # Endgame drives (now uses pre-calculated counts)
    # Total non-king pieces for EG check
    w_non_king = wp_count + wn_count + wb_count + wr_count + wri_count + wq_count
    b_non_king = bp_count + bn_count + bb_count + br_count + bri_count + bq_count
    
    def _eg_extras_for_color(white: bool) -> int:
        opp_k = BK if white else WK
        my_k = WK if white else BK
        if opp_k == 0:
            return 0
        ex, ey = index_to_sq(_pop_lsb_njit(opp_k))
        
        # Use pre-calculated counts
        q_count = wq_count if white else bq_count
        r_count = wr_count if white else br_count
        ri_count = wri_count if white else bri_count
        opp_non_king = b_non_king if white else w_non_king

        is_kqk = (q_count == 1) and ((r_count + ri_count) == 0) and (opp_non_king == 0)
        is_krk = (q_count == 0) and ((r_count + ri_count) >= 1) and (opp_non_king == 0)
        extra = 0
        if is_kqk or is_krk:
            if my_k:
                kx, ky = index_to_sq(_pop_lsb_njit(my_k))
                kprox = 4 - _chebyshev((kx, ky), (ex, ey))
                extra += EVAL_EG_OWN_KING_PROXIMITY * kprox
            edge_w = EVAL_EG_OPP_KING_TO_EDGE_R if is_krk else EVAL_EG_OPP_KING_TO_EDGE_Q
            corner_w = EVAL_EG_OPP_KING_TO_CORNER_R if is_krk else EVAL_EG_OPP_KING_TO_CORNER_Q
            extra += edge_w * (2 - _edge_distance(ex, ey))
            extra += corner_w * (2 - _corner_distance(ex, ey))
            if is_krk:
                rr_mask = (WR if white else BR) | (WRi if white else BRi)
                for rsq in _iter_set_bits(rr_mask):
                    rx, ry = index_to_sq(rsq)
                    if rx == ex or ry == ey:
                        extra += EVAL_EG_ROOK_CUTOFF
            if is_kqk and my_k:
                q_bb = WQ if white else BQ
                if q_bb:
                    qx, qy = index_to_sq(_pop_lsb_njit(q_bb))
                    kx, ky = index_to_sq(_pop_lsb_njit(my_k))
                    qd = _chebyshev((qx, qy), (ex, ey))
                    kd = _chebyshev((kx, ky), (ex, ey))
                    if qd <= 1 and kd >= 2:
                        extra += EVAL_EG_QUEEN_ADJ_PENALTY
        return extra

    eg += _eg_extras_for_color(True) - _eg_extras_for_color(False)

    # Rooks on the 7th rank (uses accumulated counts)
    if w_rooks_on_7th:
        mg += EVAL_ROOK_ON_7TH * w_rooks_on_7th
        eg += EVAL_ROOK_ON_7TH * w_rooks_on_7th
        if w_rooks_on_7th >= 2:
            mg += EVAL_ROOKS_ON_7TH
            eg += EVAL_ROOKS_ON_7TH
    if b_rooks_on_7th:
        mg -= EVAL_ROOK_ON_7TH * b_rooks_on_7th
        eg -= EVAL_ROOK_ON_7TH * b_rooks_on_7th
        if b_rooks_on_7th >= 2:
            mg -= EVAL_ROOKS_ON_7TH
            eg -= EVAL_ROOKS_ON_7TH
            
    # Extra if enemy king is on back rank (no change)
    if BK:
        bkx, bky = index_to_sq(_pop_lsb_njit(BK))
        if bky == 0 and w_rooks_on_7th:
            mg += EVAL_ROOK_ON_7TH_VS_KING * w_rooks_on_7th
            eg += EVAL_ROOK_ON_7TH_VS_KING * w_rooks_on_7th
    if WK:
        wkx, wky = index_to_sq(_pop_lsb_njit(WK))
        if wky == 4 and b_rooks_on_7th:
            mg -= EVAL_ROOK_ON_7TH_VS_KING * b_rooks_on_7th
            eg -= EVAL_ROOK_ON_7TH_VS_KING * b_rooks_on_7th

    # Knight outposts (now handled in single pass)

    # Good vs Bad bishops (uses accumulated counts)
    mg += EVAL_BAD_BISHOP_PENALTY_PER_PAWN * (w_bishops_light * wp_light_count + w_bishops_dark * wp_dark_count)
    eg += EVAL_BAD_BISHOP_PENALTY_PER_PAWN * (w_bishops_light * wp_light_count + w_bishops_dark * wp_dark_count)
    mg -= EVAL_BAD_BISHOP_PENALTY_PER_PAWN * (b_bishops_light * bp_light_count + b_bishops_dark * bp_dark_count)
    eg -= EVAL_BAD_BISHOP_PENALTY_PER_PAWN * (b_bishops_light * bp_light_count + b_bishops_dark * bp_dark_count)

    # Mobility (now handled in single pass)

    # === 4. Phase Blend ===
    phase = _phase_from_bb(bb)
    mg_w = phase / MAX_PHASE
    eg_w = (MAX_PHASE - phase) / MAX_PHASE
    blended = mg * mg_w + eg * eg_w
    
    return blended if bb.side_to_move == 0 else -blended


_IDX_TO_NAME = {0: "Pawn", 1: "Knight", 2: "Bishop", 3: "Rook", 4: "Queen", 5: "King", 6: "Right"}

def fast_static_eval(bb: "BitboardState") -> float:
    """
    Cheap static eval using only incremental Material+PST blend.
    """
    phase = _phase_from_bb(bb)
    mg_w = phase / MAX_PHASE
    eg_w = (MAX_PHASE - phase) / MAX_PHASE
    val = bb.mg_mat_pst * mg_w + bb.eg_mat_pst * eg_w
    return val if bb.side_to_move == 0 else -val

def score_move_internal(
    bb: "BitboardState",
    m,
    var: Dict[str, Any],
    ply: int,
    tt_move_tuple: Optional[Tuple[int, int, int, int]],
) -> int:
    """
    Assigns a heuristic score to a move for move ordering purposes.
    
    Move ordering is critical for alpha-beta search efficiency. This function
    assigns scores to moves so they can be sorted, with higher-scored moves
    searched first. This maximizes beta cutoffs and improves search performance.
    
    Args:
        bb: Current BitboardState (used for SEE calculation on captures).
        m: The BBMove to score.
        var: Search state dictionary containing:
            - killers: List of killer moves per ply
            - countermoves: Dictionary mapping previous move to countermove
            - history: Dictionary mapping moves to history scores
            - cont_history: Nested dictionary for continuation history
            - _prev_move_tuple: Previous move for continuation history lookup
        ply: Current search ply (for killer move indexing).
        tt_move_tuple: Optional move tuple from transposition table. If the
            current move matches this, it gets the highest priority.
    
    Returns:
        An integer score where higher values indicate moves that should be
        searched first:
        - 1,000,000: TT move (highest priority)
        - 100,000+: Captures with MVV-LVA and SEE bonuses
        - 90,000: Killer moves (first killer at current ply)
        - 85,000: Countermoves (response to previous move)
        - 0+: History scores (quiet moves with history/continuation bonuses)
    
    Notes:
        - Captures: Uses MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
          plus SEE (Static Exchange Evaluation) for accurate capture ordering
        - Quiet moves: Uses history heuristic and continuation history for learning
        - Killers: Moves that caused beta cutoffs at the same ply
        - Countermoves: Good responses to specific opponent moves
    """
    sx, sy, dx, dy = bbmove_to_tuple_xy(m)
    if isinstance(m, int):
        from_sq, to_sq, piece_type, captured_type, promo = unpack_move(m)
    else:
        from_sq, to_sq = m.from_sq, m.to_sq
        piece_type, captured_type, promo = m.piece_type, m.captured_type, m.promo
    move_tuple = (sx, sy, dx, dy)

    if tt_move_tuple is not None and move_tuple == tt_move_tuple:
        return 1_000_000

    is_capture = captured_type >= 0
    if is_capture:
        # MVV-LVA + SEE
        victim_name = _IDX_TO_NAME.get(captured_type, "Pawn")
        aggressor_name = _IDX_TO_NAME.get(piece_type, "Pawn")
        vval = PIECE_VALUES_MG.get(victim_name, 0)
        aval = PIECE_VALUES_MG.get(aggressor_name, 0)
        base = 100_000 + (vval * 10) - aval
        try:
            occ = bb.occ_all
            tgt_sq = to_sq
            stm_white = (bb.side_to_move == 0)
            all_bbs_list = [
                bb.WP, bb.WN, bb.WB, bb.WR, bb.WQ, bb.WK, bb.WRi,
                bb.BP, bb.BN, bb.BB, bb.BR, bb.BQ, bb.BK, bb.BRi,
            ]
            see_gain = bb_see_njit(int(tgt_sq), bool(stm_white), int(occ), int(vval), all_bbs_list.copy())
            if see_gain >= 0:
                base += 500 + see_gain
            else:
                base = 1_000 + see_gain
        except Exception:
            pass
        return base

    # Non-captures: killers, countermoves, history
    try:
        if ply < len(var.get("killers", [])) and move_tuple in var["killers"][ply]:
            return 90_000
    except Exception:
        pass
    try:
        prev = var.get("_prev_move_tuple")
        if prev is not None:
            cm = var.get("countermoves", {}).get(prev)
            if cm is not None and cm == move_tuple:
                return 85_000
    except Exception:
        pass
    # History + continuation history bonus for quiet moves
    base_hist = int(var.get("history", {}).get(move_tuple, 0))
    try:
        prev = var.get("_prev_move_tuple")
        if prev is not None:
            ch = var.get("cont_history", {}).get(prev, {})
            base_hist += CH_BONUS * int(ch.get(move_tuple, 0))
    except Exception:
        pass
    return base_hist


def quiescence_search_bb(
    bb: "BitboardState",
    depth: int,
    alpha: float,
    beta: float,
    var: Dict[str, Any],
    ply: int = 0,
) -> float:
    """
    Performs quiescence search to resolve tactical sequences.
    
    Quiescence search extends the search beyond the main search depth to resolve
    "unquiet" positions - those with pending captures, checks, or other tactical
    moves. This prevents the horizon effect where the engine misses tactics
    just beyond the search depth.
    
    Args:
        bb: Current BitboardState representing the position to evaluate.
        depth: Remaining quiescence depth (typically 3-5 plies). Decrements each
            recursive call until reaching 0.
        alpha: Lower bound of the alpha-beta window (best score for maximizing player).
        beta: Upper bound of the alpha-beta window (best score for minimizing player).
        var: Search state dictionary containing:
            - Time management flags (_start_t, _soft_time_s, _hard_time_s)
            - Zobrist hasher for incremental updates
            - Flags for search heuristics (qsee for SEE pruning)
            - Node counters (_qnodes)
        ply: Current search ply (distance from root). Used for mate distance scoring.
    
    Returns:
        The evaluated score for the position from the perspective of the side-to-move.
        Positive values favor the side-to-move, negative values favor the opponent.
        Mate scores are adjusted by ply distance.
    
    Notes:
        - Stand-pat: If static evaluation is already >= beta, returns beta (beta cutoff)
        - Delta pruning: Skips captures if stand-pat + piece_value < alpha
        - SEE pruning: Skips losing captures (SEE < 0) when not in check
        - Generates all moves when in check, captures-only otherwise
        - Handles checkmate and stalemate terminal conditions
        - Respects time limits and raises exceptions on timeout
    """
    # Time checks
    try:
        start_t = var.get("_start_t")
        soft_s = float(var.get("_soft_time_s") or 0.0)
        hard_s = float(var.get("_hard_time_s") or 0.0)
    except Exception:
        start_t, soft_s, hard_s = None, 0.0, 0.0
    if start_t is not None:
        elapsed = time.perf_counter() - start_t
        var["_time_pressure"] = bool(soft_s > 0 and elapsed > 0.7 * soft_s)
        if hard_s > 0 and elapsed >= hard_s:
            var["_hard_time_stop"] = True
            raise Exception("Hard time limit reached")
        if soft_s > 0 and elapsed >= soft_s:
            var["_soft_time_stop"] = True
            raise Exception("Soft time limit reached")

    var["_qnodes"] = var.get("_qnodes", 0) + 1

    # Determine check status once and test for terminal nodes
    stm_white = (bb.side_to_move == 0)
    kbb = bb.WK if stm_white else bb.BK
    ksq = _pop_lsb_njit(kbb) if kbb else -1
    in_check = False
    if ksq >= 0:
        checkers_bb, _, _ = calculate_legality_masks(bb, ksq, stm_white)
        in_check = (checkers_bb != 0)

    if in_check:
        moves = generate_legal_moves(bb, captures_only=False)
        if not moves:
            return -MATE_VALUE + ply
    else:
        moves = None

    stand_pat = evaluate_bb_state(bb)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat
    if depth == 0:
        return alpha

    # Reuse computed in_check
    # Simple delta pruning when not in check
    if (not in_check) and (stand_pat + 900 < alpha):
        return alpha

    if moves is None:
        moves = generate_legal_moves(bb, captures_only=True)
    # Order captures by MVV-LVA (and SEE via score_move_internal)
    moves.sort(key=lambda m: score_move_internal(bb, m, var, 99, None), reverse=True)

    for m in moves:
        # SEE gating for captures under time pressure (before applying move)
        if isinstance(m, int):
            _fs, _ts, _pt, _ct, _pr = unpack_move(m)
        else:
            _fs, _ts, _pt, _ct, _pr = m.from_sq, m.to_sq, m.piece_type, m.captured_type, m.promo
        if _ct >= 0 and var.get("flags", {}).get("qsee", True) and not in_check:
            vname = _IDX_TO_NAME.get(_ct, "Pawn")
            vval = PIECE_VALUES_MG.get(vname, 0)
            all_bbs_list = [
                bb.WP, bb.WN, bb.WB, bb.WR, bb.WQ, bb.WK, bb.WRi,
                bb.BP, bb.BN, bb.BB, bb.BR, bb.BQ, bb.BK, bb.BRi,
            ]
            see_gain = bb_see_njit(int(_ts), bool(bb.side_to_move == 0), int(bb.occ_all), int(vval), all_bbs_list.copy())
            if see_gain < 0:
                continue

        if var.get("flags", {}).get("inplace", False):
            undo_ip = make_move_inplace(bb, m, var.get("zobrist"))
            try:
                score = -quiescence_search_bb(bb, depth - 1, -beta, -alpha, var, ply=ply + 1)
            finally:
                unmake_move_inplace(bb, undo_ip, var.get("zobrist"))
        else:
            child = apply_bb_move(bb, m, var.get("zobrist"))
            score = -quiescence_search_bb(child, depth - 1, -beta, -alpha, var, ply=ply + 1)
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


def negamax_bb(
    bb: "BitboardState",
    depth: int,
    alpha: float,
    beta: float,
    var: Dict[str, Any],
    ply: int,
) -> Tuple[float, Optional["BBMove"]]:
    """
    Performs negamax alpha-beta search with various optimizations.
    
    This is the core search function implementing a negamax variant of alpha-beta
    pruning. It searches the game tree to a specified depth, using numerous
    optimizations including transposition tables, null-move pruning, late move
    reduction, razoring, futility pruning, and more.
    
    Args:
        bb: Current BitboardState representing the position to search.
        depth: Remaining search depth in plies. When depth reaches 0, calls
            quiescence_search_bb() to resolve tactical sequences.
        alpha: Lower bound of the alpha-beta window (best score for maximizing player).
            Updated during search as better moves are found.
        beta: Upper bound of the alpha-beta window (best score for minimizing player).
            Used for beta cutoffs when a move is too good for the opponent.
        var: Search state dictionary containing:
            - transposition_table: TT for position caching
            - zobrist: Zobrist hasher for incremental updates
            - history, killers, countermoves: Move ordering heuristics
            - flags: Search heuristic toggles (nmp, lmr, futility, qsee)
            - Time management flags and node counters
        ply: Current search ply (distance from root). Used for:
            - Repetition detection
            - Mate distance adjustment
            - Killer move indexing
    
    Returns:
        A tuple (score, best_move) where:
        - score: The best score found for this position from side-to-move's perspective.
          Positive favors side-to-move, negative favors opponent. Mate scores adjusted by ply.
        - best_move: The BBMove that achieves the best score, or None if no moves
          or if the position is terminal.
    
    Notes:
        - Transposition table: Probes TT for cached results, stores new results
        - Repetition: Detects 3-fold repetition and returns contempt score
        - Razoring: At low depths, if eval + margin <= alpha, skip to quiescence
        - Futility pruning: At depth 1, skip if eval + margin <= alpha
        - Extended futility: At depth 3-5, skip quiet moves if eval + margin <= alpha
        - Null-move pruning: Try null move, if score >= beta, assume position is good
        - Late move reduction: Reduce depth for late quiet moves, re-search if promising
        - Check extension: Extend search by 1 ply if move gives check
        - Move ordering: TT move first, then captures (MVV-LVA+SEE), then killers/history
        - Updates history, killers, and countermoves on beta cutoffs
    """
    var["_nodes"] = var.get("_nodes", 0) + 1

    # Repetition handling using position keys
    rep_counts = var.setdefault("_bb_rep_counts", {})
    if ply == 0:
        rep_counts.clear()
        for k in var.get("_bb_game_hist", []):
            rep_counts[k] = rep_counts.get(k, 0) + 1

    rep_counts[bb.zkey] = rep_counts.get(bb.zkey, 0) + 1
    try:
        if rep_counts[bb.zkey] >= 3:
            contempt = float(var.get("contempt", 0.0))
            sc = contempt if (bb.side_to_move == 0) else -contempt
            return sc, None
    finally:
        # Balance our increment even if we return early above
        c = rep_counts.get(bb.zkey, 0) - 1
        if c > 0:
            rep_counts[bb.zkey] = c
        else:
            rep_counts.pop(bb.zkey, None)

    # Time checks
    try:
        start_t = var.get("_start_t")
        soft_s = float(var.get("_soft_time_s") or 0.0)
        hard_s = float(var.get("_hard_time_s") or 0.0)
    except Exception:
        start_t, soft_s, hard_s = None, 0.0, 0.0
    if start_t is not None:
        elapsed = time.perf_counter() - start_t
        if hard_s > 0 and elapsed >= hard_s:
            var["_hard_time_stop"] = True
            raise Exception("Hard time limit reached")
        if soft_s > 0 and elapsed >= soft_s:
            var["_soft_time_stop"] = True
            raise Exception("Soft time limit reached")

    tt: TranspositionTable = var["transposition_table"]
    tt_entry = tt.probe(bb.zkey)
    hash_move_tuple = None
    if tt_entry:
        hash_move_tuple = tt_entry.best_move_tuple
        if tt_entry.depth >= depth:
            score = tt_entry.score
            try:
                if score > MATE_VALUE:
                    score = score - ply
                elif score < -MATE_VALUE:
                    score = score + ply
            except Exception:
                pass
            flag = tt_entry.flag
            if flag == TT_FLAG_EXACT:
                return score, None
            elif flag == TT_FLAG_LOWER:
                alpha = max(alpha, score)
            elif flag == TT_FLAG_UPPER:
                beta = min(beta, score)
            if alpha >= beta:
                return score, None

    if depth == 0:
        q = quiescence_search_bb(bb, 3, alpha, beta, var, ply=ply)
        return q, None

    # Determine PV-node (simple proxy via infinite bounds)
    is_pv_node = (alpha != -INF and beta != INF)

    # Determine check status for current node
    stm_white = (bb.side_to_move == 0)
    kbb = bb.WK if stm_white else bb.BK
    ksq = _pop_lsb_njit(kbb) if kbb else -1
    in_check = False
    legality_info = None
    if ksq >= 0:
        legality_info = calculate_legality_masks(bb, ksq, stm_white)
        in_check = (legality_info[0] != 0)  # legality_info[0] is checkers_bb
    # EGTB probe (stub) for low-piece positions
    try:
        piece_count = _count_bits(bb.occ_all)
    except Exception:
        piece_count = 7
    if piece_count <= 6:
        res = probe_egtb(bb)
        if res:
            # Map perfect info to engine score
            out_score: float
            try:
                if res.get("result") == "draw":
                    out_score = float(var.get("contempt", 0.0))
                elif res.get("result") == "win":
                    # mate-in-N equivalent
                    out_score = MATE_VALUE - ply
                elif res.get("result") == "loss":
                    out_score = -MATE_VALUE + ply
                else:
                    out_score = float(var.get("contempt", 0.0))
            except Exception:
                out_score = float(var.get("contempt", 0.0))
            # Store and return
            tt.store(bb.zkey, depth, out_score, TT_FLAG_EXACT, None, int(var.get("game_ply", 0)))
            return out_score, None

    # Razoring and extended futility (only if not in check)
    static_eval: Optional[float] = None
    if not in_check and ply > 0 and (not is_pv_node):
        if depth in (1, 2):
            static_eval = fast_static_eval(bb)
            margin = RAZOR_MARGIN.get(depth, 0)
            if static_eval + margin <= alpha:
                q = quiescence_search_bb(bb, 3, alpha, beta, var, ply=ply)
                return q, None

    # Futility pruning lite
    if depth <= 1 and not in_check and ply > 0 and (not is_pv_node):
        static_eval = fast_static_eval(bb)
        if static_eval + 120 * depth <= alpha:
            return static_eval, None

    # Extended futility flag: skip quiets if static eval is far below alpha
    skip_quiet = False
    if not in_check and 3 <= depth <= 5 and ply > 0 and (not is_pv_node):
        if static_eval is None:
            static_eval = fast_static_eval(bb)
        fut_m = FUTILITY_MARGIN.get(depth, 0)
        if static_eval + fut_m <= alpha:
            skip_quiet = True

    # Reverse Futility Pruning: if already clearly >= beta, skip quiets (captures/checks still searched)
    rfp_skip_quiet = False
    if var.get("flags", {}).get("rfp", True) and (not in_check) and ply > 0 and (not is_pv_node) and (1 <= depth <= 5):
        if static_eval is None:
            static_eval = fast_static_eval(bb)
        rfp_margin = RFP_MARGIN.get(depth, RFP_MARGIN[5])
        if static_eval - rfp_margin >= beta:
            rfp_skip_quiet = True

    # Move gen and ordering
    all_moves = generate_legal_moves(bb, captures_only=False, legality_info=legality_info)
    if not all_moves:
        # No legal moves: stalemate or checkmate -> loss per board_rules
        return (-MATE_VALUE + ply), None
    flags = var.get("flags", {})
    all_moves.sort(key=lambda m: score_move_internal(bb, m, var, ply, hash_move_tuple), reverse=True)

    best_score = -INF
    best_move = None
    original_alpha = alpha

    # Null Move Pruning (skip on PV nodes, check, or shallow depth)
    if var.get("flags", {}).get("nmp", True) and depth >= 3 and (not is_pv_node) and (not in_check):
        R = 3
        # Flip side to move and toggle zobrist
        try:
            z: Zobrist = var.get("zobrist")
        except Exception:
            z = None
        null_child = BitboardState(
            WP=bb.WP, WN=bb.WN, WB=bb.WB, WR=bb.WR, WQ=bb.WQ, WK=bb.WK, WRi=bb.WRi,
            BP=bb.BP, BN=bb.BN, BB=bb.BB, BR=bb.BR, BQ=bb.BQ, BK=bb.BK, BRi=bb.BRi,
            occ_white=bb.occ_white, occ_black=bb.occ_black, occ_all=bb.occ_all,
            side_to_move=(1 if bb.side_to_move == 0 else 0),
            zkey=(z.toggle_black_to_move(bb.zkey) if z else bb.zkey),
            mg_mat_pst=bb.mg_mat_pst,
            eg_mat_pst=bb.eg_mat_pst,
        )
        s, _ = negamax_bb(null_child, depth - 1 - R, -beta, -beta + 1, var, ply + 1)
        score_nmp = -s
        if score_nmp >= beta:
            return beta, None

    for i, m in enumerate(all_moves):
        # Prepare prev move tuple for continuation history in child
        pre_tuple = bbmove_to_tuple_xy(m)
        if isinstance(m, int):
            _fs, _ts, _pt, _ct, _pr = unpack_move(m)
        else:
            _fs, _ts, _pt, _ct, _pr = m.from_sq, m.to_sq, m.piece_type, m.captured_type, m.promo

        if flags.get("inplace", False):
            undo_ip = make_move_inplace(bb, m, var.get("zobrist"))
            # Check extension and child-in-check on mutated bb
            child_stm_white = (bb.side_to_move == 0)
            child_kbb = bb.WK if child_stm_white else bb.BK
            child_ksq = _pop_lsb_njit(child_kbb) if child_kbb else -1
            child_in_check = False
            if child_ksq >= 0:
                c_checkers, _, _ = calculate_legality_masks(bb, child_ksq, child_stm_white)
                child_in_check = (c_checkers != 0)
            extend = (not in_check) and child_in_check
            next_depth = max(0, depth - 1 + (1 if extend else 0))

            # Skip non-capture, non-promo, non-checks if flagged (extended futility or RFP)
            if (skip_quiet or rfp_skip_quiet) and (_ct < 0) and (_pr == 0) and (not child_in_check):
                unmake_move_inplace(bb, undo_ip, var.get("zobrist"))
                continue

            # Track previous move tuple for child node (continuation history)
            old_prev = var.get("_prev_move_tuple")
            var["_prev_move_tuple"] = pre_tuple

            try:
                # Principal variation search on mutated bb
                if i == 0:
                    s, _ = negamax_bb(bb, next_depth, -beta, -alpha, var, ply + 1)
                    score = -s
                else:
                    # Late Move Reductions for quiet moves
                    do_lmr = (
                        flags.get("lmr", True)
                        and (_ct < 0)
                        and (_pr == 0)
                        and (not in_check)
                        and (not child_in_check)
                        and depth >= 3
                        and i > 3
                    )
                    if do_lmr:
                        r = 1 + (1 if i > 8 else 0) + (1 if depth > 5 else 0)
                        red_depth = max(0, next_depth - r)
                        s, _ = negamax_bb(bb, red_depth, -(alpha + 1), -alpha, var, ply + 1)
                        score = -s
                        if score > alpha:
                            s, _ = negamax_bb(bb, next_depth, -beta, -alpha, var, ply + 1)
                            score = -s
                    else:
                        s, _ = negamax_bb(bb, next_depth, -(alpha + 1), -alpha, var, ply + 1)
                        score = -s
                        if score > alpha and score < beta:
                            s, _ = negamax_bb(bb, next_depth, -beta, -alpha, var, ply + 1)
                            score = -s
            finally:
                unmake_move_inplace(bb, undo_ip, var.get("zobrist"))

            # Restore previous move tuple after child search
            if old_prev is None:
                try:
                    del var["_prev_move_tuple"]
                except Exception:
                    var["_prev_move_tuple"] = None
            else:
                var["_prev_move_tuple"] = old_prev
        else:
            child, undo = make_move(bb, m, var.get("zobrist"))
            # Check extension: extend one ply if this move gives check to the opponent,
            # but avoid runaway when current node is already in check.
            child_stm_white = (child.side_to_move == 0)
            child_kbb = child.WK if child_stm_white else child.BK
            child_ksq = _pop_lsb_njit(child_kbb) if child_kbb else -1
            child_in_check = False
            if child_ksq >= 0:
                c_checkers, _, _ = calculate_legality_masks(child, child_ksq, child_stm_white)
                child_in_check = (c_checkers != 0)
            extend = (not in_check) and child_in_check
            next_depth = max(0, depth - 1 + (1 if extend else 0))

            # Skip non-capture, non-promo, non-checks if flagged (extended futility or RFP)
            if (skip_quiet or rfp_skip_quiet) and (_ct < 0) and (_pr == 0) and (not child_in_check):
                bb = unmake_move(child, undo)
                continue

            # Track previous move tuple for child node (continuation history)
            old_prev = var.get("_prev_move_tuple")
            var["_prev_move_tuple"] = pre_tuple

            # Principal variation search
            if i == 0:
                s, _ = negamax_bb(child, next_depth, -beta, -alpha, var, ply + 1)
                score = -s
            else:
                # Late Move Reductions for quiet moves
                do_lmr = (
                    flags.get("lmr", True)
                    and (_ct < 0)
                    and (_pr == 0)
                    and (not in_check)
                    and (not child_in_check)
                    and depth >= 3
                    and i > 3
                )
                if do_lmr:
                    r = 1 + (1 if i > 8 else 0) + (1 if depth > 5 else 0)
                    red_depth = max(0, next_depth - r)
                    s, _ = negamax_bb(child, red_depth, -(alpha + 1), -alpha, var, ply + 1)
                    score = -s
                    if score > alpha:
                        s, _ = negamax_bb(child, next_depth, -beta, -alpha, var, ply + 1)
                        score = -s
                else:
                    s, _ = negamax_bb(child, next_depth, -(alpha + 1), -alpha, var, ply + 1)
                    score = -s
                    if score > alpha and score < beta:
                        s, _ = negamax_bb(child, next_depth, -beta, -alpha, var, ply + 1)
                        score = -s

            # Restore previous move tuple after child search
            if old_prev is None:
                try:
                    del var["_prev_move_tuple"]
                except Exception:
                    var["_prev_move_tuple"] = None
            else:
                var["_prev_move_tuple"] = old_prev

        if score > best_score:
            best_score = score
            best_move = m

        if score > alpha:
            alpha = score

        if alpha >= beta:
            # Update killer/history
            if ply < len(var.get("killers", [])):
                killers = var["killers"][ply]
                if pre_tuple != killers[0]:
                    killers[1] = killers[0]
                    killers[0] = pre_tuple
            if _ct >= 0:
                ch = var.setdefault("capture_history", {})
                ch[pre_tuple] = ch.get(pre_tuple, 0) + depth**2
            else:
                var["history"][pre_tuple] = var["history"].get(pre_tuple, 0) + depth**2
            # Continuation history and countermove update
            try:
                prev_t = old_prev
                if prev_t is not None:
                    ch2 = var.setdefault("cont_history", {}).setdefault(prev_t, {})
                    ch2[pre_tuple] = ch2.get(pre_tuple, 0) + depth * depth
                    cm = var.setdefault("countermoves", {})
                    cm[prev_t] = pre_tuple
            except Exception:
                pass
            break

    # Store in TT
    tt_flag = TT_FLAG_EXACT
    if best_score <= original_alpha:
        tt_flag = TT_FLAG_UPPER
    elif best_score >= beta:
        tt_flag = TT_FLAG_LOWER
    best_tuple = bbmove_to_tuple_xy(best_move) if best_move else None
    score_to_store = best_score
    try:
        if best_score > WIN_VALUE:
            score_to_store = best_score + ply
        elif best_score < -WIN_VALUE:
            score_to_store = best_score - ply
    except Exception:
        pass
    tt.store(bb.zkey, depth, score_to_store, tt_flag, best_tuple, int(var.get("game_ply", 0)))
    return best_score, best_move

def bb_from_board(board) -> BBPos:
    """
    Converts a ChessMaker board to bitboard representation.
    
    This function scans all pieces on the board and builds bitboards for each
    piece type and color. It creates 14 individual piece bitboards (7 types × 2 colors)
    and computes aggregate occupancy masks. This is the initial conversion step
    before the engine can work with bitboard-based operations.
    
    Args:
        board: A ChessMaker-compatible board object that supports:
            - `board.get_pieces()`: Iterator over all pieces on the board
            Each piece must have:
            - `.name`: Piece type ("Pawn", "Knight", "Bishop", "Rook", "Queen", "King", "Right")
            - `.position`: Position object with `.x` and `.y` attributes (0-4)
            - `.player.name`: Color name ("white" or "black")
    
    Returns:
        A BBPos named tuple containing:
        - 14 piece bitboards: WP, WN, WB, WR, WQ, WK, WRi, BP, BN, BB, BR, BQ, BK, BRi
        - 3 occupancy masks: occ_white, occ_black, occ_all
        Each bitboard is a 25-bit integer where bit i (0-24) represents square i.
    
    Notes:
        - Handles the custom "Right" piece type (rook|knight combination)
        - Skips pieces with invalid types or missing position information
        - Used as an intermediate step in convert_board_to_bb_state()
    """
    bbs_list = [0] * 14

    try:
        for piece in board.get_pieces():
            try:
                pos = piece.position
                x, y = pos.x, pos.y
            except Exception as e:
                print(f"Error getting piece position: {e}")
                continue
            idx_base = _COLOR_OFFSET.get(getattr(piece.player, "name", "white"), 0)
            type_idx = _PIECE_TYPE_TO_IDX.get(piece.name, -1)
            if type_idx == -1:
                continue
            sq_bit = 1 << square_index(x, y)
            bbs_list[idx_base + type_idx] |= sq_bit
    except Exception as e:
        print(f"Error building bitboards: {e}")

    occ_white = (
        bbs_list[0]
        | bbs_list[1]
        | bbs_list[2]
        | bbs_list[3]
        | bbs_list[4]
        | bbs_list[5]
        | bbs_list[6]
    )
    occ_black = (
        bbs_list[7]
        | bbs_list[8]
        | bbs_list[9]
        | bbs_list[10]
        | bbs_list[11]
        | bbs_list[12]
        | bbs_list[13]
    )
    occ_all = occ_white | occ_black

    return BBPos(
        WP=bbs_list[0],
        WN=bbs_list[1],
        WB=bbs_list[2],
        WR=bbs_list[3],
        WQ=bbs_list[4],
        WK=bbs_list[5],
        WRi=bbs_list[6],
        BP=bbs_list[7],
        BN=bbs_list[8],
        BB=bbs_list[9],
        BR=bbs_list[10],
        BQ=bbs_list[11],
        BK=bbs_list[12],
        BRi=bbs_list[13],
        occ_white=occ_white,
        occ_black=occ_black,
        occ_all=occ_all,
    )


def _find_lva_njit(sq: int, occ: int, stm_white: bool, all_bbs: list[int]) -> Tuple[int, int, int]:
    """
    Finds the least-valuable attacker (LVA) on a target square for the side-to-move.
    
    This function is used in Static Exchange Evaluation (SEE) to determine the
    weakest piece that can attack a given square. It searches through pieces in
    order of increasing value (pawns, knights, bishops, rooks, right, queens, king)
    and returns the first attacker found.
    
    Args:
        sq: Target square index [0, 24] being contested in the exchange.
        occ: Occupancy bitboard for all pieces (used for sliding piece attacks).
        stm_white: True if the side-to-move is white, False if black.
        all_bbs: List of 14 piece bitboards in fixed order:
            [WP, WN, WB, WR, WQ, WK, WRi, BP, BN, BB, BR, BQ, BK, BRi]
    
    Returns:
        A tuple (attacker_sq, attacker_value, attacker_index) where:
        - attacker_sq: Square index of the least-valuable attacker, or -1 if none
        - attacker_value: Base material value of the attacker (MG scale: 120-20000)
        - attacker_index: Index into all_bbs list for the attacker piece type
    
    Notes:
        - Search order: Pawns → Knights → Bishops → Rooks → Right → Queens → King
        - Uses precomputed attack masks for knights and king
        - Uses sliding attack functions for rooks, bishops, and queens
        - Right piece uses both rook and knight attacks
        - Returns the first (least valuable) attacker found
    """
    r, c = sq // 5, sq % 5

    if stm_white:
        if c > 0 and r > 0:
            s = (r - 1) * 5 + (c - 1)
            if all_bbs[0] & (1 << s):
                return s, 120, 0
        if c < 4 and r > 0:
            s = (r - 1) * 5 + (c + 1)
            if all_bbs[0] & (1 << s):
                return s, 120, 0

        attackers_n = _KNIGHT_MOVES[sq] & all_bbs[1]
        if attackers_n:
            return _pop_lsb_njit(attackers_n), 350, 1

        attackers_b = _get_bishop_attacks(sq, occ) & all_bbs[2]
        if attackers_b:
            return _pop_lsb_njit(attackers_b), 330, 2

        attackers_r = _get_rook_attacks(sq, occ) & all_bbs[3]
        if attackers_r:
            return _pop_lsb_njit(attackers_r), 500, 3

        attackers_ri = (_get_rook_attacks(sq, occ) | _KNIGHT_MOVES[sq]) & all_bbs[6]
        if attackers_ri:
            return _pop_lsb_njit(attackers_ri), 500, 6

        attackers_q = (_get_rook_attacks(sq, occ) | _get_bishop_attacks(sq, occ)) & all_bbs[4]
        if attackers_q:
            return _pop_lsb_njit(attackers_q), 900, 4

        attackers_k = _KING_MOVES[sq] & all_bbs[5]
        if attackers_k:
            return _pop_lsb_njit(attackers_k), 20000, 5

    else:
        if c > 0 and r < 4:
            s = (r + 1) * 5 + (c - 1)
            if all_bbs[7] & (1 << s):
                return s, 120, 7
        if c < 4 and r < 4:
            s = (r + 1) * 5 + (c + 1)
            if all_bbs[7] & (1 << s):
                return s, 120, 7

        attackers_n = _KNIGHT_MOVES[sq] & all_bbs[8]
        if attackers_n:
            return _pop_lsb_njit(attackers_n), 350, 8

        attackers_b = _get_bishop_attacks(sq, occ) & all_bbs[9]
        if attackers_b:
            return _pop_lsb_njit(attackers_b), 330, 9

        attackers_r = _get_rook_attacks(sq, occ) & all_bbs[10]
        if attackers_r:
            return _pop_lsb_njit(attackers_r), 500, 10

        attackers_ri = (_get_rook_attacks(sq, occ) | _KNIGHT_MOVES[sq]) & all_bbs[13]
        if attackers_ri:
            return _pop_lsb_njit(attackers_ri), 500, 13

        attackers_q = (_get_rook_attacks(sq, occ) | _get_bishop_attacks(sq, occ)) & all_bbs[11]
        if attackers_q:
            return _pop_lsb_njit(attackers_q), 900, 11

        attackers_k = _KING_MOVES[sq] & all_bbs[12]
        if attackers_k:
            return _pop_lsb_njit(attackers_k), 20000, 12

    return -1, 0, -1


def bb_see_njit(
    sq: int,
    stm_white: bool,
    occ: int,
    victim_val: int,
    all_bbs_in: list[int],
) -> int:
    """
    Performs Static Exchange Evaluation (SEE) on a target square.
    
    Static Exchange Evaluation simulates an optimal capture sequence on a square
    to determine the net material gain. It alternates sides, repeatedly selecting
    the least-valuable attacker, and computes the final exchange value using
    minimax backup through a gain stack.
    
    Args:
        sq: Target square index [0, 24] where the exchange occurs.
        stm_white: True if white is to move first in the exchange, False if black.
        occ: Initial occupancy bitboard representing all pieces before the exchange.
        victim_val: Base material value of the initial victim piece on sq (MG scale).
        all_bbs_in: List of 14 piece bitboards [WP...BRi]. WARNING: This list is
            mutated during evaluation. Callers must pass a copy if the original
            bitboards need to be preserved.
    
    Returns:
        The net material gain for the side-to-move (positive is good for STM).
        The value represents the expected material outcome of the exchange,
        accounting for all possible capture sequences.
    
    Notes:
        - Uses a fixed-size gain stack (32 entries) to avoid recursion
        - Stops early on double-negative cutoffs (both sides losing)
        - Alternates sides after each capture
        - Uses minimax backup to find optimal exchange value
        - Mutates all_bbs_in during evaluation (removes pieces as they're captured)
    """
    gain_stack = [0] * 32
    d = 0
    gain_stack[d] = victim_val

    cur_stm = stm_white
    cur_occ = occ

    while True:
        lva_sq, lva_val, lva_idx = _find_lva_njit(sq, cur_occ, cur_stm, all_bbs_in)
        if lva_sq == -1:
            break

        d += 1
        gain_stack[d] = lva_val - gain_stack[d - 1]

        if gain_stack[d] < 0 and gain_stack[d - 1] < 0:
            break

        cur_occ ^= 1 << lva_sq
        all_bbs_in[lva_idx] ^= 1 << lva_sq
        cur_stm = not cur_stm

    while d > 0:
        gain_stack[d - 1] = -max(-gain_stack[d - 1], gain_stack[d])
        d -= 1

    return gain_stack[0]


def bb_see(bbpos: BBPos, sq: int, occ: int, stm_white: bool, victim_val: int) -> int:
    """
    Convenience wrapper for Static Exchange Evaluation using a BBPos container.
    
    This function provides a simpler interface to bb_see_njit() by accepting a
    BBPos named tuple instead of a list of bitboards. It extracts the bitboards
    from the BBPos, makes a copy (since bb_see_njit mutates the list), and
    calls the core SEE function.
    
    Args:
        bbpos: A BBPos named tuple containing all 14 piece bitboards in order:
            WP, WN, WB, WR, WQ, WK, WRi, BP, BN, BB, BR, BQ, BK, BRi
        sq: Target square index [0, 24] where the exchange occurs.
        occ: Occupancy bitboard representing all pieces before the exchange.
        stm_white: True if white is to move first in the exchange, False if black.
        victim_val: Base material value of the initial victim piece on sq (MG scale).
    
    Returns:
        The net material gain for the side-to-move (positive is good for STM).
        The value represents the expected material outcome of the exchange.
    
    Notes:
        - Makes a copy of bitboards before passing to bb_see_njit (which mutates them)
        - Returns 0 if an error occurs during SEE calculation
        - Used in move ordering to prioritize good captures
    """
    try:
        all_bbs_list = [
            bbpos.WP,
            bbpos.WN,
            bbpos.WB,
            bbpos.WR,
            bbpos.WQ,
            bbpos.WK,
            bbpos.WRi,
            bbpos.BP,
            bbpos.BN,
            bbpos.BB,
            bbpos.BR,
            bbpos.BQ,
            bbpos.BK,
            bbpos.BRi,
        ]

        return bb_see_njit(int(sq), bool(stm_white), int(occ), int(victim_val), all_bbs_list.copy())
    except Exception as e:
        print(f"Error in bb_see: {e}")
        return 0


# =====================================================================
# === Endgame Tablebases (EGTB) Stub
# =====================================================================

def probe_egtb(bb: "BitboardState"):
    """
    Tablebase probe for 3-man KQK/KRK (Right treated as Rook).
    Returns None if no EGTB is available for current material.
    On hit, returns dict: {'result': 'win'|'loss'|'draw', 'dtm': int}
    """
    import os
    import errno

    def count_total_pieces(b: "BitboardState") -> int:
        total = 0
        total += _count_bits(b.WP | b.WN | b.WB | b.WR | b.WQ | b.WK | b.WRi)
        total += _count_bits(b.BP | b.BN | b.BB | b.BR | b.BQ | b.BK | b.BRi)
        return total

    # Gate by piece count
    try:
        if count_total_pieces(bb) > 5:
            return None
    except Exception:
        return None

    # Identify supported 3-man families
    def material_signature(b: "BitboardState"):
        wq = _count_bits(b.WQ)
        wr_total = _count_bits(b.WR) + _count_bits(b.WRi)
        wk = _count_bits(b.WK)
        bq = _count_bits(b.BQ)
        br_total = _count_bits(b.BR) + _count_bits(b.BRi)
        bk = _count_bits(b.BK)
        other = (
            _count_bits(b.WP | b.WN | b.WB) +
            _count_bits(b.BP | b.BN | b.BB)
        )
        if other != 0:
            return None
        if wk == 1 and bk == 1 and wq == 1 and bq == 0 and br_total == 0 and wr_total == 0:
            return "KQK_w"
        if wk == 1 and bk == 1 and wr_total == 1 and wq == 0 and bq == 0 and br_total == 0:
            return "KRK_w"
        if wk == 1 and bk == 1 and bq == 1 and wq == 0 and wr_total == 0 and br_total == 0:
            return "KQK_b"
        if wk == 1 and bk == 1 and br_total == 1 and wq == 0 and bq == 0 and wr_total == 0:
            return "KRK_b"
        return None

    sig = material_signature(bb)
    if not sig:
        return None

    tb_dir = os.path.join("egtb")
    tb_file = os.path.join(tb_dir, f"{sig[:3]}.bin")
    if not os.path.exists(tb_file):
        return None

    def encode_index_from_bb(b: "BitboardState") -> int:
        stm = 0 if b.side_to_move == 0 else 1
        wk = _pop_lsb_njit(b.WK)
        bk = _pop_lsb_njit(b.BK)
        if "KQK" in sig:
            qbb = b.WQ if sig.endswith("_w") else b.BQ
            attacker = _pop_lsb_njit(qbb)
        else:
            if sig.endswith("_w"):
                rbb = b.WR if b.WR else b.WRi
            else:
                rbb = b.BR if b.BR else b.BRi
            attacker = _pop_lsb_njit(rbb)
        big = 25 * 25 * 25
        return (stm * big) + (wk * 25 + attacker) * 25 + bk

    try:
        idx = encode_index_from_bb(bb)
    except Exception:
        return None

    try:
        with open(tb_file, "rb") as f:
            try:
                f.seek(idx)
                val = f.read(1)
            except OSError as e:
                if e.errno == errno.EINVAL:
                    return None
                raise
    except Exception:
        return None
    if not val:
        return None
    v = val[0]
    if v == 0x00:
        return {"result": "draw", "dtm": 0}
    if (v & 0xC0) == 0x80:
        return {"result": "win", "dtm": int(v & 0x3F)}
    if (v & 0xC0) == 0x40:
        return {"result": "loss", "dtm": int(v & 0x3F)}
    return {"result": "draw", "dtm": 0}

# =====================================================================
# === Zobrist Hashing (UPDATED FOR 'Right' PIECE and Incremental Updates)
# =====================================================================

@dataclass(slots=True)
class Zobrist:
    """Implements Zobrist hashing for 5x5 chess.

    Provides a reproducible PRNG-backed table mapping (piece, color, square)
    to 64-bit integers and utilities to compute and update hashes.
    """
    rand_gen: random.Random = field(init=False)
    width: int = field(init=False)
    height: int = field(init=False)
    num_squares: int = field(init=False)
    PIECE_TO_INT: Dict[str, int] = field(init=False)
    NUM_PIECE_TYPES: int = field(init=False)
    COLOR_TO_INT: Dict[str, int] = field(init=False)
    NUM_COLORS: int = field(init=False)
    zkeys: List[List[List[int]]] = field(init=False)
    black_to_move_hash: int = field(init=False)
    
    def __init__(self, seed=42):
        """
        Initializes a Zobrist hashing context for a 5x5 board.

        Args:
            seed: Optional seed for the internal PRNG to make hashes
                reproducible across runs. The default is 42.

        Side effects:
            Builds a table of (7 piece types x 2 colors x 25 squares) 64-bit
            keys and a separate key for the side-to-move (black).
        """
        self.rand_gen = random.Random(seed)
        self.width = 5
        self.height = 5
        self.num_squares = self.width * self.height

        self.PIECE_TO_INT = {
            "Pawn": 0,
            "Knight": 1,
            "Bishop": 2,
            "Rook": 3,
            "Queen": 4,
            "King": 5,
            "Right": 6,
        }
        self.NUM_PIECE_TYPES = 7
        self.COLOR_TO_INT = {"white": 0, "black": 1}
        self.NUM_COLORS = 2

        self.zkeys = [
            [
                [self._rand_64() for _ in range(self.num_squares)]
                for _ in range(self.NUM_COLORS)
            ]
            for _ in range(self.NUM_PIECE_TYPES)
        ]

        self.black_to_move_hash = self._rand_64()

    def _rand_64(self) -> int:
        """
        Returns a uniformly distributed 64-bit random integer.

        Returns:
            An integer in [0, 2**64-1].
        """
        return self.rand_gen.getrandbits(64)

    def get_piece_hash(self, piece, x: int, y: int) -> int:
        """
        Looks up the Zobrist key for a specific piece on a square.

        Args:
            piece: A piece object exposing `.name` and `.player.name`.
            x: File/column [0, 4].
            y: Rank/row [0, 4].

        Returns:
            0 if the piece is None/invalid; otherwise the 64-bit key for
            (piece, color, square).
        """
        if not piece:
            return 0
        try:
            p_idx = self.PIECE_TO_INT[piece.name]
            c_idx = self.COLOR_TO_INT[piece.player.name]
            sq_idx = y * self.width + x
            return self.zkeys[p_idx][c_idx][sq_idx]
        except (KeyError, IndexError, AttributeError):
            return 0

    def compute_full_hash(self, board, player_name: str) -> int:
        """
        Computes the Zobrist hash from scratch for the entire position.

        Args:
            board: Board supporting square access `board[Position(x,y)]` and
                `.piece` on squares.
            player_name: "white" or "black"; if "black", the STM key is XORed.

        Returns:
            The 64-bit Zobrist hash for the given board and side to move.
        """
        h = 0
        try:
            for piece in board.get_pieces():
                pos = getattr(piece, "position", None)
                if pos is None:
                    continue
                h ^= self.get_piece_hash(piece, pos.x, pos.y)
        except Exception as e:
            print(f"Error computing full hash: {e}")

        if player_name == "black":
            h ^= self.black_to_move_hash

        return h

    def toggle_piece(self, h: int, piece, x: int, y: int) -> int:
        """
        XORs the hash key for the given piece on (x,y) into an existing hash.

        Args:
            h: The current Zobrist hash.
            piece: The piece instance to toggle.
            x: File [0, 4].
            y: Rank [0, 4].

        Returns:
            The updated hash after XORing the piece-square key.
        """
        return h ^ self.get_piece_hash(piece, x, y)

    def toggle_black_to_move(self, h: int) -> int:
        """
        Toggles the side-to-move component of the Zobrist hash.

        Args:
            h: The current Zobrist hash.

        Returns:
            The updated hash after XORing the black-to-move key.
        """
        return h ^ self.black_to_move_hash

    # --- Bitboard helpers for incremental hashing (by indices) ---
    def get_piece_hash_by_indices(self, piece_type_idx: int, color_idx: int, sq_idx: int) -> int:
        """
        Returns the Zobrist key for (piece_type_idx, color_idx, sq_idx).
        Indices must match the ranges used to build the table:
          piece_type_idx in [0..6], color_idx in [0..1], sq_idx in [0..24].
        """
        return self.zkeys[piece_type_idx][color_idx][sq_idx]


    def toggle_by_indices(self, h: int, piece_type_idx: int, color_idx: int, sq_idx: int) -> int:
        """
        XOR toggles the hash by specifying indices directly.
        """
        return h ^ self.zkeys[piece_type_idx][color_idx][sq_idx]



# =====================================================================
# === Pawn-only Zobrist and Pawn Hash Table (Pawn TT)
# =====================================================================

# Deterministic 64-bit keys for pawn-only hashing
_PAWN_Z_RAND = random.Random(7737)
PAWN_Z_KEYS_WHITE: list[int] = [_PAWN_Z_RAND.getrandbits(64) for _ in range(25)]
PAWN_Z_KEYS_BLACK: list[int] = [_PAWN_Z_RAND.getrandbits(64) for _ in range(25)]


def pawn_zobrist_key(bb: "BitboardState") -> int:
    """
    Computes a Zobrist hash key using only pawn positions.
    
    This function generates a hash key based solely on pawn positions, ignoring
    all other pieces. This is used for pawn structure evaluation caching, since
    pawn structure evaluation depends only on pawn positions, not on other pieces.
    
    Args:
        bb: BitboardState containing WP and BP (white and black pawn) bitboards.
    
    Returns:
        A 64-bit integer hash key computed by XORing precomputed keys for each
        pawn square. The same pawn configuration always produces the same key.
    
    Notes:
        - Uses separate key tables for white and black pawns (PAWN_Z_KEYS_WHITE/BLACK)
        - Only depends on pawn positions, making it suitable for pawn TT caching
        - Used by pawn_eval() to cache pawn structure evaluation results
    """
    h = 0
    wp = bb.WP
    while wp:
        lsb = wp & -wp
        idx = lsb.bit_length() - 1
        h ^= PAWN_Z_KEYS_WHITE[idx]
        wp ^= lsb
    bp = bb.BP
    while bp:
        lsb = bp & -bp
        idx = lsb.bit_length() - 1
        h ^= PAWN_Z_KEYS_BLACK[idx]
        bp ^= lsb
    return h


class PawnTTEntry:
    __slots__ = ["key", "mg", "eg"]

    def __init__(self, key: int, mg: int, eg: int):
        self.key = key
        self.mg = mg
        self.eg = eg


class PawnHashTable:
    """
    Fixed-size pawn-only cache storing MG/EG differential scores for pawn structure.
    """
    __slots__ = ("size", "mask", "table", "hits", "probes")
    def __init__(self, size: int = 1 << 18):
        self.size = size
        self.mask = self.size - 1
        self.table: list[PawnTTEntry | None] = [None] * self.size
        self.hits = 0
        self.probes = 0

    def index(self, key: int) -> int:
        return key & self.mask

    def probe(self, key: int) -> PawnTTEntry | None:
        self.probes += 1
        idx = self.index(key)
        e = self.table[idx]
        if e is not None and e.key == key:
            self.hits += 1
            return e
        return None

    def store(self, key: int, mg: int, eg: int) -> None:
        idx = self.index(key)
        self.table[idx] = PawnTTEntry(key, mg, eg)


_PAWN_TT = PawnHashTable()


def pawn_eval(bb: "BitboardState") -> tuple[int, int]:
    """
    Evaluates pawn structure and returns middlegame and endgame contributions.
    
    This function analyzes pawn structure features including doubled pawns,
    isolated pawns, passed pawns, connected passers, and backward pawns.
    Results are cached in a pawn-only transposition table since pawn structure
    evaluation depends only on pawn positions.
    
    Args:
        bb: BitboardState containing WP and BP (white and black pawn) bitboards.
    
    Returns:
        A tuple (mg_diff, eg_diff) where:
        - mg_diff: Middlegame pawn structure score (positive favors White)
        - eg_diff: Endgame pawn structure score (positive favors White)
        Both values are integers representing centipawns (1/100 of a pawn).
    
    Notes:
        - Cached using pawn_zobrist_key() for performance
        - Evaluates features:
          * Doubled pawns: Multiple pawns on same file (penalty)
          * Isolated pawns: Pawns with no friendly pawns on adjacent files (penalty)
          * Passed pawns: Pawns with no enemy pawns blocking their path (bonus, increases with rank)
          * Connected passers: Adjacent passed pawns (bonus)
          * Backward pawns: Pawns that can't be supported and are attacked (penalty)
        - Passed pawn bonuses increase with distance from home rank
        - Results are differential (White score - Black score)
    """
    key = pawn_zobrist_key(bb)
    e = _PAWN_TT.probe(key)
    if e is not None:
        return e.mg, e.eg

    mg = 0
    eg = 0

    # Doubled/isolated and passed pawns; connected passers; backward pawns
    # File counts
    wp_files = [0] * 5
    bp_files = [0] * 5
    for sq in _iter_set_bits(bb.WP):
        x, _y = index_to_sq(sq)
        wp_files[x] += 1
    for sq in _iter_set_bits(bb.BP):
        x, _y = index_to_sq(sq)
        bp_files[x] += 1

    # Doubled / Isolated (White adds, Black subtracts)
    for f in range(5):
        if wp_files[f] >= 2:
            mg += EVAL_DOUBLED_PAWN
            eg += EVAL_DOUBLED_PAWN
        if wp_files[f] > 0:
            left = wp_files[f - 1] > 0 if f - 1 >= 0 else False
            right = wp_files[f + 1] > 0 if f + 1 < 5 else False
            if not left and not right:
                mg += EVAL_ISOLATED_PAWN
                eg += EVAL_ISOLATED_PAWN
        if bp_files[f] >= 2:
            mg -= EVAL_DOUBLED_PAWN
            eg -= EVAL_DOUBLED_PAWN
        if bp_files[f] > 0:
            left = bp_files[f - 1] > 0 if f - 1 >= 0 else False
            right = bp_files[f + 1] > 0 if f + 1 < 5 else False
            if not left and not right:
                mg -= EVAL_ISOLATED_PAWN
                eg -= EVAL_ISOLATED_PAWN

    # Passed pawns and connected passers
    white_passers_files: list[int] = []
    black_passers_files: list[int] = []
    for sq in _iter_set_bits(bb.WP):
        x, y = index_to_sq(sq)
        is_passed = True
        for fx in range(max(0, x - 1), min(4, x + 1) + 1):
            for ry in range(y - 1, -1, -1):
                if bb.BP & (1 << square_index(fx, ry)):
                    is_passed = False
                    break
            if not is_passed:
                break
        if is_passed:
            rank_from_home = (4 - y)
            rank_from_home = max(0, min(4, rank_from_home))
            mg += EVAL_PASSED_PAWN_MG[rank_from_home]
            eg += EVAL_PASSED_PAWN_EG[rank_from_home]
            white_passers_files.append(x)
    for sq in _iter_set_bits(bb.BP):
        x, y = index_to_sq(sq)
        is_passed = True
        for fx in range(max(0, x - 1), min(4, x + 1) + 1):
            for ry in range(y + 1, 5, 1):
                if bb.WP & (1 << square_index(fx, ry)):
                    is_passed = False
                    break
            if not is_passed:
                break
        if is_passed:
            rank_from_home = y
            rank_from_home = max(0, min(4, rank_from_home))
            mg -= EVAL_PASSED_PAWN_MG[rank_from_home]
            eg -= EVAL_PASSED_PAWN_EG[rank_from_home]
            black_passers_files.append(x)

    if white_passers_files:
        wf = sorted(set(white_passers_files))
        for i in range(len(wf) - 1):
            if wf[i + 1] == wf[i] + 1:
                mg += EVAL_CONNECTED_PASSERS
                eg += EVAL_CONNECTED_PASSERS
    if black_passers_files:
        bf = sorted(set(black_passers_files))
        for i in range(len(bf) - 1):
            if bf[i + 1] == bf[i] + 1:
                mg -= EVAL_CONNECTED_PASSERS
                eg -= EVAL_CONNECTED_PASSERS

    # Backward pawns (approx)
    for sq in _iter_set_bits(bb.WP):
        x, y = index_to_sq(sq)
        has_support = False
        for dx in (-1, 1):
            sx, sy = x + dx, y + 1
            if 0 <= sx < 5 and 0 <= sy < 5:
                if bb.WP & (1 << square_index(sx, sy)):
                    has_support = True
                    break
        attacked_by_bp = False
        for dx in (-1, 1):
            ax, ay = x + dx, y - 1
            if 0 <= ax < 5 and 0 <= ay < 5:
                if bb.BP & (1 << square_index(ax, ay)):
                    attacked_by_bp = True
                    break
        if (not has_support) and attacked_by_bp:
            mg += EVAL_BACKWARD_PAWN
            eg += EVAL_BACKWARD_PAWN
    for sq in _iter_set_bits(bb.BP):
        x, y = index_to_sq(sq)
        has_support = False
        for dx in (-1, 1):
            sx, sy = x + dx, y - 1
            if 0 <= sx < 5 and 0 <= sy < 5:
                if bb.BP & (1 << square_index(sx, sy)):
                    has_support = True
                    break
        attacked_by_wp = False
        for dx in (-1, 1):
            ax, ay = x + dx, y + 1
            if 0 <= ax < 5 and 0 <= ay < 5:
                if bb.WP & (1 << square_index(ax, ay)):
                    attacked_by_wp = True
                    break
        if (not has_support) and attacked_by_wp:
            mg -= EVAL_BACKWARD_PAWN
            eg -= EVAL_BACKWARD_PAWN

    _PAWN_TT.store(key, int(mg), int(eg))
    return int(mg), int(eg)


# =====================================================================
# === Transposition Table
# =====================================================================


class TTEntry:
    """
    Entry stored in the transposition table.

    Attributes:
        key: Zobrist hash of the position.
        depth: Search depth (plies) for which the score is valid.
        score: The stored score; may be mate-distance encoded.
        flag: Bound type: TT_FLAG_EXACT, TT_FLAG_LOWER, or TT_FLAG_UPPER.
        best_move_tuple: Optional principal move tuple (sx, sy, dx, dy).
    """

    __slots__ = ["key", "depth", "score", "flag", "best_move_tuple", "age"]

    def __init__(self, key, depth, score, flag, best_move_tuple, age):
        """
        Creates a `TTEntry` with the supplied data.

        Args:
            key: Zobrist hash identifying the position.
            depth: Depth in plies at which the score was evaluated.
            score: Stored search score (could be mate adjusted when stored).
            flag: One of TT_FLAG_EXACT/LOWER/UPPER describing the bound.
            best_move_tuple: Optional move tuple for reconstructing PV.
        """
        self.key: int = key
        self.depth: int = depth
        self.score: float = score
        self.flag: int = flag
        self.best_move_tuple: Optional[Tuple[int, int, int, int]] = best_move_tuple
        self.age: int = int(age)


TT_FLAG_EXACT = 0
TT_FLAG_LOWER = 1
TT_FLAG_UPPER = 2


class TranspositionTable:
    """
    Fixed-size transposition table backed by a contiguous bytearray.

    Direct-mapped indexing by masked Zobrist key. Shallow replacement:
    replace empty, same key, deeper depth, or same depth with newer age.
    Tracks probe/hit statistics for diagnostics.
    """
    __slots__ = ("size", "index_mask", "buf", "hits", "probes")

    # Packed TT entry layout (little-endian, 24 bytes total):
    # key:Q, depth:H, score:i, flag:B, sx:B, sy:B, dx:B, dy:B, age:H, pad:3x
    _TT_STRUCT = struct.Struct("<Q H i B B B B B H 3x")
    _ENTRY_SIZE = _TT_STRUCT.size
    _MOVE_NONE = 255

    def __init__(self, entry_count: int = 1048576):
        self.size = entry_count
        self.index_mask = self.size - 1
        self.buf = bytearray(self.size * self._ENTRY_SIZE)
        self.hits = 0
        self.probes = 0

    def clear(self) -> None:
        self.buf = bytearray(len(self.buf))
        self.hits = 0
        self.probes = 0

    def get_index(self, zobrist_key: int) -> int:
        return zobrist_key & self.index_mask

    def _offset(self, index: int) -> int:
        return index * self._ENTRY_SIZE

    def probe(self, zobrist_key: int) -> Optional[TTEntry]:
        self.probes += 1
        index = self.get_index(zobrist_key)
        off = self._offset(index)
        key, depth, score_i, flag, sx, sy, dx, dy, age = self._TT_STRUCT.unpack_from(self.buf, off)
        if depth == 0 or key != zobrist_key:
            return None
        self.hits += 1
        if sx == self._MOVE_NONE or sy == self._MOVE_NONE or dx == self._MOVE_NONE or dy == self._MOVE_NONE:
            move = None
        else:
            move = (int(sx), int(sy), int(dx), int(dy))
        return TTEntry(int(key), int(depth), int(score_i), int(flag), move, int(age))

    def store(
        self,
        zobrist_key: int,
        depth: int,
        score: float,
        flag: int,
        best_move_tuple: Optional[Tuple[int, int, int, int]],
        age: int,
    ) -> None:
        index = self.get_index(zobrist_key)
        off = self._offset(index)
        cur_key, cur_depth, _cur_score, _cur_flag, _sx, _sy, _dx, _dy, cur_age = self._TT_STRUCT.unpack_from(self.buf, off)

        replace = False
        if cur_depth == 0:
            replace = True
        elif depth > cur_depth:
            replace = True
        elif depth == cur_depth and int(age) > int(cur_age):
            replace = True
        if not replace:
            return

        if best_move_tuple is None:
            sx = sy = dx = dy = self._MOVE_NONE
        else:
            sx, sy, dx, dy = best_move_tuple
            sx = int(sx) if 0 <= int(sx) <= 255 else self._MOVE_NONE
            sy = int(sy) if 0 <= int(sy) <= 255 else self._MOVE_NONE
            dx = int(dx) if 0 <= int(dx) <= 255 else self._MOVE_NONE
            dy = int(dy) if 0 <= int(dy) <= 255 else self._MOVE_NONE

        self._TT_STRUCT.pack_into(
            self.buf,
            off,
            int(zobrist_key),
            int(depth),
            int(score),
            int(flag),
            sx,
            sy,
            dx,
            dy,
            int(age),
        )


# =====================================================================
# === Helper Functions (UPDATED FOR CHESSMAKER COMPATIBILITY)
# =====================================================================


def get_last_search_info() -> Dict[str, Any]:
    """
    Retrieves diagnostic information from the most recent root-level search.
    
    This function provides access to search statistics and diagnostics that were
    collected during the last call to the agent() function. Useful for debugging,
    performance analysis, and understanding search behavior.
    
    Returns:
        A shallow copy of the internal _LAST_SEARCH_INFO dictionary containing
        search metrics such as:
        - depth: Maximum search depth reached
        - nodes: Total nodes searched
        - qnodes: Quiescence nodes searched
        - tthits: Transposition table hits
        - pv: Principal variation (best line of play)
        - score: Best score found
        - time: Search time in seconds
        And other diagnostic information.
    
    Notes:
        - Returns a copy to prevent external modification of internal state
        - May be empty if no search has been performed yet
        - Updated by agent() after each search iteration
    """
    return dict(_LAST_SEARCH_INFO)


def opponent_name(board, name: str) -> str:
    """
    Returns the name of the opponent color.
    
    A simple utility function that returns the opposite color name. Used for
    convenience in code that needs to reference the opponent's color.
    
    Args:
        board: Unused parameter, kept for API compatibility with call sites
            that pass the board object.
        name: The current player's color name, either "white" or "black".
    
    Returns:
        "black" if name is "white", "white" if name is "black".
    
    Example:
        >>> opponent_name(None, "white")
        "black"
        >>> opponent_name(None, "black")
        "white"
    """
    return "black" if name == "white" else "white"


def move_to_str(piece, move_opt) -> str:
    """
    Formats a chess move as a human-readable string for debugging and logging.
    
    This function converts a piece and move option into a compact string
    representation showing the piece name and source/destination coordinates.
    Used primarily for debug output and logging.
    
    Args:
        piece: The chess piece making the move. Must have:
            - .name: Piece type name (e.g., "Knight", "Pawn")
            - .position: Position object with .x and .y attributes
        move_opt: The move option representing the destination. Must have:
            - .position: Position object with .x and .y attributes
    
    Returns:
        A string in the format "piecename(sx,sy)->(dx,dy)" where:
        - piecename: Lowercase piece name
        - sx, sy: Source coordinates
        - dx, dy: Destination coordinates
        Returns str(move_opt) if attributes are missing.
    
    Example:
        >>> move_to_str(piece, move)
        "knight(1,2)->(2,4)"
    
    Notes:
        - Handles missing attributes gracefully
        - Piece name is converted to lowercase
        - Used for logging and debug output
    """
    try:
        sx, sy = piece.position.x, piece.position.y
        dx, dy = move_opt.position.x, move_opt.position.y
        return f"{piece.name.lower()}({sx},{sy})->({dx},{dy})"
    except Exception as e:
        print(f"Error in move_to_str: {e}")
        return str(move_opt)


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    """
    Returns the Chebyshev distance between two board coordinates on a 5x5 grid.

    The Chebyshev distance is appropriate for king moves (max of file/rank deltas).

    Args:
        a: Coordinate pair (x, y) for the first location.
        b: Coordinate pair (x, y) for the second location.

    Returns:
        Integer distance in the range [0, 4] on a 5x5 board.
    """
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _edge_distance(x: int, y: int) -> int:
    """
    Returns the minimum number of steps from (x, y) to any board edge on 5x5.

    On a 5x5 board this is in [0, 2]. Edges score 0 (already on edge),
    the center (2,2) scores 2.

    Args:
        x: File/column [0, 4].
        y: Rank/row [0, 4].

    Returns:
        Minimum steps to the nearest edge.
    """
    return min(x, 4 - x, y, 4 - y)


def _corner_distance(x: int, y: int) -> int:
    """
    Returns the Chebyshev distance from (x, y) to the nearest corner on 5x5.

    On a 5x5 board this is in [0, 2]. Corners score 0 (already in corner),
    the center (2,2) scores 2.

    Args:
        x: File/column [0, 4].
        y: Rank/row [0, 4].

    Returns:
        Chebyshev distance to the closest corner.
    """
    corners = [(0, 0), (0, 4), (4, 0), (4, 4)]
    return min(max(abs(x - cx), abs(y - cy)) for (cx, cy) in corners)


# =====================================================================
# === Hoisted helper functions (top-level single definitions)
# =====================================================================

def file_mask(file_x: int) -> int:
    return FILE_MASKS[file_x]

def rank_mask(rank_y: int) -> int:
    return RANK_MASKS[rank_y]

def ring1_mask(kx: int, ky: int) -> int:
    return RING1_MASKS[square_index(kx, ky)]

def count_attackers_to_zone(bb: "BitboardState", zone_mask: int, white_attacking: bool) -> dict:
    """
    Counts the number of pieces attacking any square in a target zone.
    
    This function analyzes attack patterns to determine how many pieces of
    each type can attack squares within a given zone mask. Used primarily for
    king safety evaluation, where the zone is typically the ring-1 around the king.
    
    Args:
        bb: BitboardState containing all piece bitboards.
        zone_mask: A 25-bit bitboard where set bits represent target squares
            in the zone to check for attacks. Typically ring1_mask() around a king.
        white_attacking: True to count white attackers, False to count black attackers.
    
    Returns:
        A dictionary with keys "Knight", "Bishop", "Rook", "Right", "Queen"
        and integer values representing the count of each piece type that attacks
        at least one square in the zone. King and pawn attacks are not counted
        (handled separately in king safety evaluation).
    
    Notes:
        - Uses precomputed attack masks for knights
        - Uses sliding attack functions for bishops, rooks, and queens
        - Right piece uses both rook and knight attacks
        - Used in king safety evaluation to weight attack strength
    """
    counts = {"Knight": 0, "Bishop": 0, "Rook": 0, "Right": 0, "Queen": 0}
    occ = bb.occ_all
    if white_attacking:
        for sq in _iter_set_bits(bb.WN):
            if _KNIGHT_MOVES[sq] & zone_mask:
                counts["Knight"] += 1
        for sq in _iter_set_bits(bb.WB):
            if _get_bishop_attacks(sq, occ) & zone_mask:
                counts["Bishop"] += 1
        for sq in _iter_set_bits(bb.WR):
            if _get_rook_attacks(sq, occ) & zone_mask:
                counts["Rook"] += 1
        for sq in _iter_set_bits(bb.WRi):
            if (_get_rook_attacks(sq, occ) | _KNIGHT_MOVES[sq]) & zone_mask:
                counts["Right"] += 1
        for sq in _iter_set_bits(bb.WQ):
            if (_get_bishop_attacks(sq, occ) | _get_rook_attacks(sq, occ)) & zone_mask:
                counts["Queen"] += 1
    else:
        for sq in _iter_set_bits(bb.BN):
            if _KNIGHT_MOVES[sq] & zone_mask:
                counts["Knight"] += 1
        for sq in _iter_set_bits(bb.BB):
            if _get_bishop_attacks(sq, occ) & zone_mask:
                counts["Bishop"] += 1
        for sq in _iter_set_bits(bb.BR):
            if _get_rook_attacks(sq, occ) & zone_mask:
                counts["Rook"] += 1
        for sq in _iter_set_bits(bb.BRi):
            if (_get_rook_attacks(sq, occ) | _KNIGHT_MOVES[sq]) & zone_mask:
                counts["Right"] += 1
        for sq in _iter_set_bits(bb.BQ):
            if (_get_bishop_attacks(sq, occ) | _get_rook_attacks(sq, occ)) & zone_mask:
                counts["Queen"] += 1
    return counts


def king_shield_penalty(bb: "BitboardState", kx: int, ky: int, white_defender: bool) -> int:
    """
    Evaluates the pawn shield in front of the king and applies penalties for weaknesses.
    
    A strong pawn shield (pawns directly in front of the king) is crucial for
    king safety. This function checks the three squares directly in front of the
    king and penalizes positions where the shield is weak or missing.
    
    Args:
        bb: BitboardState containing WP and BP pawn bitboards.
        kx: King's file (column) coordinate [0, 4].
        ky: King's rank (row) coordinate [0, 4].
        white_defender: True if evaluating white king's shield, False for black.
    
    Returns:
        An integer penalty score (non-positive, added to king safety evaluation):
        - 0: Strong shield (2-3 pawns in front of king)
        - EVAL_KING_SHIELD_WEAK (typically -10): Weak shield (1 pawn)
        - EVAL_KING_SHIELD_GONE (typically -20): No shield (0 pawns)
    
    Notes:
        - Checks the three squares directly in front of the king
        - For white: checks squares at rank ky-1 (in front = towards rank 0)
        - For black: checks squares at rank ky+1 (in front = towards rank 4)
        - Penalties are negative (reduce king safety score)
    """
    penalty = 0
    front_dy = -1 if white_defender else 1
    pxs = []
    for dx in (-1, 0, 1):
        fx, fy = kx + dx, ky + front_dy
        if 0 <= fx < 5 and 0 <= fy < 5:
            pxs.append(square_index(fx, fy))
    have = 0
    for psq in pxs:
        if white_defender:
            if bb.WP & (1 << psq):
                have += 1
        else:
            if bb.BP & (1 << psq):
                have += 1
    if have <= 1:
        penalty += EVAL_KING_SHIELD_GONE if have == 0 else EVAL_KING_SHIELD_WEAK
    return penalty


def open_file_to_king_penalty(bb: "BitboardState", kx: int, ky: int, white_defender: bool) -> int:
    """
    Evaluates penalties for the king being on an open or semi-open file with enemy pieces.
    
    A king on an open file (no pawns of either color) or semi-open file (no friendly
    pawns) is vulnerable to attack, especially if the opponent has heavy pieces
    (rooks, right, queens) on that file. This function detects this vulnerability.
    
    Args:
        bb: BitboardState containing all piece bitboards.
        kx: King's file (column) coordinate [0, 4].
        ky: King's rank (row) coordinate [0, 4] (unused but kept for API consistency).
        white_defender: True if evaluating white king, False for black.
    
    Returns:
        An integer penalty score (non-positive):
        - 0: Safe (no enemy heavy pieces on file, or friendly pawn blocks)
        - EVAL_SEMIOPEN_FILE_TO_KING (typically -10): Semi-open file with enemy heavies
        - EVAL_OPEN_FILE_TO_KING (typically -18): Open file with enemy heavies
    
    Notes:
        - Checks if opponent has rooks, right pieces, or queens on the king's file
        - Open file: No pawns of either color on the file (most dangerous)
        - Semi-open file: No friendly pawns, but opponent has pawns (less dangerous)
        - Only applies penalty if opponent has heavy pieces on the file
    """
    f = kx
    fmask = file_mask(f)
    if white_defender:
        my_pawn_on_file = bool(bb.WP & fmask)
        opp_pawn_on_file = bool(bb.BP & fmask)
        opp_heavies_on_file = bool(((bb.BR | bb.BRi | bb.BQ) & fmask))
    else:
        my_pawn_on_file = bool(bb.BP & fmask)
        opp_pawn_on_file = bool(bb.WP & fmask)
        opp_heavies_on_file = bool(((bb.WR | bb.WRi | bb.WQ) & fmask))
    if opp_heavies_on_file and not my_pawn_on_file:
        return EVAL_OPEN_FILE_TO_KING if not opp_pawn_on_file else EVAL_SEMIOPEN_FILE_TO_KING
    return 0


def same_file_pressure(bb: "BitboardState", kx: int, white_defender: bool) -> int:
    """
    Evaluates additional pressure from enemy pieces on the king's file.
    
    This function counts enemy rooks, right pieces, and queens that are on the
    same file as the king, adding incremental pressure penalties. This complements
    open_file_to_king_penalty() by providing additional penalties for each piece.
    
    Args:
        bb: BitboardState containing all piece bitboards.
        kx: King's file (column) coordinate [0, 4].
        white_defender: True if evaluating white king, False for black.
    
    Returns:
        An integer penalty score (non-positive) representing file pressure:
        - EVAL_KING_FILE_PRESSURE (typically -2) per enemy rook or right piece
        - EVAL_KING_FILE_PRESSURE + 1 (typically -3) per enemy queen
        - Summed across all enemy heavy pieces on the file
    
    Notes:
        - Provides incremental penalties for each piece on the file
        - Queens get slightly higher penalty than rooks/right
        - Used in conjunction with open_file_to_king_penalty() for comprehensive evaluation
    """
    f = kx
    score = 0
    if white_defender:
        for sq in _iter_set_bits(bb.BR):
            rx, _ = index_to_sq(sq)
            if rx == f:
                score += EVAL_KING_FILE_PRESSURE
        for sq in _iter_set_bits(bb.BRi):
            rx, _ = index_to_sq(sq)
            if rx == f:
                score += EVAL_KING_FILE_PRESSURE
        for sq in _iter_set_bits(bb.BQ):
            qx, _ = index_to_sq(sq)
            if qx == f:
                score += EVAL_KING_FILE_PRESSURE + 1
    else:
        for sq in _iter_set_bits(bb.WR):
            rx, _ = index_to_sq(sq)
            if rx == f:
                score += EVAL_KING_FILE_PRESSURE
        for sq in _iter_set_bits(bb.WRi):
            rx, _ = index_to_sq(sq)
            if rx == f:
                score += EVAL_KING_FILE_PRESSURE
        for sq in _iter_set_bits(bb.WQ):
            qx, _ = index_to_sq(sq)
            if qx == f:
                score += EVAL_KING_FILE_PRESSURE + 1
    return score


def is_white_knight_outpost(bb: "BitboardState", sq: int) -> bool:
    """
    Determines if a white knight is on an outpost square.
    
    A knight outpost is a square where a knight is:
    1. Protected by a friendly pawn (pawn behind the knight)
    2. Not attackable by enemy pawns (no enemy pawns can attack the square)
    
    Outpost knights are very strong because they're safe from pawn attacks and
    can't be easily dislodged. This function checks these conditions for white knights.
    
    Args:
        bb: BitboardState containing WP and BP pawn bitboards.
        sq: Square index [0, 24] where the white knight is located.
    
    Returns:
        True if the knight at sq is on an outpost (protected by pawn and not
        attackable by enemy pawns), False otherwise.
    
    Notes:
        - Checks for friendly pawns behind the knight (at rank y+1)
        - Checks that no enemy pawns can attack the square (no black pawns at y-1)
        - Outpost knights receive evaluation bonuses
    """
    x, y = index_to_sq(sq)
    protect = 0
    for dx in (-1, 1):
        px, py = x + dx, y + 1
        if 0 <= px < 5 and 0 <= py < 5:
            if bb.WP & (1 << square_index(px, py)):
                protect += 1
    if protect == 0:
        return False
    for dx in (-1, 1):
        px, py = x + dx, y - 1
        if 0 <= px < 5 and 0 <= py < 5:
            if bb.BP & (1 << square_index(px, py)):
                return False
    return True


def is_black_knight_outpost(bb: "BitboardState", sq: int) -> bool:
    """
    Determines if a black knight is on an outpost square.
    
    A knight outpost is a square where a knight is:
    1. Protected by a friendly pawn (pawn behind the knight)
    2. Not attackable by enemy pawns (no enemy pawns can attack the square)
    
    Outpost knights are very strong because they're safe from pawn attacks and
    can't be easily dislodged. This function checks these conditions for black knights.
    
    Args:
        bb: BitboardState containing WP and BP pawn bitboards.
        sq: Square index [0, 24] where the black knight is located.
    
    Returns:
        True if the knight at sq is on an outpost (protected by pawn and not
        attackable by enemy pawns), False otherwise.
    
    Notes:
        - Checks for friendly pawns behind the knight (at rank y-1, since black moves down)
        - Checks that no enemy pawns can attack the square (no white pawns at y+1)
        - Outpost knights receive evaluation bonuses
    """
    x, y = index_to_sq(sq)
    protect = 0
    for dx in (-1, 1):
        px, py = x + dx, y - 1
        if 0 <= px < 5 and 0 <= py < 5:
            if bb.BP & (1 << square_index(px, py)):
                protect += 1
    if protect == 0:
        return False
    for dx in (-1, 1):
        px, py = x + dx, y + 1
        if 0 <= px < 5 and 0 <= py < 5:
            if bb.WP & (1 << square_index(px, py)):
                return False
    return True

# =====================================================================
# === Main Agent Entry Point
# =====================================================================


def agent(board, player, var):
    """
    Computes the best move from the current position using iterative deepening.

    Runs a negamax search from `START_SEARCH_DEPTH` to `MAX_SEARCH_DEPTH`,
    honoring optional soft/hard time budgets and using aspiration windows
    after the first iteration. Maintains and reuses search state across calls
    (TT, history, counters) when `var` is a persistent dictionary.

    Args:
        board: Current board.
        player: Player to move.
        var: Mutable state/config dict. Recognized keys include:
            - "zobrist": Zobrist hasher (created if missing)
            - "transposition_table": TT instance (created if missing)
            - "time_budget_s": Optional soft time in seconds (float)
            - "flags": Dict controlling heuristics {"nmp","lmr","futility","qsee"}
            - Internal counters/diagnostics updated by the search

    Returns:
        Tuple (best_piece, best_move_opt). If no legal moves, returns (None, None).

    How it works:
        - Initializes or reuses TT, history, killers, and hash history.
        - Computes the current Zobrist hash and runs ID-DFS with aspiration.
        - After each iteration, updates `_LAST_SEARCH_INFO` with diagnostics
          including PV reconstruction and evaluation breakdowns before/after
          the best move.
    """
    global _PERSISTENT_VAR
    var_state: Dict[str, Any] = var if isinstance(var, dict) else _PERSISTENT_VAR

    # Initialize logging on first move
    # global LOG_FILE, MOVE_COUNTER
    # if LOG_FILE is None:
    #     init_log_file()
    # MOVE_COUNTER += 1
    
    # Track time across the game and per move
    current_move_start_time = time.perf_counter()
    var_state.setdefault("total_time_used_s", 0.0)
    var_state.setdefault("game_ply", 0)
    var_state["game_ply"] += 1
    
    ply_so_far = var_state["game_ply"]
    # log_message(f"\n{'='*60}")
    # log_message(f"Move #{MOVE_COUNTER} (Ply {ply_so_far})")
    # log_message(f"{'='*60}")

    var_state.setdefault("zobrist", Zobrist())
    var_state.setdefault("contempt", 0.0)
    var_state.setdefault("capture_history", {})
    var_state.setdefault("cont_history", {})
    var_state.setdefault("verbose", False)
    var_state.setdefault(
        "flags",
        {
            "nmp": True,
            "lmr": True,
            "futility": True,
            "qsee": True,
            "inplace": True,
            "rfp": True,
        },
    )

    if "transposition_table" not in var_state:
        var_state["transposition_table"] = TranspositionTable()

    if not var_state.get("_initialized_this_game", False):
        var_state["transposition_table"].clear()
        var_state["history"] = {}
        var_state["countermoves"] = {}
        var_state["killers"] = [[None, None] for _ in range(MAX_SEARCH_DEPTH + 10)]
        var_state["_rep_stack"] = []
        var_state["_initialized_this_game"] = True

    # Time Management Logic (dynamic budgeting over a 300s total)
    GAME_TIME_S = 300.0
    MOVE_HARD_LIMIT_S = 13.8
    remaining_time_s = GAME_TIME_S - var_state["total_time_used_s"]
    # External per-move budget handling (var may be [ply, budget_s])
    DEFAULT_THINKING_TIME = 13.0
    try:
        if isinstance(var, list) and len(var) == 2:
            try:
                cur_ply_from_harness = int(var[0])
                var_state["game_ply"] = cur_ply_from_harness
            except Exception:
                pass
            try:
                THINKING_TIME_BUDGET = float(var[1])
            except Exception:
                THINKING_TIME_BUDGET = DEFAULT_THINKING_TIME
        else:
            THINKING_TIME_BUDGET = DEFAULT_THINKING_TIME
    except Exception:
        THINKING_TIME_BUDGET = DEFAULT_THINKING_TIME

    ply_so_far = var_state["game_ply"]
    moves_remaining = max(10, 40 - (ply_so_far // 2))
    # dynamic_budget_s = (remaining_time_s / moves_remaining)  # Computed but not currently used
    soft_time = 12.5 # max(0.3, min(dynamic_budget_s, THINKING_TIME_BUDGET * 0.9))
    hard_time = max(0.5, min(MOVE_HARD_LIMIT_S, THINKING_TIME_BUDGET * 0.98))
    start_t = current_move_start_time
    var_state["_start_t"] = start_t
    var_state["_soft_time_s"] = soft_time
    var_state["_hard_time_s"] = hard_time
    var_state["_soft_time_stop"] = False
    var_state["_hard_time_stop"] = False
    try:
        _LAST_SEARCH_INFO.clear()
    except Exception:
        pass
    # if var_state.get("verbose", False):
    print(f"DEBUG: Ply {ply_so_far}, Rem_Time {remaining_time_s:.2f}s, Moves_Rem {moves_remaining}, Budget {soft_time:.2f}s")

    # --- Bitboard bridge: run the internal engine and return (piece, move_opt) ---
    try:
        zobrist_hasher: Zobrist = var_state["zobrist"]
        # Build root move map once for mapping back BBMove -> (piece, move_opt)
        try:
            root_legal_moves = list_legal_moves_for(board, player)
        except Exception:
            root_legal_moves = []
        root_moves_map: Dict[Tuple[int, int, int, int], Tuple[Any, Any]] = {}
        for p, m in root_legal_moves:
            if p is None or m is None:
                continue
            try:
                sx, sy = p.position.x, p.position.y
                dx, dy = m.position.x, m.position.y
            except Exception:
                continue
            root_moves_map[(sx, sy, dx, dy)] = (p, m)
        var_state["_root_moves_map"] = root_moves_map
        # If no legal moves at root, return immediately (terminal position)
        if not root_moves_map:
            total_time = time.perf_counter() - current_move_start_time
            # log_message("ROOT: No legal moves (terminal). Returning (None, None).")
            try:
                var_state["total_time_used_s"] += total_time
            except Exception:
                pass
            if not isinstance(var, dict):
                _PERSISTENT_VAR = var_state
            return None, None

        bb0 = convert_board_to_bb_state(board, player, zobrist_hasher)
        # Record root hash for repetition across game
        var_state.setdefault("_bb_game_hist", []).append(bb0.zkey)
        var_state["_nodes"] = 0
        var_state["_qnodes"] = 0
        var_state["transposition_table"].hits = 0
        var_state["transposition_table"].probes = 0

        best_move_bb: Optional[BBMove] = None
        best_score = -INF
        ASPIRATION_WINDOW_DELTA = 50
        alpha_base = -INF
        beta_base = INF
        
        # log_message(f"Time budget: soft={soft_time:.2f}s, hard={hard_time:.2f}s")
        # Adaptive time management trackers per-iteration
        # prev_iter_best_score: Optional[float] = None  # Declared but never used
        prev_iter_best_tuple: Optional[Tuple[int, int, int, int]] = None

        for depth in range(START_SEARCH_DEPTH, MAX_SEARCH_DEPTH + 1):
            var_state["_id_depth"] = depth
            depth_start_time = time.perf_counter()
            # Time guard
            elapsed = time.perf_counter() - start_t
            if hard_time > 0 and elapsed >= hard_time:
                var_state["_hard_time_stop"] = True
                break
            if soft_time > 0 and elapsed >= soft_time:
                var_state["_soft_time_stop"] = True
                break

            if depth == START_SEARCH_DEPTH or best_score in (-INF, INF):
                alpha = alpha_base
                beta = beta_base
                delta = None
            else:
                delta = ASPIRATION_WINDOW_DELTA
                alpha = best_score - delta
                beta = best_score + delta

            attempt = 0
            max_attempts = 2
            used_full_window = False
            depth_completed = False
            try:
                while True:
                    score, mv = negamax_bb(
                        bb0,
                        depth,
                        alpha,
                        beta,
                        var_state,
                        0,
                    )
                    if delta is not None and score <= alpha:
                        if attempt < max_attempts:
                            attempt += 1
                            delta *= 2
                            alpha = best_score - delta
                            beta = best_score + delta
                            continue
                        if not used_full_window:
                            alpha = alpha_base
                            beta = beta_base
                            used_full_window = True
                            continue
                        break
                    elif delta is not None and score >= beta:
                        if attempt < max_attempts:
                            attempt += 1
                            delta *= 2
                            alpha = best_score - delta
                            beta = best_score + delta
                            continue
                        if not used_full_window:
                            alpha = alpha_base
                            beta = beta_base
                            used_full_window = True
                            continue
                        break
                    # Success within window
                    best_score = score
                    best_move_bb = mv if mv is not None else best_move_bb
                    depth_completed = True
                    break
            except Exception as e:
                # Timeout or other error during search - break to preserve best_move_bb from previous depth
                if "time limit" in str(e).lower() or "timeout" in str(e).lower():
                    # if var_state.get("verbose", False):
                    print(f"[BB_AGENT] Timeout at depth {depth}, using best move from previous depth")
                    # log_message(f"Depth {depth}: TIMEOUT")
                else:
                    if var_state.get("verbose", False):
                        print(f"[BB_AGENT] Error at depth {depth}: {e}")
                    # log_message(f"Depth {depth}: ERROR - {e}")
                break  # Exit the depth loop, preserving best_move_bb from previous iterations

            if depth_completed:
                depth_time = time.perf_counter() - depth_start_time
                nodes = var_state.get('_nodes', 0)
                qnodes = var_state.get('_qnodes', 0)
                move_str = "None"
                if best_move_bb:
                    sx, sy, dx, dy = bbmove_to_tuple_xy(best_move_bb)
                    move_str = f"({sx},{sy})->({dx},{dy})"
                
                # if var_state.get("verbose", False):
                print(f"DEBUG[BB]: Depth {depth} finished. Nodes={nodes}, QNodes={qnodes}, time={depth_time:.3f}s")
                # log_message(f"Depth {depth}: score={best_score:.0f}, move={move_str}, time={depth_time:.3f}s, nodes={nodes}, qnodes={qnodes}")
                # Adaptive soft time tuning based on stability and score magnitude
                try:
                    cur_tuple = None
                    if best_move_bb:
                        cur_tuple = bbmove_to_tuple_xy(best_move_bb)
                    unstable = (prev_iter_best_tuple is not None and cur_tuple is not None and cur_tuple != prev_iter_best_tuple)
                    big_adv = (abs(best_score) > (WIN_VALUE / 10))
                    new_soft = var_state.get("_soft_time_s", soft_time)
                    if big_adv:
                        new_soft *= 0.7
                    elif unstable or abs(best_score) < 50:
                        new_soft *= 1.25
                    # Clamp
                    new_soft = max(0.3, min(new_soft, MOVE_HARD_LIMIT_S * 0.95))
                    var_state["_soft_time_s"] = new_soft
                    # prev_iter_best_score = best_score  # Assigned but never used
                    prev_iter_best_tuple = cur_tuple
                    var_state["_root_best_move_tuple"] = cur_tuple
                except Exception:
                    pass

        # Convert internal move to board objects
        total_time = time.perf_counter() - current_move_start_time
        total_nodes = var_state.get('_nodes', 0)
        total_qnodes = var_state.get('_qnodes', 0)
        
        if best_move_bb is not None:
            sx, sy, dx, dy = bbmove_to_tuple_xy(best_move_bb)
            # log_message(f"Mapping root move via precomputed table: move_tuple=({sx},{sy},{dx},{dy}), best_move_bb={best_move_bb}")
            pair = var_state.get("_root_moves_map", {}).get((sx, sy, dx, dy))
            if pair:
                piece, move = pair
                move_str = log_move_str(piece, move)
                # log_message(f"FINAL: {move_str}, score={best_score:.0f}, time={total_time:.3f}s, nodes={total_nodes}, qnodes={total_qnodes}")
                # log_message(f"{'='*60}\n")
                try:
                    var_state["total_time_used_s"] += total_time
                except Exception:
                    pass
                if not isinstance(var, dict):
                    _PERSISTENT_VAR = var_state
                return pair
            # Fallback if mapping failed; try random legal
        # Fallback: random legal move if no bb move found
        # Deterministic fallback: pick first from precomputed root moves if any
        rm = var_state.get("_root_moves_map", {})
        if rm:
            first_tuple = next(iter(rm.keys()))
            piece, move = rm[first_tuple]
            move_str = log_move_str(piece, move)
            # log_message(f"FALLBACK (deterministic): {move_str}, time={total_time:.3f}s")
            # log_message(f"{'='*60}\n")
            try:
                var_state["total_time_used_s"] += total_time
            except Exception:
                pass
            if not isinstance(var, dict):
                _PERSISTENT_VAR = var_state
            return piece, move
        else:
            # log_message("FALLBACK: No legal moves found at root (terminal).")
            # log_message(f"{'='*60}\n")
            try:
                var_state["total_time_used_s"] += total_time
            except Exception:
                pass
            if not isinstance(var, dict):
                _PERSISTENT_VAR = var_state
            return None, None
    except Exception as e:
        if var_state.get("verbose", False):
            print(f"[BB_AGENT_ERROR] {e}")

        # Return the best move from the *previous* completed depth if available
        total_time = time.perf_counter() - current_move_start_time
        if best_move_bb is not None:
            if var_state.get("verbose", False):
                print("[BB_AGENT_FIX] Using best move from previous depth due to timeout.")
            sx, sy, dx, dy = bbmove_to_tuple_xy(best_move_bb)
            # log_message(f"[TIMEOUT] Mapping via precomputed root table for move_tuple=({sx},{sy},{dx},{dy})")
            pair = var_state.get("_root_moves_map", {}).get((sx, sy, dx, dy))
            if pair:
                piece, move = pair
                move_str = log_move_str(piece, move)
                # log_message(f"FINAL (timeout): {move_str}, time={total_time:.3f}s")
                # log_message(f"{'='*60}\n")
                try:
                    var_state["total_time_used_s"] += total_time
                except Exception:
                    pass
                if not isinstance(var, dict):
                    _PERSISTENT_VAR = var_state
                return pair
        
        # If we timed out on depth 1 (best_move_bb is None), fall to random.
        try:
            best_tuple = var_state.get("_root_best_move_tuple")
            if best_tuple:
                # log_message(f"[FALLBACK] Mapping best_tuple via precomputed table: {best_tuple}")
                fallback_pair = var_state.get("_root_moves_map", {}).get(tuple(best_tuple))
                if fallback_pair:
                    piece, move = fallback_pair
                    move_str = log_move_str(piece, move)
                    # if var_state.get("verbose", False):
                    print(f"FALLBACK (using previous depth): {move_str}, time={total_time:.3f}s")
                    return fallback_pair
        except Exception:
            pass # Failed to use previous depth's move, proceed to random.
        # --- END FIX ---

        # Deterministic final fallback using precomputed root moves map
        rm = var_state.get("_root_moves_map", {})
        if rm:
            first_tuple = next(iter(rm.keys()))
            piece, move = rm[first_tuple]
            # log_message(f"FALLBACK (deterministic final): {log_move_str(piece, move)}, time={total_time:.3f}s")
            return piece, move
        # log_message("FALLBACK: No legal moves found (terminal).")
        return None, None