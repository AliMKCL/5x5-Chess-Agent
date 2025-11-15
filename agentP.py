# agent.py
import random
import math
import time
import sys
from typing import Dict, Tuple, Optional, List, Any, NamedTuple
from dataclasses import dataclass
from chessmaker.chess.base import Square, Position
from extension.board_utils import list_legal_moves_for, copy_piece_move
# Logging/diagnostics flags
DEBUG = False
ENABLE_TREE_LOGGING = False
DEBUG_LEGAL = True

# =====================================================================
# === Logging Infrastructure
# =====================================================================

LOG_FILE = None
MOVE_COUNTER = 0

def init_log_file():
    """Initialize the log file for this game session."""
    global LOG_FILE, MOVE_COUNTER
    LOG_FILE = open("agent_log.txt", "a")
    LOG_FILE.write("=== GAME LOG (agent.py) ===\n\n")
    MOVE_COUNTER = 0

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

def log_move_str(piece, move_opt):
    """Convert move to string format for logging."""
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
    cache_key = (board_hash, player.name)
    if cache_key not in var.setdefault('_legal_moves_cache', {}):
        var['_legal_moves_cache'][cache_key] = list_legal_moves_for(board, player)
    return var['_legal_moves_cache'][cache_key]



def _json_safe_number(x: Any):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            if math.isinf(x):
                return "inf" if x > 0 else "-inf"
            if math.isnan(x):
                return "nan"
            return float(x)
        return x
    except Exception:
        return x




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
    return y * 5 + x


def index_to_sq(idx: int) -> Tuple[int, int]:
    """
    Converts a packed 0..24 index back to (x, y) board coordinates.

    Args:
        idx: Linear index in [0, 24].

    Returns:
        A tuple (x, y) with x,y in [0, 4].
    """
    return (idx % 5, idx // 5)


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
    Extracts a compact occupancy mask for a line from the global 25-bit occ.
    Bit i in the returned mask corresponds to line[i].
    """
    mask = 0
    for i, sq in enumerate(line):
        if occ & (1 << sq):
            mask |= (1 << i)
    return mask


def _get_rook_attacks(sq: int, occ: int) -> int:
    """
    Constant-time rook attacks via precomputed rank/file tables.
    Falls back to scan if tables are unavailable.
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
    Constant-time bishop attacks via precomputed diagonal tables.
    Falls back to scan if tables are unavailable.
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


 
 
@dataclass
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


def convert_board_to_bb_state(board, player, zobrist: "Zobrist") -> BitboardState:
    """
    One-time bridge from chessmaker board to the internal BitboardState.
    """
    bbpos = bb_from_board(board)
    stm = 0 if getattr(player, "name", "white") == "white" else 1
    try:
        zkey = zobrist.compute_full_hash(board, player.name)
    except Exception:
        zkey = 0
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
    )


@dataclass(frozen=True)
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


def bbmove_to_tuple_xy(m: "BBMove") -> Tuple[int, int, int, int]:
    """
    Converts BBMove to (sx, sy, dx, dy) tuple for compatibility with find_move_from_tuple.
    """
    sx, sy = index_to_sq(m.from_sq)
    dx, dy = index_to_sq(m.to_sq)
    return sx, sy, dx, dy


# --- Helpers for bit iteration and capture typing ---
def _iter_set_bits(bb: int):
    while bb:
        lsb = bb & -bb
        idx = lsb.bit_length() - 1
        yield idx
        bb ^= lsb


def _get_piece_type_at_square(bb: "BitboardState", color_white: bool, sq: int) -> int:
    """
    Returns piece type index 0..6 for piece of given color at sq, or -1 if empty.
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
    Given full list [WP,WN,WB,WR,WQ,WK,WRi,BP,BN,BB,BR,BQ,BK,BRi] returns (occ_white, occ_black, occ_all).
    """
    occ_white = bbvals[0] | bbvals[1] | bbvals[2] | bbvals[3] | bbvals[4] | bbvals[5] | bbvals[6]
    occ_black = bbvals[7] | bbvals[8] | bbvals[9] | bbvals[10] | bbvals[11] | bbvals[12] | bbvals[13]
    return occ_white, occ_black, (occ_white | occ_black)


def apply_bb_move(bb: "BitboardState", m: "BBMove", zobrist: Optional["Zobrist"]) -> "BitboardState":
    """
    Applies a move to produce a new BitboardState. Uses incremental Zobrist updates when available.
    """
    stm_white = (bb.side_to_move == 0)
    # Copy all piece bitboards into list for easier updates
    parts = [
        bb.WP, bb.WN, bb.WB, bb.WR, bb.WQ, bb.WK, bb.WRi,
        bb.BP, bb.BN, bb.BB, bb.BR, bb.BQ, bb.BK, bb.BRi,
    ]
    from_mask = 1 << m.from_sq
    to_mask = 1 << m.to_sq
    color_idx = 0 if stm_white else 1
    zkey = bb.zkey

    # Remove captured piece if any
    if m.captured_type >= 0:
        opp_offset = 7 if stm_white else 0
        cap_idx = opp_offset + m.captured_type
        if parts[cap_idx] & to_mask:
            parts[cap_idx] ^= to_mask
            if zobrist:
                zkey = zobrist.toggle_by_indices(zkey, m.captured_type, 1 - color_idx, m.to_sq)

    # Move the piece (and handle promotion replacement)
    my_offset = 0 if stm_white else 7
    src_idx = my_offset + m.piece_type
    # Remove from source
    parts[src_idx] ^= from_mask
    if zobrist:
        zkey = zobrist.toggle_by_indices(zkey, m.piece_type, color_idx, m.from_sq)
    # Destination: either same piece or promotion replacement
    if m.promo == 4 and m.piece_type == 0:
        # Promotion to Queen replaces pawn with queen at destination
        dst_idx = my_offset + 4
        parts[dst_idx] |= to_mask
        if zobrist:
            zkey = zobrist.toggle_by_indices(zkey, 4, color_idx, m.to_sq)
    else:
        parts[src_idx] |= to_mask
        if zobrist:
            zkey = zobrist.toggle_by_indices(zkey, m.piece_type, color_idx, m.to_sq)

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
    )


@dataclass(frozen=True)
class UndoInfo:
    parent: BitboardState


def make_move(bb: "BitboardState", m: "BBMove", zobrist: Optional["Zobrist"]) -> Tuple["BitboardState", "UndoInfo"]:
    """
    Creates a child state and returns it with an undo token for fast restoration.
    """
    child = apply_bb_move(bb, m, zobrist)
    return child, UndoInfo(parent=bb)


def unmake_move(_child: "BitboardState", undo: "UndoInfo") -> "BitboardState":
    """
    Restores the previous state using the undo token.
    """
    return undo.parent


def _gen_sliding_moves(from_sq: int, occ_all: int, own_occ: int, attack_fn) -> Tuple[int, int]:
    """
    Returns (quiet_mask, capture_mask) for a sliding piece from 'from_sq'.
    attack_fn should be one of _get_rook_attacks or _get_bishop_attacks.
    """
    attacks = attack_fn(from_sq, occ_all)
    legal = attacks & (~own_occ)
    # Quiet squares are those not occupied
    quiet = legal & (~occ_all)
    captures = legal & occ_all
    return quiet, captures


def generate_legal_moves(bb: "BitboardState", captures_only: bool = False) -> List["BBMove"]:
    """
    Pure-legal move generator using one-time legality masks (checkers and pins).
    Supports captures-only mode for quiescence.
    """
    moves: List[BBMove] = []
    stm_white = (bb.side_to_move == 0)
    own_occ = bb.occ_white if stm_white else bb.occ_black
    opp_occ = bb.occ_black if stm_white else bb.occ_white
    occ_all = bb.occ_all

    kbb = bb.WK if stm_white else bb.BK
    if kbb == 0:
        return moves
    ksq = _pop_lsb_njit(kbb)

    checkers_bb, pin_mask, pin_ray_map = calculate_legality_masks(bb, ksq, stm_white)
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
        moves.append(BBMove(ksq, to_sq, 0, 5, cap_type))

    if is_double_check:
        return moves

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
        moves.append(BBMove(from_sq, to_sq, promo, 0, -1))

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
        moves.append(BBMove(from_sq, to_sq, promo, 0, cap_type))

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
        moves.append(BBMove(from_sq, to_sq, promo, 0, cap_type))

    # Knights
    N = bb.WN if stm_white else bb.BN
    for from_sq in _iter_set_bits(N):
        targets = _KNIGHT_MOVES[from_sq] & (~own_occ) & target_mask
        if pin_mask & (1 << from_sq):
            targets &= pin_ray_map.get(from_sq, 0)
        if captures_only:
            targets &= opp_occ
        for to_sq in _iter_set_bits(targets):
            cap_type = _get_piece_type_at_square(bb, not stm_white, to_sq) if (opp_occ & (1 << to_sq)) else -1
            moves.append(BBMove(from_sq, to_sq, 0, 1, cap_type))

    # Bishops
    B = bb.WB if stm_white else bb.BB
    for from_sq in _iter_set_bits(B):
        targets = _get_bishop_attacks(from_sq, occ_all) & (~own_occ) & target_mask
        if pin_mask & (1 << from_sq):
            targets &= pin_ray_map.get(from_sq, 0)
        if captures_only:
            targets &= opp_occ
        for to_sq in _iter_set_bits(targets):
            cap_type = _get_piece_type_at_square(bb, not stm_white, to_sq) if (opp_occ & (1 << to_sq)) else -1
            moves.append(BBMove(from_sq, to_sq, 0, 2, cap_type))

    # Rooks
    R = bb.WR if stm_white else bb.BR
    for from_sq in _iter_set_bits(R):
        targets = _get_rook_attacks(from_sq, occ_all) & (~own_occ) & target_mask
        if pin_mask & (1 << from_sq):
            targets &= pin_ray_map.get(from_sq, 0)
        if captures_only:
            targets &= opp_occ
        for to_sq in _iter_set_bits(targets):
            cap_type = _get_piece_type_at_square(bb, not stm_white, to_sq) if (opp_occ & (1 << to_sq)) else -1
            moves.append(BBMove(from_sq, to_sq, 0, 3, cap_type))

    # Queens
    Q = bb.WQ if stm_white else bb.BQ
    for from_sq in _iter_set_bits(Q):
        targets = (_get_rook_attacks(from_sq, occ_all) | _get_bishop_attacks(from_sq, occ_all)) & (~own_occ) & target_mask
        if pin_mask & (1 << from_sq):
            targets &= pin_ray_map.get(from_sq, 0)
        if captures_only:
            targets &= opp_occ
        for to_sq in _iter_set_bits(targets):
            cap_type = _get_piece_type_at_square(bb, not stm_white, to_sq) if (opp_occ & (1 << to_sq)) else -1
            moves.append(BBMove(from_sq, to_sq, 0, 4, cap_type))

    # Right (rook | knight)
    Ri = bb.WRi if stm_white else bb.BRi
    for from_sq in _iter_set_bits(Ri):
        r_att = _get_rook_attacks(from_sq, occ_all)
        n_att = _KNIGHT_MOVES[from_sq]
        targets = (r_att | n_att) & (~own_occ) & target_mask
        if pin_mask & (1 << from_sq):
            targets &= pin_ray_map.get(from_sq, 0)
        if captures_only:
            targets &= opp_occ
        for to_sq in _iter_set_bits(targets):
            cap_type = _get_piece_type_at_square(bb, not stm_white, to_sq) if (opp_occ & (1 << to_sq)) else -1
            moves.append(BBMove(from_sq, to_sq, 0, 6, cap_type))

    return moves

# (obsolete generate_all_moves removed in favor of generate_legal_moves)


def _count_bits(x: int) -> int:
    return int(x.bit_count())


def _piece_on_square_color(bb: "BitboardState", sq: int) -> int:
    """
    Returns 1 if white piece, -1 if black piece, 0 if empty on square sq.
    """
    mask = 1 << sq
    if bb.occ_white & mask:
        return 1
    if bb.occ_black & mask:
        return -1
    return 0


def _phase_from_bb(bb: "BitboardState") -> int:
    total = 0
    # White
    total += PHASE_VALUES["Pawn"] * _count_bits(bb.WP)
    total += PHASE_VALUES["Knight"] * _count_bits(bb.WN)
    total += PHASE_VALUES["Bishop"] * _count_bits(bb.WB)
    total += PHASE_VALUES["Rook"] * _count_bits(bb.WR)
    total += PHASE_VALUES["Right"] * _count_bits(bb.WRi)
    total += PHASE_VALUES["Queen"] * _count_bits(bb.WQ)
    # Black
    total += PHASE_VALUES["Pawn"] * _count_bits(bb.BP)
    total += PHASE_VALUES["Knight"] * _count_bits(bb.BN)
    total += PHASE_VALUES["Bishop"] * _count_bits(bb.BB)
    total += PHASE_VALUES["Rook"] * _count_bits(bb.BR)
    total += PHASE_VALUES["Right"] * _count_bits(bb.BRi)
    total += PHASE_VALUES["Queen"] * _count_bits(bb.BQ)
    return max(1, min(total, MAX_PHASE))


def evaluate_bb_state(bb: "BitboardState") -> float:
    """
    Side-to-move centric evaluation (positive is good for bb.side_to_move).
    """
    # Material
    # (base material, no PST)
    w_material_mg = (
        PIECE_VALUES_MG["Pawn"] * _count_bits(bb.WP)
        + PIECE_VALUES_MG["Knight"] * _count_bits(bb.WN)
        + PIECE_VALUES_MG["Bishop"] * _count_bits(bb.WB)
        + PIECE_VALUES_MG["Rook"] * _count_bits(bb.WR)
        + PIECE_VALUES_MG["Right"] * _count_bits(bb.WRi)
        + PIECE_VALUES_MG["Queen"] * _count_bits(bb.WQ)
    )
    b_material_mg = (
        PIECE_VALUES_MG["Pawn"] * _count_bits(bb.BP)
        + PIECE_VALUES_MG["Knight"] * _count_bits(bb.BN)
        + PIECE_VALUES_MG["Bishop"] * _count_bits(bb.BB)
        + PIECE_VALUES_MG["Rook"] * _count_bits(bb.BR)
        + PIECE_VALUES_MG["Right"] * _count_bits(bb.BRi)
        + PIECE_VALUES_MG["Queen"] * _count_bits(bb.BQ)
    )
    w_material_eg = (
        PIECE_VALUES_EG["Pawn"] * _count_bits(bb.WP)
        + PIECE_VALUES_EG["Knight"] * _count_bits(bb.WN)
        + PIECE_VALUES_EG["Bishop"] * _count_bits(bb.WB)
        + PIECE_VALUES_EG["Rook"] * _count_bits(bb.WR)
        + PIECE_VALUES_EG["Right"] * _count_bits(bb.WRi)
        + PIECE_VALUES_EG["Queen"] * _count_bits(bb.WQ)
    )
    b_material_eg = (
        PIECE_VALUES_EG["Pawn"] * _count_bits(bb.BP)
        + PIECE_VALUES_EG["Knight"] * _count_bits(bb.BN)
        + PIECE_VALUES_EG["Bishop"] * _count_bits(bb.BB)
        + PIECE_VALUES_EG["Rook"] * _count_bits(bb.BR)
        + PIECE_VALUES_EG["Right"] * _count_bits(bb.BRi)
        + PIECE_VALUES_EG["Queen"] * _count_bits(bb.BQ)
    )

    mg = w_material_mg - b_material_mg
    eg = w_material_eg - b_material_eg

    # PST contributions (white adds, black subtracts)
    pst_mg = 0
    pst_eg = 0
    # White pieces
    for sq in _iter_set_bits(bb.WP):
        x, y = index_to_sq(sq)
        pst_mg += PSTS_MG["white"]["Pawn"][y][x]
        pst_eg += PSTS_EG["white"]["Pawn"][y][x]
    for sq in _iter_set_bits(bb.WN):
        x, y = index_to_sq(sq)
        pst_mg += PSTS_MG["white"]["Knight"][y][x]
        pst_eg += PSTS_EG["white"]["Knight"][y][x]
    for sq in _iter_set_bits(bb.WB):
        x, y = index_to_sq(sq)
        pst_mg += PSTS_MG["white"]["Bishop"][y][x]
        pst_eg += PSTS_EG["white"]["Bishop"][y][x]
    for sq in _iter_set_bits(bb.WR):
        x, y = index_to_sq(sq)
        pst_mg += PSTS_MG["white"]["Rook"][y][x]
        pst_eg += PSTS_EG["white"]["Rook"][y][x]
    for sq in _iter_set_bits(bb.WRi):
        x, y = index_to_sq(sq)
        pst_mg += PSTS_MG["white"]["Right"][y][x]
        pst_eg += PSTS_EG["white"]["Right"][y][x]
    for sq in _iter_set_bits(bb.WQ):
        x, y = index_to_sq(sq)
        pst_mg += PSTS_MG["white"]["Queen"][y][x]
        pst_eg += PSTS_EG["white"]["Queen"][y][x]
    for sq in _iter_set_bits(bb.WK):
        x, y = index_to_sq(sq)
        pst_mg += PSTS_MG["white"]["King"][y][x]
        pst_eg += PSTS_EG["white"]["King"][y][x]
    # Black pieces
    for sq in _iter_set_bits(bb.BP):
        x, y = index_to_sq(sq)
        pst_mg -= PSTS_MG["black"]["Pawn"][y][x]
        pst_eg -= PSTS_EG["black"]["Pawn"][y][x]
    for sq in _iter_set_bits(bb.BN):
        x, y = index_to_sq(sq)
        pst_mg -= PSTS_MG["black"]["Knight"][y][x]
        pst_eg -= PSTS_EG["black"]["Knight"][y][x]
    for sq in _iter_set_bits(bb.BB):
        x, y = index_to_sq(sq)
        pst_mg -= PSTS_MG["black"]["Bishop"][y][x]
        pst_eg -= PSTS_EG["black"]["Bishop"][y][x]
    for sq in _iter_set_bits(bb.BR):
        x, y = index_to_sq(sq)
        pst_mg -= PSTS_MG["black"]["Rook"][y][x]
        pst_eg -= PSTS_EG["black"]["Rook"][y][x]
    for sq in _iter_set_bits(bb.BRi):
        x, y = index_to_sq(sq)
        pst_mg -= PSTS_MG["black"]["Right"][y][x]
        pst_eg -= PSTS_EG["black"]["Right"][y][x]
    for sq in _iter_set_bits(bb.BQ):
        x, y = index_to_sq(sq)
        pst_mg -= PSTS_MG["black"]["Queen"][y][x]
        pst_eg -= PSTS_EG["black"]["Queen"][y][x]
    for sq in _iter_set_bits(bb.BK):
        x, y = index_to_sq(sq)
        pst_mg -= PSTS_MG["black"]["King"][y][x]
        pst_eg -= PSTS_EG["black"]["King"][y][x]
    mg += pst_mg
    eg += pst_eg

    # Bishop pair
    if _count_bits(bb.WB) >= 2:
        mg += EVAL_BISHOP_PAIR
        eg += EVAL_BISHOP_PAIR
    if _count_bits(bb.BB) >= 2:
        mg -= EVAL_BISHOP_PAIR
        eg -= EVAL_BISHOP_PAIR

    # Center control
    for (cx, cy) in CENTER_SQUARES:
        sq = square_index(cx, cy)
        col = _piece_on_square_color(bb, sq)
        if col == 1:
            mg += EVAL_CENTER_CONTROL
            eg += EVAL_CENTER_CONTROL
        elif col == -1:
            mg -= EVAL_CENTER_CONTROL
            eg -= EVAL_CENTER_CONTROL

    # Pawn structure (cached): doubled, isolated, passed, connected passers, backward
    pe_mg, pe_eg = pawn_eval(bb)
    mg += pe_mg
    eg += pe_eg

    # Open/semi-open files for rook/right presence
    w_has_rook = _count_bits(bb.WR) > 0
    w_has_right = _count_bits(bb.WRi) > 0
    b_has_rook = _count_bits(bb.BR) > 0
    b_has_right = _count_bits(bb.BRi) > 0
    for f in range(5):
        fmask = file_mask(f)
        w_pawns_on_file = bool(bb.WP & fmask)
        b_pawns_on_file = bool(bb.BP & fmask)
        # White rook file
        if w_has_rook:
            if (not w_pawns_on_file) and (not b_pawns_on_file):
                mg += EVAL_ROOK_OPEN_FILE
                eg += EVAL_ROOK_OPEN_FILE
            elif not w_pawns_on_file:
                mg += EVAL_ROOK_SEMIOPEN
                eg += EVAL_ROOK_SEMIOPEN
        # Black rook file
        if b_has_rook:
            if (not w_pawns_on_file) and (not b_pawns_on_file):
                mg -= EVAL_ROOK_OPEN_FILE
                eg -= EVAL_ROOK_OPEN_FILE
            elif not b_pawns_on_file:
                mg -= EVAL_ROOK_SEMIOPEN
                eg -= EVAL_ROOK_SEMIOPEN
        # White Right file
        if w_has_right:
            if (not w_pawns_on_file) and (not b_pawns_on_file):
                mg += EVAL_RIGHT_OPEN_FILE
                eg += EVAL_RIGHT_OPEN_FILE
            elif not w_pawns_on_file:
                mg += EVAL_RIGHT_SEMIOPEN
                eg += EVAL_RIGHT_SEMIOPEN
        # Black Right file
        if b_has_right:
            if (not w_pawns_on_file) and (not b_pawns_on_file):
                mg -= EVAL_RIGHT_OPEN_FILE
                eg -= EVAL_RIGHT_OPEN_FILE
            elif not b_pawns_on_file:
                mg -= EVAL_RIGHT_SEMIOPEN
                eg -= EVAL_RIGHT_SEMIOPEN

    # Advanced King Safety and pressure evaluation
    # Compute king attack scores for both sides (count attackers to ring-1 zone)
    w_k_sq = _pop_lsb_njit(bb.WK) if bb.WK else -1
    b_k_sq = _pop_lsb_njit(bb.BK) if bb.BK else -1
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

    # Apply king safety as differential (black pressure increases our score)
    mg += EVAL_KING_ATTACK_SCALE * (b_king_score - w_king_score)
    eg += EVAL_KING_ATTACK_SCALE * (b_king_score - w_king_score)

    # Endgame drives (KQK / KRK) using bitboards
    def _eg_extras_for_color(white: bool) -> int:
        opp_k = bb.BK if white else bb.WK
        my_k = bb.WK if white else bb.BK
        if opp_k == 0:
            return 0
        ex, ey = index_to_sq(_pop_lsb_njit(opp_k))
        q_count = _count_bits(bb.WQ if white else bb.BQ)
        r_count = _count_bits(bb.WR if white else bb.BR)
        ri_count = _count_bits(bb.WRi if white else bb.BRi)
        opp_non_king = (
            _count_bits((bb.BP | bb.BN | bb.BB | bb.BR | bb.BRi | bb.BQ))
            if white
            else _count_bits((bb.WP | bb.WN | bb.WB | bb.WR | bb.WRi | bb.WQ))
        )
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
                rr_mask = (bb.WR if white else bb.BR) | (bb.WRi if white else bb.BRi)
                for rsq in _iter_set_bits(rr_mask):
                    rx, ry = index_to_sq(rsq)
                    if rx == ex or ry == ey:
                        extra += EVAL_EG_ROOK_CUTOFF
            if is_kqk and my_k:
                q_bb = bb.WQ if white else bb.BQ
                if q_bb:
                    qx, qy = index_to_sq(_pop_lsb_njit(q_bb))
                    kx, ky = index_to_sq(_pop_lsb_njit(my_k))
                    qd = _chebyshev((qx, qy), (ex, ey))
                    kd = _chebyshev((kx, ky), (ex, ey))
                    if qd <= 1 and kd >= 2:
                        extra += EVAL_EG_QUEEN_ADJ_PENALTY
        return extra

    eg += _eg_extras_for_color(True) - _eg_extras_for_color(False)

    # Rooks (and 'Right') on the 7th rank (y=1 for White, y=3 for Black)
    w_seventh_mask = rank_mask(1)
    b_seventh_mask = rank_mask(3)
    w_rooks_on_7th = _count_bits((bb.WR | bb.WRi) & w_seventh_mask)
    b_rooks_on_7th = _count_bits((bb.BR | bb.BRi) & b_seventh_mask)
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
    # Extra if enemy king is on back rank
    if bb.BK:
        bkx, bky = index_to_sq(_pop_lsb_njit(bb.BK))
        if bky == 0 and w_rooks_on_7th:
            mg += EVAL_ROOK_ON_7TH_VS_KING * w_rooks_on_7th
            eg += EVAL_ROOK_ON_7TH_VS_KING * w_rooks_on_7th
    if bb.WK:
        wkx, wky = index_to_sq(_pop_lsb_njit(bb.WK))
        if wky == 4 and b_rooks_on_7th:
            mg -= EVAL_ROOK_ON_7TH_VS_KING * b_rooks_on_7th
            eg -= EVAL_ROOK_ON_7TH_VS_KING * b_rooks_on_7th

    # Knight outposts (protected by own pawn; cannot be attacked by enemy pawn)
    center_set = set(CENTER_SQUARES)
    for sq in _iter_set_bits(bb.WN):
        if is_white_knight_outpost(bb, sq):
            x, y = index_to_sq(sq)
            bonus = EVAL_KNIGHT_OUTPOST
            if (x, y) in center_set:
                bonus += EVAL_KNIGHT_OUTPOST_CENTER_BONUS
            mg += bonus
            eg += bonus
    for sq in _iter_set_bits(bb.BN):
        if is_black_knight_outpost(bb, sq):
            x, y = index_to_sq(sq)
            bonus = EVAL_KNIGHT_OUTPOST
            if (x, y) in center_set:
                bonus += EVAL_KNIGHT_OUTPOST_CENTER_BONUS
            mg -= bonus
            eg -= bonus

    # Good vs Bad bishops (penalize bishops with many own pawns on same color)
    light_mask = 0
    dark_mask = 0
    for ry in range(5):
        for rx in range(5):
            sqi = square_index(rx, ry)
            if ((rx + ry) & 1) == 0:
                light_mask |= 1 << sqi
            else:
                dark_mask |= 1 << sqi
    wp_light = _count_bits(bb.WP & light_mask)
    wp_dark = _count_bits(bb.WP & dark_mask)
    bp_light = _count_bits(bb.BP & light_mask)
    bp_dark = _count_bits(bb.BP & dark_mask)
    for sq in _iter_set_bits(bb.WB):
        pen = wp_light if (light_mask & (1 << sq)) else wp_dark
        mg += EVAL_BAD_BISHOP_PENALTY_PER_PAWN * pen
        eg += EVAL_BAD_BISHOP_PENALTY_PER_PAWN * pen
    for sq in _iter_set_bits(bb.BB):
        pen = bp_light if (light_mask & (1 << sq)) else bp_dark
        mg -= EVAL_BAD_BISHOP_PENALTY_PER_PAWN * pen
        eg -= EVAL_BAD_BISHOP_PENALTY_PER_PAWN * pen

    # Mobility by attacked squares (fast proxy)
    def _pawn_attacks_mask(is_white: bool, pawns_bb: int) -> int:
        mask = 0
        for sq in _iter_set_bits(pawns_bb):
            x, y = index_to_sq(sq)
            if is_white:
                ny = y - 1
                if ny >= 0:
                    if x - 1 >= 0:
                        mask |= 1 << square_index(x - 1, ny)
                    if x + 1 < 5:
                        mask |= 1 << square_index(x + 1, ny)
            else:
                ny = y + 1
                if ny < 5:
                    if x - 1 >= 0:
                        mask |= 1 << square_index(x - 1, ny)
                    if x + 1 < 5:
                        mask |= 1 << square_index(x + 1, ny)
        return mask

    occ = bb.occ_all
    # White mobility
    w_mob = 0
    for sq in _iter_set_bits(bb.WN):
        w_mob += MOBILITY_WEIGHTS["Knight"] * _count_bits(_KNIGHT_MOVES[sq] & ~bb.occ_white)
    for sq in _iter_set_bits(bb.WB):
        w_mob += MOBILITY_WEIGHTS["Bishop"] * _count_bits(_get_bishop_attacks(sq, occ) & ~bb.occ_white)
    for sq in _iter_set_bits(bb.WR):
        w_mob += MOBILITY_WEIGHTS["Rook"] * _count_bits(_get_rook_attacks(sq, occ) & ~bb.occ_white)
    for sq in _iter_set_bits(bb.WRi):
        # Right mobility: rook OR knight patterns
        right_att = _get_rook_attacks(sq, occ) | _KNIGHT_MOVES[sq]
        w_mob += MOBILITY_WEIGHTS["Right"] * _count_bits(right_att & ~bb.occ_white)
    for sq in _iter_set_bits(bb.WQ):
        w_mob += MOBILITY_WEIGHTS["Queen"] * _count_bits(
            (_get_bishop_attacks(sq, occ) | _get_rook_attacks(sq, occ)) & ~bb.occ_white
        )
    for sq in _iter_set_bits(bb.WK):
        w_mob += MOBILITY_WEIGHTS["King"] * _count_bits(_KING_MOVES[sq] & ~bb.occ_white)
    w_mob += MOBILITY_WEIGHTS["Pawn"] * _count_bits(_pawn_attacks_mask(True, bb.WP) & ~bb.occ_white)

    # Black mobility
    b_mob = 0
    for sq in _iter_set_bits(bb.BN):
        b_mob += MOBILITY_WEIGHTS["Knight"] * _count_bits(_KNIGHT_MOVES[sq] & ~bb.occ_black)
    for sq in _iter_set_bits(bb.BB):
        b_mob += MOBILITY_WEIGHTS["Bishop"] * _count_bits(_get_bishop_attacks(sq, occ) & ~bb.occ_black)
    for sq in _iter_set_bits(bb.BR):
        b_mob += MOBILITY_WEIGHTS["Rook"] * _count_bits(_get_rook_attacks(sq, occ) & ~bb.occ_black)
    for sq in _iter_set_bits(bb.BRi):
        right_att = _get_rook_attacks(sq, occ) | _KNIGHT_MOVES[sq]
        b_mob += MOBILITY_WEIGHTS["Right"] * _count_bits(right_att & ~bb.occ_black)
    for sq in _iter_set_bits(bb.BQ):
        b_mob += MOBILITY_WEIGHTS["Queen"] * _count_bits(
            (_get_bishop_attacks(sq, occ) | _get_rook_attacks(sq, occ)) & ~bb.occ_black
        )
    for sq in _iter_set_bits(bb.BK):
        b_mob += MOBILITY_WEIGHTS["King"] * _count_bits(_KING_MOVES[sq] & ~bb.occ_black)
    b_mob += MOBILITY_WEIGHTS["Pawn"] * _count_bits(_pawn_attacks_mask(False, bb.BP) & ~bb.occ_black)

    mob_diff = w_mob - b_mob
    mg += EVAL_MOBILITY_MG * mob_diff
    eg += EVAL_MOBILITY_EG * mob_diff

    # Phase blend
    phase = _phase_from_bb(bb)
    mg_w = phase / MAX_PHASE
    eg_w = (MAX_PHASE - phase) / MAX_PHASE
    blended = mg * mg_w + eg * eg_w
    return blended if bb.side_to_move == 0 else -blended


_IDX_TO_NAME = {0: "Pawn", 1: "Knight", 2: "Bishop", 3: "Rook", 4: "Queen", 5: "King", 6: "Right"}


def score_move_internal(
    bb: "BitboardState",
    m: "BBMove",
    var: Dict[str, Any],
    ply: int,
    tt_move_tuple: Optional[Tuple[int, int, int, int]],
) -> int:
    """
    Heuristic scoring for move ordering within bitboard search.
    """
    sx, sy, dx, dy = bbmove_to_tuple_xy(m)
    move_tuple = (sx, sy, dx, dy)

    if tt_move_tuple is not None and move_tuple == tt_move_tuple:
        return 1_000_000

    is_capture = m.captured_type >= 0
    if is_capture:
        # MVV-LVA + SEE
        victim_name = _IDX_TO_NAME.get(m.captured_type, "Pawn")
        aggressor_name = _IDX_TO_NAME.get(m.piece_type, "Pawn")
        vval = PIECE_VALUES_MG.get(victim_name, 0)
        aval = PIECE_VALUES_MG.get(aggressor_name, 0)
        base = 100_000 + (vval * 10) - aval
        try:
            occ = bb.occ_all
            tgt_sq = m.to_sq
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

    any_moves = generate_legal_moves(bb, captures_only=False)
    if not any_moves:
        if in_check:
            return -MATE_VALUE + ply
        contempt = float(var.get("contempt", 0.0))
        return contempt

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

    moves = generate_legal_moves(bb, captures_only=(not in_check))
    # Order captures by MVV-LVA via score_move_internal with deterministic tie-breaker
    scored_moves = []
    for m in moves:
        s = score_move_internal(bb, m, var, 99, None)
        sx, sy, dx, dy = bbmove_to_tuple_xy(m)
        scored_moves.append((s, (sx, sy, dx, dy), m))
    scored_moves.sort(key=lambda x: (-x[0], x[1]))

    for _, __, m in scored_moves:
        child = apply_bb_move(bb, m, var.get("zobrist"))
        # SEE gating for captures under time pressure
        if m.captured_type >= 0 and var.get("flags", {}).get("qsee", True) and not in_check:
            vname = _IDX_TO_NAME.get(m.captured_type, "Pawn")
            vval = PIECE_VALUES_MG.get(vname, 0)
            all_bbs_list = [
                bb.WP, bb.WN, bb.WB, bb.WR, bb.WQ, bb.WK, bb.WRi,
                bb.BP, bb.BN, bb.BB, bb.BR, bb.BQ, bb.BK, bb.BRi,
            ]
            see_gain = bb_see_njit(int(m.to_sq), bool(bb.side_to_move == 0), int(bb.occ_all), int(vval), all_bbs_list.copy())
            if see_gain < 0:
                continue

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
    var["_nodes"] = var.get("_nodes", 0) + 1

    # Repetition handling using position keys
    var.setdefault("_bb_rep_stack", [])
    if ply == 0:
        var["_bb_rep_stack"] = list(var.get("_bb_game_hist", []))
    var["_bb_rep_stack"].append(bb.zkey)
    try:
        if var.get("_bb_rep_stack", []).count(bb.zkey) >= 3:
            contempt = float(var.get("contempt", 0.0))
            sc = contempt if (bb.side_to_move == 0) else -contempt
            return sc, None
    except Exception:
        pass

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
    # Determine check status for current node
    stm_white = (bb.side_to_move == 0)
    kbb = bb.WK if stm_white else bb.BK
    ksq = _pop_lsb_njit(kbb) if kbb else -1
    in_check = False
    if ksq >= 0:
        checkers_bb, _, _ = calculate_legality_masks(bb, ksq, stm_white)
        in_check = (checkers_bb != 0)
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
            tt.store(bb.zkey, depth, out_score, TT_FLAG_EXACT, None)
            return out_score, None

    # Razoring and extended futility (only if not in check)
    static_eval: Optional[float] = None
    if not in_check and ply > 0 and (not is_pv_node):
        if depth in (1, 2):
            static_eval = evaluate_bb_state(bb)
            margin = RAZOR_MARGIN.get(depth, 0)
            if static_eval + margin <= alpha:
                q = quiescence_search_bb(bb, 3, alpha, beta, var, ply=ply)
                return q, None

    # Futility pruning lite
    if depth <= 1 and not in_check and ply > 0 and (not is_pv_node):
        static_eval = evaluate_bb_state(bb)
        if static_eval + 120 * depth <= alpha:
            return static_eval, None

    # Extended futility flag: skip quiets if static eval is far below alpha
    skip_quiet = False
    if not in_check and 3 <= depth <= 5 and ply > 0 and (not is_pv_node):
        if static_eval is None:
            static_eval = evaluate_bb_state(bb)
        fut_m = FUTILITY_MARGIN.get(depth, 0)
        if static_eval + fut_m <= alpha:
            skip_quiet = True

    # Move gen and ordering
    all_moves = generate_legal_moves(bb, captures_only=False)
    if not all_moves:
        # No legal moves: stalemate or checkmate -> loss per board_rules
        return (-MATE_VALUE + ply), None

    scored = [(score_move_internal(bb, m, var, ply, hash_move_tuple), m) for m in all_moves]
    scored.sort(key=lambda x: x[0], reverse=True)

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
        )
        s, _ = negamax_bb(null_child, depth - 1 - R, -beta, -beta + 1, var, ply + 1)
        score_nmp = -s
        if score_nmp >= beta:
            return beta, None

    for i, (_, m) in enumerate(scored):
        # Prepare prev move tuple for continuation history in child
        pre_tuple = bbmove_to_tuple_xy(m)

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

        # Extended futility: skip non-capture, non-promo, non-checks if flagged
        if skip_quiet and (m.captured_type < 0) and (m.promo == 0) and (not child_in_check):
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
                var.get("flags", {}).get("lmr", True)
                and (m.captured_type < 0)
                and (m.promo == 0)
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
            if m.captured_type >= 0:
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
    tt.store(bb.zkey, depth, score_to_store, tt_flag, best_tuple)
    return best_score, best_move

def clone_and_apply_move(board, piece, move_opt):
    """Clones the board, maps (piece, move_opt) to the clone, and applies it.

    Returns (new_board, mapped_piece, mapped_move) or (None, None, None) on failure.
    """
    try:
        new_board = board.clone()
    except Exception as e:
        print(f"Error cloning board: {e}")
        return None, None, None

    try:
        _, mapped_piece, mapped_move = copy_piece_move(new_board, piece, move_opt)
        if mapped_piece and mapped_move:
            mapped_piece.move(mapped_move)
            return new_board, mapped_piece, mapped_move
    except Exception as e:
        print(f"Error copying piece move: {e}")
    return None, None, None


def bb_from_board(board) -> BBPos:
    """
    Builds bitboards for all piece types and colors from the current board.

    This function scans the 5x5 board once, classifies pieces by color and
    piece type, sets the corresponding bit in a compact list of bitboards,
    and computes aggregate occupancy masks for white, black, and both.

    Args:
        board: A ChessMaker-compatible board exposing `__getitem__(Position)`
            to access squares and a `.piece` attribute on each square.

    Returns:
        A `BBPos` named tuple containing 14 piece bitboards and three
        occupancy masks (`occ_white`, `occ_black`, `occ_all`).
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
    Finds the least-valuable attacker (LVA) on a target square for STM.

    The search order is by piece value: pawns, knights, bishops, rooks, the
    custom Right piece (rook|knight), queens, and finally king. It uses pre-
    generated attack masks and sliding attack generators that respect `occ`.

    Args:
        sq: Target square index being contested.
        occ: Occupancy bitboard for all pieces.
        stm_white: True if the side-to-move is white; False if black.
        all_bbs: List of length 14 with piece bitboards in fixed order
            [WP, WN, WB, WR, WQ, WK, WRi, BP, BN, BB, BR, BQ, BK, BRi].

    Returns:
        A tuple (attacker_sq, attacker_value, attacker_index) where
        - attacker_sq is the square index of the chosen attacker or -1 if none,
        - attacker_value is the base value of the attacker (MG scale),
        - attacker_index is the index into `all_bbs` for that attacker.
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

    This iterative SEE simulates optimal capture sequences by alternating
    the side-to-move, repeatedly selecting the least-valuable attacker and
    accumulating a gain/loss stack, then backing up the best achievable
    exchange result.

    Args:
        sq: Target square index where the exchange occurs.
        stm_white: True if white is to move first in the exchange.
        occ: Initial occupancy bitboard for the position.
        victim_val: Base value of the initial victim on `sq`.
        all_bbs_in: Piece bitboards (length 14). This function mutates the
            list during evaluation; the caller should pass a copy.

    Returns:
        The net material gain for the side-to-move (positive is good for STM).

    Notes:
        - Uses a fixed-size gain stack and stops early on double-negative
          cutoffs.
        - Mutates `all_bbs_in`; callers must pass `copy()` if reusing.
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
    Convenience wrapper around `bb_see_njit` using a `BBPos` container.

    Args:
        bbpos: A `BBPos` namedtuple carrying all piece bitboards and occupancy.
        sq: Target square index for the exchange.
        occ: Occupancy bitboard to start the exchange.
        stm_white: True if white is to move; False if black.
        victim_val: Base value of the initial victim on `sq`.

    Returns:
        The SEE score (net gain for side-to-move) as an integer.
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
    Stub tablebase probe for 5x5 variant.
    Returns None for now. Can be replaced with a real probe that returns
    a dict like {'result': 'win'|'loss'|'draw', 'dtm': int}.
    """
    return None

# =====================================================================
# === Zobrist Hashing (UPDATED FOR 'Right' PIECE and Incremental Updates)
# =====================================================================


class Zobrist:
    """Implements Zobrist hashing for 5x5 chess.

    Provides a reproducible PRNG-backed table mapping (piece, color, square)
    to 64-bit integers and utilities to compute and update hashes.
    """

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

        self.zobrist_table = {}
        for p_idx in range(self.NUM_PIECE_TYPES):
            for c_idx in range(self.NUM_COLORS):
                for sq_idx in range(self.num_squares):
                    self.zobrist_table[(p_idx, c_idx, sq_idx)] = self._rand_64()

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
            return self.zobrist_table[(p_idx, c_idx, sq_idx)]
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
        try:
            return self.zobrist_table[(int(piece_type_idx), int(color_idx), int(sq_idx))]
        except Exception:
            return 0

    def toggle_by_indices(self, h: int, piece_type_idx: int, color_idx: int, sq_idx: int) -> int:
        """
        XOR toggles the hash by specifying indices directly.
        """
        return h ^ self.get_piece_hash_by_indices(piece_type_idx, color_idx, sq_idx)


# =====================================================================
# === Pawn-only Zobrist and Pawn Hash Table (Pawn TT)
# =====================================================================

# Deterministic 64-bit keys for pawn-only hashing
_PAWN_Z_RAND = random.Random(7737)
PAWN_Z_KEYS_WHITE: list[int] = [_PAWN_Z_RAND.getrandbits(64) for _ in range(25)]
PAWN_Z_KEYS_BLACK: list[int] = [_PAWN_Z_RAND.getrandbits(64) for _ in range(25)]


def pawn_zobrist_key(bb: "BitboardState") -> int:
    """
    Computes a 64-bit key using only white/black pawn bitboards.
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
    Returns (mg_diff, eg_diff) pawn-structure contribution, cached by pawn TT.
    Positive favors White, negative favors Black.
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


# Removed make/unmake; we now clone boards and use piece.move for applying moves


def compute_incremental_hash_after_move(
    zobrist: "Zobrist",
    current_hash: int,
    board,
    piece,
    move_opt,
    mapped_piece_after_move,
) -> int:
    """
    Computes the child position hash from the current node hash using
    incremental Zobrist updates. Assumes `board` is the pre-move board
    and `mapped_piece_after_move` is the post-move piece instance on the
    destination square (from a cloned/applied board).
    """
    try:
        h = zobrist.toggle_black_to_move(current_hash)
        sx, sy = piece.position.x, piece.position.y
        dx, dy = move_opt.position.x, move_opt.position.y

        # Remove moving piece from source
        h = zobrist.toggle_piece(h, piece, sx, sy)

        # Remove any captured pieces reported on the move (multi-capture safe)
        toggled_caps = set()
        try:
            cap_list = getattr(move_opt, "captures", None) or []
        except Exception:
            cap_list = []
        for pos in cap_list:
            try:
                cap_piece = board[Position(pos.x, pos.y)].piece
            except Exception:
                cap_piece = None
            if cap_piece is not None:
                h = zobrist.toggle_piece(h, cap_piece, pos.x, pos.y)
                toggled_caps.add((pos.x, pos.y))

        # If destination square held a piece and wasn't already toggled, remove it
        if (dx, dy) not in toggled_caps:
            try:
                occ = board[Square(dx, dy)].piece
            except Exception:
                occ = None
            if occ is not None and occ is not piece:
                h = zobrist.toggle_piece(h, occ, dx, dy)

        # Add the moved (possibly promoted) piece on destination
        h = zobrist.toggle_piece(h, mapped_piece_after_move, dx, dy)
        return h
    except Exception:
        # Fallback safety: on any error, keep original behavior to avoid mismatch
        try:
            moving_name = piece.player.name
        except Exception:
            moving_name = "white"
        return zobrist.compute_full_hash(board, moving_name)


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

    __slots__ = ["key", "depth", "score", "flag", "best_move_tuple"]

    def __init__(self, key, depth, score, flag, best_move_tuple):
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


TT_FLAG_EXACT = 0
TT_FLAG_LOWER = 1
TT_FLAG_UPPER = 2


class TranspositionTable:
    """
    Fixed-size transposition table using Zobrist hashing and replacement.

    Uses direct indexing by masking the Zobrist key; shallow replacement on
    equal/greater depth entries. Tracks probe/hit statistics for diagnostics.
    """

    def __init__(self, entry_count: int = 1048576):  # 1048576):
        """
        Initializes an empty transposition table.

        Args:
            entry_count: Number of table buckets. Must be a power of two for
                efficient masking; defaults to 262,144.
        """
        self.size = entry_count
        self.index_mask = self.size - 1
        self.table: List[Optional[TTEntry]] = [None] * self.size
        self.hits = 0
        self.probes = 0

    def clear(self) -> None:
        """
        Clears all stored entries and resets hit/probe counters.

        Returns:
            None.
        """
        self.table = [None] * self.size
        self.hits = 0
        self.probes = 0

    def get_index(self, zobrist_key: int) -> int:
        """
        Computes the table index from a Zobrist key via masking.

        Args:
            zobrist_key: 64-bit Zobrist hash of the position.

        Returns:
            Integer index into the table in [0, size-1].
        """
        return zobrist_key & self.index_mask

    def store(
        self,
        zobrist_key: int,
        depth: int,
        score: float,
        flag: int,
        best_move_tuple: Optional[Tuple[int, int, int, int]],
    ) -> None:
        """
        Stores or replaces a table entry for the given key.

        Prefers replacing entries with lower depth; on collision, keeps the
        entry with the greater or equal depth.

        Args:
            zobrist_key: Zobrist hash key for the position.
            depth: Search depth in plies.
            score: Score to store (may be mate-distance encoded by caller).
            flag: Bound type (EXACT/LOWER/UPPER).
            best_move_tuple: Optional best move (sx,sy,dx,dy) to aid PV.

        Returns:
            None.
        """
        index = self.get_index(zobrist_key)
        existing = self.table[index]

        if existing is None or depth >= existing.depth:
            self.table[index] = TTEntry(zobrist_key, depth, score, flag, best_move_tuple)

    def probe(self, zobrist_key: int) -> Optional[TTEntry]:
        """
        Probes the table for an entry matching the exact Zobrist key.

        Args:
            zobrist_key: 64-bit Zobrist hash of the position.

        Returns:
            The matching `TTEntry` if present; otherwise None.

        Side effects:
            Increments `probes` on every call and `hits` on exact match.
        """
        self.probes += 1
        index = self.get_index(zobrist_key)
        entry = self.table[index]

        if entry is not None and entry.key == zobrist_key:
            self.hits += 1
            return entry

        return None


# =====================================================================
# === Helper Functions (UPDATED FOR CHESSMAKER COMPATIBILITY)
# =====================================================================


def get_last_search_info() -> Dict[str, Any]:
    """
    Returns a snapshot of diagnostics from the most recent root search.

    Returns:
        A shallow copy of the internal `_LAST_SEARCH_INFO` dictionary,
        including metrics like depth, nodes, qnodes, tthits, PV, and more.
    """
    return dict(_LAST_SEARCH_INFO)


def opponent_name(board, name: str) -> str:
    """
    Returns the opposing color name for convenience.

    Args:
        board: Unused; present for call-site compatibility.
        name: "white" or "black".

    Returns:
        "black" if name == "white" else "white".
    """
    return "black" if name == "white" else "white"


def move_to_str(piece, move_opt) -> str:
    """
    Formats a move as a compact string for debug output.

    Args:
        piece: The moving piece with `.name` and `.position`.
        move_opt: A move option with destination `.position`.

    Returns:
        A string like "knight(1,2)->(2,4)"; falls back to `str(move_opt)`
        if the expected attributes are missing.
    """
    try:
        sx, sy = piece.position.x, piece.position.y
        dx, dy = move_opt.position.x, move_opt.position.y
        return f"{piece.name.lower()}({sx},{sy})->({dx},{dy})"
    except Exception as e:
        print(f"Error in move_to_str: {e}")
        return str(move_opt)


def _interpolate_eval(
    mg_score: float, eg_score: float, phase_factor_mg: float, phase_factor_eg: float
) -> float:
    """
    Blends middle-game and endgame scores using phase weights.

    Args:
        mg_score: Middle-game score subtotal.
        eg_score: Endgame score subtotal.
        phase_factor_mg: Weight for MG phase in [0, 1].
        phase_factor_eg: Weight for EG phase in [0, 1].

    Returns:
        Weighted sum: mg_score*phase_factor_mg + eg_score*phase_factor_eg.
    """
    return (mg_score * phase_factor_mg) + (eg_score * phase_factor_eg)


def _has_xy(obj) -> bool:
    """
    Checks whether an object exposes a valid `.position` with `.x` and `.y`.

    Args:
        obj: Any object that may carry a `.position` attribute.

    Returns:
        True if `obj.position.x` and `obj.position.y` are both not None;
        False otherwise or on error.
    """
    try:
        pos = getattr(obj, "position", None)
        return (pos is not None) and (pos.x is not None) and (pos.y is not None)
    except Exception as e:
        print(f"Error in _has_xy: {e}")
        return False


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
    """
    Returns a bitboard mask for a given file on 5x5.
    """
    m = 0
    for r in range(5):
        m |= 1 << square_index(file_x, r)
    return m


def rank_mask(rank_y: int) -> int:
    """
    Returns a bitboard mask for a given rank on 5x5.
    """
    m = 0
    for fx in range(5):
        m |= 1 << square_index(fx, rank_y)
    return m


def ring1_mask(kx: int, ky: int) -> int:
    """
    Returns a ring-1 bitboard around (kx, ky) (no center).
    """
    m = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            tx, ty = kx + dx, ky + dy
            if 0 <= tx < 5 and 0 <= ty < 5:
                m |= 1 << square_index(tx, ty)
    return m


def count_attackers_to_zone(bb: "BitboardState", zone_mask: int, white_attacking: bool) -> dict:
    """
    Counts piece attackers to a mask from the given color perspective.
    Returns counts for Knight, Bishop, Rook, Right, Queen.
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
# === Evaluation Functions (OPTIMIZED)
# =====================================================================


def get_piece_base_value(piece, phase_name):
    """
    Returns the base (material) value for a piece in a given phase.

    Args:
        piece: Piece instance or None.
        phase_name: "mg" (middle game) or "eg" (endgame).

    Returns:
        Integer value for the piece in the requested phase; 0 if piece is None.
    """
    if not piece:
        return 0
    return PIECE_VALUES[phase_name].get(piece.name, 0)


def get_pst_score(piece, phase_name):
    """
    Looks up the piece-square table (PST) score for a piece and phase.

    Args:
        piece: Piece instance with `.player.name` and `.position`.
        phase_name: "mg" or "eg".

    Returns:
        Integer PST score at the piece's current square; 0 on error.
    """
    if not piece:
        return 0
    try:
        pst = PSTS[phase_name][piece.player.name][piece.name]
        return pst[piece.position.y][piece.position.x]
    except (KeyError, IndexError, AttributeError):
        return 0


def _get_mobility_score(board, player, opponent):
    """
    Computes a mobility differential: 4 * (my_moves - opp_moves).
    """
    # return 0
    try:
        my_moves = len(list_legal_moves_for(board, player))
        opp_moves = len(list_legal_moves_for(board, opponent))
        return 4 * (my_moves - opp_moves)
    except Exception as e:
        print(f"Error in mobility score: {e}")
        return 0


def calculate_game_phase(pieces):
    """
    Computes a simplified game phase value based on remaining material.

    Args:
        pieces: Iterable of pieces currently on the board.

    Returns:
        An integer in [1, MAX_PHASE] used to blend MG/EG evaluations.

    How it works:
        Sums per-piece phase weights (e.g., queens/rooks contribute more),
        and clamps the result to [1, MAX_PHASE].
    """
    total_phase = 0
    for piece in pieces:
        total_phase += PHASE_VALUES.get(piece.name, 0)
    return max(1, min(total_phase, MAX_PHASE))


def evaluate_position_static(board, player, pieces=None, game_phase=None, skip_mobility: bool = False):
    """
    Computes a static evaluation (material + PST + extras) for a player.
    This version is optimized to iterate over pieces ONCE.
    """
    if pieces is None:
        try:
            pieces = list(board.get_pieces())
        except Exception as e:
            print(f"Error getting pieces in eval: {e}")
            return 0
            
    if game_phase is None:
        game_phase = calculate_game_phase(pieces)

    try:
        my_name = player.name
        opponent = board.players[1] if my_name == "white" else board.players[0]
        opp_name = opponent.name
    except Exception:
        my_name = "white"
        opp_name = "black"
        opponent = None

    phase_factor_mg = game_phase / MAX_PHASE
    phase_factor_eg = (MAX_PHASE - game_phase) / MAX_PHASE

    mg_score = 0
    eg_score = 0
    
    # --- Single Pass ---
    board_grid = [[None for _ in range(5)] for _ in range(5)]
    files_my_pawns = [0] * 5
    files_opp_pawns = [0] * 5
    my_pawns = [] # Store (x, y)
    opp_pawns = [] # Store (x, y)
    
    my_bishops = 0
    my_rooks = 0
    my_rights = 0
    my_q = 0
    my_r = 0
    my_ri = 0
    opp_non_king = 0
    
    my_king_pos = None
    opp_king_pos = None
    my_queen_pos = None
    my_rooks_pos = []
    my_rights_pos = []

    for piece in pieces:
        # 1. Add Material and PST scores
        val_mg = get_piece_base_value(piece, "mg")
        val_eg = get_piece_base_value(piece, "eg")
        pst_mg = get_pst_score(piece, "mg")
        pst_eg = get_pst_score(piece, "eg")

        if piece.player.name == my_name:
            mg_score += val_mg + pst_mg
            eg_score += val_eg + pst_eg
        else:
            mg_score -= val_mg + pst_mg
            eg_score -= val_eg + pst_eg
            
        # 2. Collect data for 'extras'
        try:
            px, py = piece.position.x, piece.position.y
            board_grid[py][px] = piece
        except Exception:
            continue
            
        if piece.player.name == my_name:
            if piece.name == "Pawn":
                files_my_pawns[px] += 1
                my_pawns.append((px, py))
            elif piece.name == "Bishop":
                my_bishops += 1
            elif piece.name == "Rook":
                my_rooks += 1
                my_r += 1
                my_rooks_pos.append((px, py))
            elif piece.name == "Right":
                my_rights += 1
                my_ri += 1
                my_rights_pos.append((px, py))
            elif piece.name == "Queen":
                my_q += 1
                my_queen_pos = (px, py)
            elif piece.name == "King":
                my_king_pos = (px, py)
        else:
            if piece.name == "Pawn":
                files_opp_pawns[px] += 1
                opp_pawns.append((px, py))
            elif piece.name == "King":
                opp_king_pos = (px, py)
            else:
                opp_non_king += 1
    # --- End Single Pass ---

    extra_mg = 0
    extra_eg = 0

    if my_bishops >= 2:
        extra_mg += EVAL_BISHOP_PAIR
        extra_eg += EVAL_BISHOP_PAIR

    try:
        for cx, cy in CENTER_SQUARES:
            pc = board_grid[cy][cx]
            if pc and pc.player.name == my_name:
                extra_mg += EVAL_CENTER_CONTROL
                extra_eg += EVAL_CENTER_CONTROL
    except Exception:
        pass # Center control is minor

    # Passed Pawns (now iterates over small list)
    for x, y in my_pawns:
        dirs = -1 if my_name == "white" else 1
        ahead_ranks = range(y + dirs, 5, dirs) if dirs == 1 else range(y + dirs, -1, dirs)
        is_passed = True
        for fx in range(max(0, x - 1), min(4, x + 1) + 1):
            for ry in ahead_ranks:
                try:
                    pp = board_grid[ry][fx]
                except Exception:
                    pp = None
                if pp and pp.player.name == opp_name and pp.name == "Pawn":
                    is_passed = False
                    break
            if not is_passed:
                break
        if is_passed:
            rank_from_home = (4 - y) if my_name == "white" else y
            rank_from_home = max(0, min(4, rank_from_home))
            extra_mg += EVAL_PASSED_PAWN_MG[rank_from_home]
            extra_eg += EVAL_PASSED_PAWN_EG[rank_from_home]

    # Doubled/Isolated Pawns
    for f in range(5):
        if files_my_pawns[f] >= 2:
            extra_mg += EVAL_DOUBLED_PAWN
            extra_eg += EVAL_DOUBLED_PAWN
        if files_my_pawns[f] > 0:
            has_left = files_my_pawns[f - 1] > 0 if f - 1 >= 0 else False
            has_right = files_my_pawns[f + 1] > 0 if f + 1 <= 4 else False
            if not has_left and not has_right:
                extra_mg += EVAL_ISOLATED_PAWN
                extra_eg += EVAL_ISOLATED_PAWN

    # Open Files
    def file_has_pawn_for(files, file_idx):
        try:
            return files[file_idx] > 0
        except Exception:
            return False

    if my_rooks > 0:
        for f in range(5):
            if not file_has_pawn_for(files_my_pawns, f) and not file_has_pawn_for(
                files_opp_pawns, f
            ):
                extra_mg += EVAL_ROOK_OPEN_FILE
                extra_eg += EVAL_ROOK_OPEN_FILE
            elif not file_has_pawn_for(files_my_pawns, f):
                extra_mg += EVAL_ROOK_SEMIOPEN
                extra_eg += EVAL_ROOK_SEMIOPEN
    if my_rights > 0:
        for f in range(5):
            if not file_has_pawn_for(files_my_pawns, f) and not file_has_pawn_for(
                files_opp_pawns, f
            ):
                extra_mg += EVAL_RIGHT_OPEN_FILE
                extra_eg += EVAL_RIGHT_OPEN_FILE
            elif not file_has_pawn_for(files_my_pawns, f):
                extra_mg += EVAL_RIGHT_SEMIOPEN
                extra_eg += EVAL_RIGHT_SEMIOPEN

    # Dynamic King Safety
    if my_king_pos is not None:
        kx, ky = my_king_pos
        danger_zone_bonus = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                sq_x, sq_y = kx + dx, ky + dy
                if 0 <= sq_x < 5 and 0 <= sq_y < 5:
                    try:
                        pc = board_grid[sq_y][sq_x]
                        if pc:
                            if pc.player.name == my_name:
                                danger_zone_bonus += 5 if pc.name == "Pawn" else 2
                            else:
                                danger_zone_bonus -= {
                                    "Pawn": 5, "Knight": 10, "Bishop": 10,
                                    "Rook": 15, "Right": 15, "Queen": 25,
                                }.get(pc.name, 10)
                    except Exception:
                        pass
        extra_mg += danger_zone_bonus
        extra_eg += danger_zone_bonus

    # Endgame drives (KQK / KRK)
    try:
        if opp_king_pos:
            is_kqk = my_q == 1 and (my_r + my_ri) == 0 and opp_non_king == 0
            is_krk = (my_r + my_ri) >= 1 and my_q == 0 and opp_non_king == 0

            if is_kqk or is_krk:
                ex, ey = opp_king_pos
                if my_king_pos is not None:
                    kprox = 4 - _chebyshev(my_king_pos, opp_king_pos)
                    extra_eg += EVAL_EG_OWN_KING_PROXIMITY * kprox
                
                edge_w = EVAL_EG_OPP_KING_TO_EDGE_R if is_krk else EVAL_EG_OPP_KING_TO_EDGE_Q
                corner_w = EVAL_EG_OPP_KING_TO_CORNER_R if is_krk else EVAL_EG_OPP_KING_TO_CORNER_Q
                extra_eg += edge_w * (2 - _edge_distance(ex, ey))
                extra_eg += corner_w * (2 - _corner_distance(ex, ey))

                if is_krk:
                    for rx, ry in my_rooks_pos + my_rights_pos:
                        if rx == ex or ry == ey:
                            extra_eg += EVAL_EG_ROOK_CUTOFF
                
                if is_kqk and my_queen_pos is not None and my_king_pos is not None:
                    qd = _chebyshev(my_queen_pos, opp_king_pos)
                    kd = _chebyshev(my_king_pos, opp_king_pos)
                    if qd <= 1 and kd >= 2:
                        extra_eg += EVAL_EG_QUEEN_ADJ_PENALTY
    except Exception:
        pass # Endgame extras are bonuses

    # --- Combine Scores ---
    mg_score += extra_mg
    eg_score += extra_eg

    # NOTE: Mobility is skipped here as it's the main bottleneck.
    # If you ever re-enable it, it should go here, guarded by skip_mobility.

    return (mg_score * phase_factor_mg) + (eg_score * phase_factor_eg)


def _evaluate_extras(board, player, pieces):
    """
    Computes classical evaluation extras (MG/EG) on a compact 5x5 board.

    OPTIMIZED: Builds a 2D grid for O(1) piece lookups.
    """
    try:
        my_name = player.name
    except Exception as e:
        print(f"Error getting player name: {e}")
        my_name = "white"

    # Build 5x5 grid for O(1) piece lookups
    board_grid = [[None for _ in range(5)] for _ in range(5)]

    files_my_pawns = [0] * 5
    files_opp_pawns = [0] * 5
    my_bishops = 0
    my_rooks = 0
    my_rights = 0
    opp_material_mg = 0
    my_king_pos = None
    opp_name = "black" if my_name == "white" else "white"
    # Precompute positions for endgame extras in the same pass
    opp_king_pos = None
    my_queen_pos = None
    my_rooks_pos = []
    my_rights_pos = []
    my_q = 0
    my_r = 0
    my_ri = 0
    opp_non_king = 0

    for piece in pieces:
        try:
            px, py = piece.position.x, piece.position.y
            board_grid[py][px] = piece
        except Exception as e:
            print(f"Error getting position: {e}")
            continue
        if piece.player.name == my_name:
            if piece.name == "Pawn":
                files_my_pawns[px] += 1
            elif piece.name == "Bishop":
                my_bishops += 1
            elif piece.name == "Rook":
                my_rooks += 1
                my_r += 1
                my_rooks_pos.append((px, py))
            elif piece.name == "Right":
                my_rights += 1
                my_ri += 1
                my_rights_pos.append((px, py))
            elif piece.name == "Queen":
                my_q += 1
                my_queen_pos = (px, py)
            elif piece.name == "King":
                my_king_pos = (px, py)
        else:
            if piece.name == "Pawn":
                files_opp_pawns[px] += 1
            elif piece.name == "King":
                opp_king_pos = (px, py)
            else:
                opp_non_king += 1
            opp_material_mg += get_piece_base_value(piece, "mg")

    mg = 0
    eg = 0

    if my_bishops >= 2:
        mg += EVAL_BISHOP_PAIR
        eg += EVAL_BISHOP_PAIR

    try:
        for cx, cy in CENTER_SQUARES:
            pc = board_grid[cy][cx]
            if pc and pc.player.name == my_name:
                mg += EVAL_CENTER_CONTROL
                eg += EVAL_CENTER_CONTROL
    except Exception as e:
        print(f"Error in center control: {e}")

    for piece in pieces:
        if getattr(piece, "player", None) and piece.player.name == my_name and piece.name == "Pawn":
            try:
                x, y = piece.position.x, piece.position.y
            except Exception as e:
                print(f"Error getting position: {e}")
                continue
            dirs = -1 if my_name == "white" else 1
            ahead_ranks = range(y + dirs, 5, dirs) if dirs == 1 else range(y + dirs, -1, dirs)
            is_passed = True
            for fx in range(max(0, x - 1), min(4, x + 1) + 1):
                for ry in ahead_ranks:
                    try:
                        pp = board_grid[ry][fx]
                    except Exception:
                        pp = None
                    if pp and pp.player.name == opp_name and pp.name == "Pawn":
                        is_passed = False
                        break
                if not is_passed:
                    break
            if is_passed:
                rank_from_home = (4 - y) if my_name == "white" else y
                rank_from_home = max(0, min(4, rank_from_home))
                mg += EVAL_PASSED_PAWN_MG[rank_from_home]
                eg += EVAL_PASSED_PAWN_EG[rank_from_home]

    for f in range(5):
        if files_my_pawns[f] >= 2:
            mg += EVAL_DOUBLED_PAWN
            eg += EVAL_DOUBLED_PAWN
        if files_my_pawns[f] > 0:
            has_left = files_my_pawns[f - 1] > 0 if f - 1 >= 0 else False
            has_right = files_my_pawns[f + 1] > 0 if f + 1 <= 4 else False
            if not has_left and not has_right:
                mg += EVAL_ISOLATED_PAWN
                eg += EVAL_ISOLATED_PAWN

    def file_has_pawn_for(files, file_idx):
        try:
            return files[file_idx] > 0
        except Exception as e:
            print(f"Error in file_has_pawn_for: {e}")
            return False

    if my_rooks > 0:
        for f in range(5):
            if not file_has_pawn_for(files_my_pawns, f) and not file_has_pawn_for(
                files_opp_pawns, f
            ):
                mg += EVAL_ROOK_OPEN_FILE
                eg += EVAL_ROOK_OPEN_FILE
            elif not file_has_pawn_for(files_my_pawns, f):
                mg += EVAL_ROOK_SEMIOPEN
                eg += EVAL_ROOK_SEMIOPEN
    if my_rights > 0:
        for f in range(5):
            if not file_has_pawn_for(files_my_pawns, f) and not file_has_pawn_for(
                files_opp_pawns, f
            ):
                mg += EVAL_RIGHT_OPEN_FILE
                eg += EVAL_RIGHT_OPEN_FILE
            elif not file_has_pawn_for(files_my_pawns, f):
                mg += EVAL_RIGHT_SEMIOPEN
                eg += EVAL_RIGHT_SEMIOPEN

    # --- Dynamic King Safety ---
    if my_king_pos is not None:
        kx, ky = my_king_pos
        danger_zone_bonus = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                sq_x, sq_y = kx + dx, ky + dy
                if 0 <= sq_x < 5 and 0 <= sq_y < 5:
                    try:
                        pc = board_grid[sq_y][sq_x]
                        if pc:
                            if pc.player.name == my_name:
                                danger_zone_bonus += 5 if pc.name == "Pawn" else 2
                            else:
                                danger_zone_bonus -= {
                                    "Pawn": 5,
                                    "Knight": 10,
                                    "Bishop": 10,
                                    "Rook": 15,
                                    "Right": 15,
                                    "Queen": 25,
                                }.get(pc.name, 10)
                    except Exception as e:
                        print(f"Error in dynamic king safety: {e}")
        mg += danger_zone_bonus
        eg += danger_zone_bonus

    # --- Endgame drives (KQK / KRK) ---
    try:
        if opp_king_pos:
            is_kqk = my_q == 1 and (my_r + my_ri) == 0 and opp_non_king == 0
            is_krk = (my_r + my_ri) >= 1 and my_q == 0 and opp_non_king == 0

            if is_kqk or is_krk:
                ex, ey = opp_king_pos

                # Bring our king closer in simple mates
                if my_king_pos is not None:
                    kprox = 4 - _chebyshev(my_king_pos, opp_king_pos)
                    eg += EVAL_EG_OWN_KING_PROXIMITY * kprox

                # Drive enemy king to edge/corner
                edge_w = EVAL_EG_OPP_KING_TO_EDGE_R if is_krk else EVAL_EG_OPP_KING_TO_EDGE_Q
                corner_w = EVAL_EG_OPP_KING_TO_CORNER_R if is_krk else EVAL_EG_OPP_KING_TO_CORNER_Q
                eg += edge_w * (2 - _edge_distance(ex, ey))
                eg += corner_w * (2 - _corner_distance(ex, ey))

                # Rook/Right cut-off bonus (same rank/file as enemy king)
                if is_krk:
                    for rx, ry in my_rooks_pos + my_rights_pos:
                        if rx == ex or ry == ey:
                            eg += EVAL_EG_ROOK_CUTOFF

                # Avoid stalemate-y queen adjacency when our king is far
                if is_kqk and my_queen_pos is not None and my_king_pos is not None:
                    qd = _chebyshev(my_queen_pos, opp_king_pos)
                    kd = _chebyshev(my_king_pos, opp_king_pos)
                    if qd <= 1 and kd >= 2:
                        eg += EVAL_EG_QUEEN_ADJ_PENALTY
    except Exception as e:
        print(f"Endgame extras error: {e}")

    return mg, eg


def evaluate_position(board, player):
    """
    Computes evaluation as static score (material + PST + extras + mobility).
    """
    try:
        pieces = list(board.get_pieces())
    except Exception as e:
        print(f"Error in pieces: {e}")
        pieces = []

    game_phase = calculate_game_phase(pieces)

    static_score = evaluate_position_static(board, player, pieces, game_phase)

    return static_score


def evaluate_position_breakdown(board, player):
    """
    Returns a detailed evaluation breakdown for diagnostics and UI.

    Args:
        board: The game board.
        player: Perspective of the evaluation.

    Returns:
        Dict with phase factors, material, PST, mobility, MG/EG sums, and
        the final blended score.
    """
    pieces = list(board.get_pieces())
    game_phase = calculate_game_phase(pieces)
    phase_factor_mg = game_phase / MAX_PHASE
    phase_factor_eg = (MAX_PHASE - game_phase) / MAX_PHASE

    material_mg = 0
    material_eg = 0
    pst_mg = 0
    pst_eg = 0

    for piece in pieces:
        val_mg = get_piece_base_value(piece, "mg")
        val_eg = get_piece_base_value(piece, "eg")
        sq_mg = get_pst_score(piece, "mg")
        sq_eg = get_pst_score(piece, "eg")
        if piece.player.name == player.name:
            material_mg += val_mg
            material_eg += val_eg
            pst_mg += sq_mg
            pst_eg += sq_eg
        else:
            material_mg -= val_mg
            material_eg -= val_eg
            pst_mg -= sq_mg
            pst_eg -= sq_eg

    try:
        my_moves = len(list_legal_moves_for(board, player))
        opponent = board.players[1] if player.name == "white" else board.players[0]
        opp_moves = len(list_legal_moves_for(board, opponent))
        mobility_score = 3 * (my_moves - opp_moves)
    except Exception as e:
        print(f"Error in mobility score: {e}")
        mobility_score = 0

    mg_score = material_mg + pst_mg + mobility_score
    eg_score = material_eg + pst_eg + mobility_score
    final_score = _interpolate_eval(mg_score, eg_score, phase_factor_mg, phase_factor_eg)

    return {
        "game_phase": game_phase,
        "phase_factor_mg": round(phase_factor_mg, 2),
        "phase_factor_eg": round(phase_factor_eg, 2),
        "material_mg": material_mg,
        "material_eg": material_eg,
        "pst_mg": pst_mg,
        "pst_eg": pst_eg,
        "mobility": mobility_score,
        "mg_score": mg_score,
        "eg_score": eg_score,
        "final_score": round(final_score, 2),
    }


# =====================================================================
# === Move Ordering (OPTIMIZED)
# =====================================================================


def score_move(
    board, piece, move_opt, var, ply, hash_move_tuple, prev_move_tuple, bbpos, all_bbs_list
):
    """
    Heuristically scores a move for ordering within the search.

    Args:
        board: The board to query for victims/captures.
        piece: The moving piece.
        move_opt: The move option being scored.
        var: Search variables dictionary (history, killers, etc.).
        ply: Current ply index in the search tree.
        hash_move_tuple: Optional transposition move tuple to prioritize.
        prev_move_tuple: Previous move tuple used for countermove heuristic.
        bbpos: Optional precomputed `BBPos` for SEE and occupancy.
        all_bbs_list: Optional list of bitboards aligned with `bbpos`.

    Returns:
        An integer score; higher scores are searched earlier.

    How it works:
        - Prioritizes TT move, winning captures (SEE), promotions, killers,
          countermoves, and history heuristics in that order of magnitude.
    """
    if piece is None or move_opt is None or not _has_xy(piece) or not _has_xy(move_opt):
        return -1_000_000

    move_tuple = (piece.position.x, piece.position.y, move_opt.position.x, move_opt.position.y)

    if move_tuple == hash_move_tuple:
        return 1000000

    is_capture = getattr(move_opt, "captures", None)

    if is_capture:
        victim = None
        victim_pos = None
        for pos in move_opt.captures:
            try:
                p = board[Square(pos.x, pos.y)].piece
            except Exception:
                p = None
            if p:
                victim = p
                victim_pos = pos
                break
        if not victim:
            try:
                victim = board[Square(move_opt.position.x, move_opt.position.y)].piece
            except Exception:
                victim = None
            victim_pos = move_opt.position

        base = 100000
        if victim:
            victim_val = get_piece_base_value(victim, "mg")
            aggressor_val = get_piece_base_value(piece, "mg")
            base += (victim_val * 10) - aggressor_val

            if bbpos is not None and all_bbs_list is not None:
                try:
                    occ = bbpos.occ_all
                    tgt_sq = square_index(victim_pos.x, victim_pos.y)
                    stm_white = getattr(piece.player, "name", "white") == "white"

                    see_gain = bb_see_njit(
                        int(tgt_sq), bool(stm_white), int(occ), int(victim_val), all_bbs_list.copy()
                    )

                    if see_gain >= 0:
                        base += 500 + see_gain
                    else:
                        base = 1000 + see_gain
                except Exception as e:
                    print(f"Error in SEE: {e}")
                    pass
            return base
        else:
            base += 500
            return base

    if ply < len(var["killers"]) and move_tuple in var["killers"][ply]:
        return 90000

    if prev_move_tuple is not None:
        cm = var.get("countermoves", {}).get(prev_move_tuple)
        if cm is not None and cm == move_tuple:
            return 85000

    return var["history"].get(move_tuple, 0)


def _classify_move_stage(board, piece, move_opt, var, ply, hash_move_tuple, bbpos, all_bbs_list):
    """
    Classifies a move into ordering stages and returns a sort key.

    Args:
        board: Current board.
        piece: Moving piece.
        move_opt: Candidate move.
        var: Search variables including killers/countermoves/history.
        ply: Current ply.
        hash_move_tuple: Move from TT to force to the front when present.
        bbpos: Optional bitboards for SEE.
        all_bbs_list: Optional list of bitboards aligned with `bbpos`.

    Returns:
        Tuple (stage_index, score, see_gain_or_None). Smaller stage_index
        sorts earlier; score is a secondary key.

    Notes:
        In addition to captures/promotions/killers/countermoves/history,
        quiet checking moves are prioritized slightly ahead of killer moves
        to improve tactical forcing lines (mates and nets) exploration.
    """
    if piece is None or move_opt is None or not _has_xy(piece) or not _has_xy(move_opt):
        return 99, -1_000_000, None
    try:
        move_tuple = (
            piece.position.x,
            piece.position.y,
            move_opt.position.x,
            move_opt.position.y,
        )
    except Exception as e:
        print(f"Error getting move tuple: {e}")
        return 99, -1_000_000, None

    if hash_move_tuple is not None and move_tuple == hash_move_tuple:
        return 1, 1_000_000, None

    is_promo = False
    try:
        is_promo = bool(getattr(move_opt, "extra", {}).get("promote"))
    except Exception as e:
        print(f"Error getting promote: {e}")
        is_promo = False

    is_capture = getattr(move_opt, "captures", None)
    see_gain = None
    if is_capture:
        victim = None
        victim_pos = None
        for pos in move_opt.captures:
            try:
                p = board[Square(pos.x, pos.y)].piece
            except Exception:
                p = None
            if p:
                victim = p
                victim_pos = pos
                break
        if not victim:
            try:
                victim = board[Square(move_opt.position.x, move_opt.position.y)].piece
            except Exception:
                victim = None
            victim_pos = move_opt.position
        if (
            victim
            and bbpos is not None
            and all_bbs_list is not None
            and not var.get("_time_pressure", False)
        ):
            try:
                occ = bbpos.occ_all
                tgt_sq = square_index(victim_pos.x, victim_pos.y)
                stm_white = getattr(piece.player, "name", "white") == "white"
                see_gain = bb_see_njit(
                    int(tgt_sq),
                    bool(stm_white),
                    int(occ),
                    int(get_piece_base_value(victim, "mg")),
                    all_bbs_list.copy(),
                )
            except Exception as e:
                print(f"Error in SEE: {e}")
                see_gain = None
        if see_gain is not None and see_gain >= 0:
            return 2, 100_000 + see_gain, see_gain
        else:
            return 7, -10_000 + (see_gain or -1), see_gain

    if is_promo:
        return 3, 95_000, None

    # Prefer quiet checks slightly ahead of killers
    # try:
    #     moving_name = piece.player.name
    #     next_player = board.players[1] if moving_name == "white" else board.players[0]
    #     nb, mp, mm = clone_and_apply_move(board, piece, move_opt)
    #     if nb is not None and is_in_check(nb, next_player):
    #         return 4, 90_500, None
    # except Exception:
    #     pass

    try:
        if ply < len(var.get("killers", [])) and move_tuple in var["killers"][ply]:
            return 4, 90_000, None
    except Exception as e:
        print(f"Error in killer check: {e}")
        pass

    try:
        prev = var.get("_prev_move_tuple")
        if prev is not None:
            cm = var.get("countermoves", {}).get(prev)
            if cm is not None and cm == move_tuple:
                return 5, 85_000, None
    except Exception as e:
        print(f"Error in countermove check: {e}")
        pass

    return 6, int(var.get("history", {}).get(move_tuple, 0)), None


def get_ordered_moves(
    board, player, var, ply, hash_move_tuple, prev_move_tuple=None, captures_only=False, board_hash=None
):
    """
    Generates and orders legal moves using several heuristics.

    Args:
        board: Current board state.
        player: The player to move.
        var: Search variables dict (history, killers, etc.).
        ply: Current ply in the search.
        hash_move_tuple: Optional TT move to prioritize.
        prev_move_tuple: Optional previous move for countermove heuristic.
        captures_only: If True, return only capturing moves.

    Returns:
        A list of (piece, move_opt) pairs ordered from best to worst.
    """
    try:
        if board_hash is not None:
            all_legal_moves = get_legal_moves_cached(board, player, var, board_hash)
        else:
            zob = var.get("zobrist")
            bh = zob.compute_full_hash(board, player.name) if zob else None
            if bh is not None and "_legal_moves_cache" in var or zob:
                all_legal_moves = get_legal_moves_cached(board, player, var, bh)
            else:
                all_legal_moves = list_legal_moves_for(board, player)
    except Exception as e:
        if DEBUG:
            print(f"Error listing legal moves in normal version: {e}")
        return []

    legal_moves = []
    for p, m in all_legal_moves:
        if p is None or m is None:
            continue
        if not _has_xy(p) or not _has_xy(m):
            continue

        if captures_only:
            if getattr(m, "captures", None):
                legal_moves.append((p, m))
        else:
            legal_moves.append((p, m))

    if not legal_moves:
        return []

    bbpos = None
    all_bbs_list = None
    try:
        bbpos = bb_from_board(board)
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
    except Exception as e:
        print(f"Error building bitboards: {e}")
        pass

    scored_moves = []
    for p, m in legal_moves:
        sc = score_move(
            board, p, m, var, ply, hash_move_tuple, prev_move_tuple, bbpos, all_bbs_list
        )
        mv_key = (p.position.x, p.position.y, m.position.x, m.position.y)
        scored_moves.append((sc, mv_key, (p, m)))
    scored_moves.sort(key=lambda x: (-x[0], x[1]))
    max_moves = 12 if var.get("_time_pressure") else 20
    trimmed = scored_moves[:max_moves]
    return [move for _, __, move in trimmed]


def get_ordered_moves_optimized(
    board,
    player,
    var,
    ply,
    hash_move_tuple,
    prev_move_tuple=None,
    captures_only=False,
    bbpos=None,
    all_bbs_list=None,
    board_hash=None,
):
    """
    Like `get_ordered_moves` but accepts precomputed bitboards for speed.

    Args:
        board: Current board.
        player: The player to move.
        var: Search variables dict.
        ply: Current ply.
        hash_move_tuple: Optional TT move to prioritize.
        prev_move_tuple: Optional previous move for countermove heuristic.
        captures_only: Limit to captures when True.
        bbpos: Optional `BBPos` computed once per node.
        all_bbs_list: Optional list of bitboards aligned with `bbpos`.

    Returns:
        Ordered list of (piece, move_opt) pairs.
    """
    try:
        if board_hash is not None:
            all_legal_moves = get_legal_moves_cached(board, player, var, board_hash)
        else:
            zob = var.get("zobrist")
            bh = zob.compute_full_hash(board, player.name) if zob else None
            if bh is not None:
                all_legal_moves = get_legal_moves_cached(board, player, var, bh)
            else:
                all_legal_moves = list_legal_moves_for(board, player)
    except Exception as e:
        if DEBUG:
            print(f"Error listing legal moves in optimized version: {e}")
        return []

    legal_moves = []
    for p, m in all_legal_moves:
        if p is None or m is None:
            continue
        if not _has_xy(p) or not _has_xy(m):
            continue
        if captures_only:
            if getattr(m, "captures", None):
                legal_moves.append((p, m))
        else:
            legal_moves.append((p, m))

    if not legal_moves:
        return []

    if bbpos is None or all_bbs_list is None:
        try:
            bbpos = bb_from_board(board)
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
        except Exception as e:
            print(f"Error building bitboards: {e}")
            pass

    staged = []
    for p, m in legal_moves:
        stg, scr, seev = _classify_move_stage(
            board, p, m, var, ply, hash_move_tuple, bbpos, all_bbs_list
        )
        mv_key = (p.position.x, p.position.y, m.position.x, m.position.y)
        staged.append(((stg, -scr), (p, m), mv_key))
    staged.sort(key=lambda x: (x[0], x[2]))
    max_moves = 12 if var.get("_time_pressure") else 20
    return [mv for _, mv, __ in staged[:max_moves]]


# =====================================================================
# === Quiescence Search (MODIFIED)
# =====================================================================


def quiescence_search(
    board,
    player,
    depth,
    alpha,
    beta,
    pieces,
    game_phase,
    current_hash: int,
    var: Dict[str, Any],
    last_cap_sq: Optional[int] = None,
    *,
    tree_parent_id: Optional[int] = None,
    move_str: Optional[str] = None,
    node_type: str = "q",
):
    """
    Searches only "noisy" continuations (captures/check evasions) to stabilize eval.

    Args:
        board: Current board.
        player: Side to move.
        depth: Remaining quiescence depth (plies) for recursion control.
        alpha: Current alpha bound.
        beta: Current beta bound.
        pieces: Precomputed list of pieces for the current node.
        game_phase: Precomputed game phase for blending MG/EG.
        current_hash: Zobrist hash at the current node.
        var: Search state dictionary (zobrist, flags, counters, etc.).
        last_cap_sq: If provided, prefer recaptures on this square.

    Returns:
        The best stand-pat or capture/escape sequence score within [alpha, beta].

    How it works:
        - Stand-pat with static evaluation and alpha-beta pruning.
        - If in check, consider all moves; otherwise only captures (and
          optionally recaptures first).
        - Use incremental make/unmake with Zobrist hashing and optional SEE
          gating to skip losing captures.
    """
    var["_qnodes"] = var.get("_qnodes", 0) + 1

    # try:
    #     result = get_result(board)
    #     if result is not None:
    #         res_lower = result.lower()
    #         if "checkmate" in res_lower:
    #             if player.name in res_lower and "loses" in res_lower:
    #                 return -MATE_VALUE  # This is a mate against us
    #             else:
    #                 return MATE_VALUE  # We delivered this mate
    #         elif "draw" in res_lower or "stalemate" in res_lower:
    #             return 0.0  # Return 0 for a draw
    # except Exception as e:
    #     print(f"Error checking get_result in q-search: {e}")

    # Enforce time budgets (identical to negamax)
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

    # Prepare local bitboards once; used for in-check tests
    try:
        bbpos_local = bb_from_board(board)
    except Exception:
        bbpos_local = None

    bbpos_for_see = bb_from_board(board)
    all_bbs_list_for_see = [
            bbpos_for_see.WP, bbpos_for_see.WN, bbpos_for_see.WB, bbpos_for_see.WR,
            bbpos_for_see.WQ, bbpos_for_see.WK, bbpos_for_see.WRi,
            bbpos_for_see.BP, bbpos_for_see.BN, bbpos_for_see.BB, bbpos_for_see.BR,
            bbpos_for_see.BQ, bbpos_for_see.BK, bbpos_for_see.BRi,
        ]

    stand_pat = evaluate_position_static(board, player, pieces, game_phase, skip_mobility=True)
    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    if depth == 0:
        return alpha

    in_check_now = is_in_check(board, player, bbpos=bbpos_local if 'bbpos_local' in locals() else None)

    # Delta pruning: if even a queen gain cannot lift alpha, cut
    DELTA_MARGIN = 900
    if (not in_check_now) and (stand_pat + DELTA_MARGIN < alpha):
        return alpha
    if in_check_now:
        moves_to_search = get_ordered_moves(
            board, player, var, 99, None, None, captures_only=False, board_hash=current_hash
        )
    else:
        moves_to_search = get_ordered_moves(
            board, player, var, 99, None, None, captures_only=True, board_hash=current_hash
        )

    if last_cap_sq is not None:
        try:

            def _is_recap(mv):
                m = mv[1]
                for pos in getattr(m, "captures", []) or []:
                    if square_index(pos.x, pos.y) == last_cap_sq:
                        return True
                return square_index(m.position.x, m.position.y) == last_cap_sq

            moves_to_search.sort(key=lambda pm: (not _is_recap(pm)))
        except Exception as e:
            print(f"Error sorting moves: {e}")
            pass
    zobrist = var.get("zobrist")

    for piece, move_opt in moves_to_search:
        new_board, mapped_piece, mapped_move = clone_and_apply_move(board, piece, move_opt)
        if new_board is None:
            continue

        next_player = new_board.players[1] if player.name == "white" else new_board.players[0]

        next_pieces = list(new_board.get_pieces())
        next_phase = calculate_game_phase(next_pieces)

        if var.get("flags", {}).get("qsee", True) and not in_check_now:
            is_capture = getattr(move_opt, "captures", None)
            if is_capture:
                victim = None
                victim_pos = None
                for pos in move_opt.captures:
                    try:
                        vp = board[Square(pos.x, pos.y)].piece
                    except Exception:
                        vp = None
                    if vp:
                        victim = vp
                        victim_pos = pos
                        break
                if not victim:
                    try:
                        victim = board[Square(move_opt.position.x, move_opt.position.y)].piece
                    except Exception:
                        victim = None
                    victim_pos = move_opt.position
                if victim:
                    vval = get_piece_base_value(victim, "mg")
                    if stand_pat + vval + 50 < alpha:
                        continue
                try:
                    bbpos = bb_from_board(board)
                    occ = bbpos.occ_all
                    tgt_sq = square_index(victim_pos.x, victim_pos.y)
                    stm_white = getattr(piece.player, "name", "white") == "white"
                    see_gain = bb_see_njit(
                        int(tgt_sq),
                        bool(stm_white),
                        int(occ),
                        int(vval if victim else 0),
                        all_bbs_list_for_see.copy() # Use the cached list
                    )
                    if see_gain < 0:
                        continue
                except Exception as e:
                    print(f"Error in SEE: {e}")
                    pass

        next_last_cap_sq = None
        try:
            # Determine if this move involved a capture
            had_capture = bool(getattr(move_opt, "captures", None))

            if had_capture:
                cap_positions = getattr(move_opt, "captures", None)
                candidate = None
                if cap_positions:
                    # Safely pick one capture square regardless of container type (list/set/tuple)
                    try:
                        candidate = next(iter(cap_positions))
                    except Exception as e:
                        print(f"Error getting capture positions: {e}")
                        candidate = None

                def _coords_from(obj):
                    try:
                        return obj.x, obj.y
                    except Exception as e:
                        print(f"Error getting coordinates from object: {e}")
                        try:
                            seq = list(obj)
                            if len(seq) >= 2:
                                return seq[0], seq[1]
                        except Exception as e:
                            print(f"Error getting coordinates from sequence: {e}")
                        return None, None

                if candidate is not None:
                    cx, cy = _coords_from(candidate)
                    if isinstance(cx, int) and isinstance(cy, int):
                        next_last_cap_sq = square_index(cx, cy)

                if next_last_cap_sq is None and getattr(move_opt, "position", None):
                    px, py = _coords_from(move_opt.position)
                    if isinstance(px, int) and isinstance(py, int):
                        next_last_cap_sq = square_index(px, py)
        except Exception as e:
            print(f"Error getting last cap sq: {e}")
            next_last_cap_sq = None

        next_hash = zobrist.compute_full_hash(new_board, next_player.name) if zobrist else 0
        score = -quiescence_search(
            new_board,
            next_player,
            depth - 1,
            -beta,
            -alpha,
            next_pieces,
            next_phase,
            next_hash,
            var,
            next_last_cap_sq,
        )

        if score >= beta:
            return beta

        if score > alpha:
            alpha = score

    return alpha


# =====================================================================
# === Main Search (MODIFIED)
# =====================================================================


 


def is_in_check(board, player, bbpos: Optional["BBPos"] = None) -> bool:
    """
    Detects whether `player`'s king is currently in check using unified attacker logic.
    
    This uses `attackers_to_square` to avoid duplication and drift from the
    main legality machinery.
    """
    bbpos = bb_from_board(board) if bbpos is None else bbpos
    # Select king and attacker color
    if player.name == "white":
        kbb = bbpos.WK
        by_white = False  # attackers are black
    else:
        kbb = bbpos.BK
        by_white = True   # attackers are white
    if kbb == 0:
        return False
    ksq = _pop_lsb_njit(kbb)
    attackers = attackers_to_square(bbpos, ksq, by_white=by_white, occ_override=bbpos.occ_all)
    return bool(attackers)




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
    global LOG_FILE, MOVE_COUNTER
    if LOG_FILE is None:
        init_log_file()
    MOVE_COUNTER += 1
    
    # Track time across the game and per move
    current_move_start_time = time.perf_counter()
    var_state.setdefault("total_time_used_s", 0.0)
    var_state.setdefault("game_ply", 0)
    var_state["game_ply"] += 1
    
    ply_so_far = var_state["game_ply"]
    log_message(f"\n{'='*60}")
    log_message(f"Move #{MOVE_COUNTER} (Ply {ply_so_far})")
    log_message(f"{'='*60}")

    var_state.setdefault("zobrist", Zobrist())
    var_state.setdefault("contempt", 0.0)
    var_state.setdefault("capture_history", {})
    var_state.setdefault("cont_history", {})
    var_state.setdefault(
        "flags",
        {
            "nmp": True,
            "lmr": True,
            "futility": True,
            "qsee": True,
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
    dynamic_budget_s = (remaining_time_s / moves_remaining)
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
    print(
        f"DEBUG: Ply {ply_so_far}, Rem_Time {remaining_time_s:.2f}s, Moves_Rem {moves_remaining}, Budget {soft_time:.2f}s"
    )

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
            log_message("ROOT: No legal moves (terminal). Returning (None, None).")
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
        
        log_message(f"Time budget: soft={soft_time:.2f}s, hard={hard_time:.2f}s")
        # Adaptive time management trackers per-iteration
        prev_iter_best_score: Optional[float] = None
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
                    print(f"[BB_AGENT] Timeout at depth {depth}, using best move from previous depth", file=sys.stderr)
                    log_message(f"Depth {depth}: TIMEOUT")
                else:
                    print(f"[BB_AGENT] Error at depth {depth}: {e}", file=sys.stderr)
                    log_message(f"Depth {depth}: ERROR - {e}")
                break  # Exit the depth loop, preserving best_move_bb from previous iterations

            if depth_completed:
                depth_time = time.perf_counter() - depth_start_time
                nodes = var_state.get('_nodes', 0)
                qnodes = var_state.get('_qnodes', 0)
                move_str = "None"
                if best_move_bb:
                    sx, sy, dx, dy = bbmove_to_tuple_xy(best_move_bb)
                    move_str = f"({sx},{sy})->({dx},{dy})"
                
                print(
                    f"DEBUG[BB]: Depth {depth} finished. Nodes={nodes}, QNodes={qnodes}",
                    file=sys.stderr,
                )
                log_message(f"Depth {depth}: score={best_score:.0f}, move={move_str}, time={depth_time:.3f}s, nodes={nodes}, qnodes={qnodes}")
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
                    prev_iter_best_score = best_score
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
            log_message(f"Mapping root move via precomputed table: move_tuple=({sx},{sy},{dx},{dy}), best_move_bb={best_move_bb}")
            pair = var_state.get("_root_moves_map", {}).get((sx, sy, dx, dy))
            if pair:
                piece, move = pair
                move_str = log_move_str(piece, move)
                log_message(f"FINAL: {move_str}, score={best_score:.0f}, time={total_time:.3f}s, nodes={total_nodes}, qnodes={total_qnodes}")
                log_message(f"{'='*60}\n")
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
            log_message(f"FALLBACK (deterministic): {move_str}, time={total_time:.3f}s")
            log_message(f"{'='*60}\n")
            try:
                var_state["total_time_used_s"] += total_time
            except Exception:
                pass
            if not isinstance(var, dict):
                _PERSISTENT_VAR = var_state
            return piece, move
        else:
            log_message("FALLBACK: No legal moves found at root (terminal).")
            log_message(f"{'='*60}\n")
            try:
                var_state["total_time_used_s"] += total_time
            except Exception:
                pass
            if not isinstance(var, dict):
                _PERSISTENT_VAR = var_state
            return None, None
    except Exception as e:
        print(f"[BB_AGENT_ERROR] {e}", file=sys.stderr)

        # Return the best move from the *previous* completed depth if available
        total_time = time.perf_counter() - current_move_start_time
        if best_move_bb is not None:
            print("[BB_AGENT_FIX] Using best move from previous depth due to timeout.", file=sys.stderr)
            sx, sy, dx, dy = bbmove_to_tuple_xy(best_move_bb)
            log_message(f"[TIMEOUT] Mapping via precomputed root table for move_tuple=({sx},{sy},{dx},{dy})")
            pair = var_state.get("_root_moves_map", {}).get((sx, sy, dx, dy))
            if pair:
                piece, move = pair
                move_str = log_move_str(piece, move)
                log_message(f"FINAL (timeout): {move_str}, time={total_time:.3f}s")
                log_message(f"{'='*60}\n")
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
                log_message(f"[FALLBACK] Mapping best_tuple via precomputed table: {best_tuple}")
                fallback_pair = var_state.get("_root_moves_map", {}).get(tuple(best_tuple))
                if fallback_pair:
                    piece, move = fallback_pair
                    move_str = log_move_str(piece, move)
                    log_message(f"FALLBACK (using previous depth): {move_str}, time={total_time:.3f}s")
                    # ... (return pair) ...
                    return fallback_pair
        except Exception:
            pass # Failed to use previous depth's move, proceed to random.
        # --- END FIX ---

        # Deterministic final fallback using precomputed root moves map
        rm = var_state.get("_root_moves_map", {})
        if rm:
            first_tuple = next(iter(rm.keys()))
            piece, move = rm[first_tuple]
            log_message(f"FALLBACK (deterministic final): {log_move_str(piece, move)}, time={total_time:.3f}s")
            return piece, move
        log_message("FALLBACK: No legal moves found (terminal).")
        return None, None
