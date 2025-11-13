# agent.py


import random
import math
import time
import sys
from typing import Dict, Tuple, Optional, List, Any, NamedTuple
from chessmaker.chess.base import Square
from extension.board_utils import list_legal_moves_for, copy_piece_move

# Logging/diagnostics flags
DEBUG = False
ENABLE_TREE_LOGGING = False

# Redirect all print output to a file
# _log_file = open("agent5_output.log", "a", encoding="utf-8")
# _original_stdout = sys.stdout
# _original_stderr = sys.stderr
# sys.stdout = _log_file
# sys.stderr = _log_file


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
EVAL_KING_SHIELD_WEAK = -10
EVAL_KING_SHIELD_GONE = -20
EVAL_KING_STORM = -6

# Endgame drive weights (5x5 tuned, modest to avoid overpowering MG)
EVAL_EG_OPP_KING_TO_EDGE_R = 14  # KRK: drive to edge
EVAL_EG_OPP_KING_TO_CORNER_R = 12  # KRK: then corner
EVAL_EG_OPP_KING_TO_EDGE_Q = 10  # KQK: edge is OK
EVAL_EG_OPP_KING_TO_CORNER_Q = 14  # KQK: prefer corner
EVAL_EG_OWN_KING_PROXIMITY = 8  # Bring our king closer in KRK/KQK
EVAL_EG_ROOK_CUTOFF = 10  # Rook/right on same rank/file as enemy king
EVAL_EG_QUEEN_ADJ_PENALTY = -20  # Avoid stalemate-y queen adjacency if our king is far


PIECE_VALUES_MG = {
    "Pawn": 120,
    "Knight": 350,
    "Bishop": 330,
    "Rook": 500,
    "Right": 500,
    "Queen": 900,
    "King": 20000,
}
PIECE_VALUES_EG = {
    "Pawn": 140,
    "Knight": 350,
    "Bishop": 330,
    "Rook": 500,
    "Right": 500,
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
PAWN_PST_MG = _ZERO_PST
PAWN_PST_EG = _ZERO_PST
KNIGHT_PST_MG = _ZERO_PST
KNIGHT_PST_EG = _ZERO_PST
BISHOP_PST_MG = _ZERO_PST
BISHOP_PST_EG = _ZERO_PST
ROOK_PST_MG = _ZERO_PST
ROOK_PST_EG = _ZERO_PST
RIGHT_PST_MG = _ZERO_PST
RIGHT_PST_EG = _ZERO_PST
QUEEN_PST_MG = _ZERO_PST
QUEEN_PST_EG = _ZERO_PST
KING_PST_MG = _ZERO_PST  # King safety is now handled dynamically
KING_PST_EG = _ZERO_PST

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

def _serialize_board_to_codes(board) -> List[List[str]]:
    """
    Returns a 5x5 array of compact piece codes (e.g., "wP", "bK", "wRi").

    Empty squares are "".
    """
    grid: List[List[str]] = [["" for _ in range(5)] for _ in range(5)]
    try:
        for piece in board.get_pieces():
            pos = getattr(piece, "position", None)
            if not pos:
                continue
            color = getattr(piece.player, "name", "white")
            color_code = "w" if color == "white" else "b"
            name = getattr(piece, "name", "")
            short = {
                "Pawn": "P",
                "Knight": "N",
                "Bishop": "B",
                "Rook": "R",
                "Queen": "Q",
                "King": "K",
                "Right": "Ri",
            }.get(name, name[:2] or "?")
            grid[pos.y][pos.x] = f"{color_code}{short}"
    except Exception as e:
        print(f"Error serializing board: {e}")
    return grid


def _tree_init(var: Dict[str, Any], root_player_name: str, board) -> None:
    """
    Initializes a fresh tree container in var under key "search_tree".
    """
    if not ENABLE_TREE_LOGGING:
        return
    tree = {
        "root": None,  # last root in this search call
        "roots": [],  # all iteration roots in this search call
        "nodes": {},
        "meta": {
            "color": root_player_name,
            "maxDepth": 0,
            "elapsedMs": 0,
        },
    }
    var["search_tree"] = tree
    var["search_tree_next_id"] = 1


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


def _tree_new_node(
    var: Dict[str, Any],
    parent_id: Optional[int],
    *,
    ply: int,
    depth: int,
    node_type: str,
    player_name: str,
    alpha0: float,
    beta0: float,
    move_str: Optional[str],
    in_check: Optional[bool],
    board_codes: Optional[List[List[str]]],
) -> int:
    """
    Appends a node to the current tree and returns its id.
    """
    if not ENABLE_TREE_LOGGING:
        return -1
    try:
        tree = var.get("search_tree")
        if tree is None:
            return -1
        node_id = int(var.get("search_tree_next_id", 1))
        var["search_tree_next_id"] = node_id + 1
        node = {
            "id": node_id,
            "parent": parent_id,
            "ply": int(ply),
            "depth": int(depth),
            "nodeType": str(node_type or "negamax"),
            "player": str(player_name or "white"),
            "alpha0": _json_safe_number(alpha0),
            "beta0": _json_safe_number(beta0),
            "score": None,
            "moveStr": move_str,
            "inCheck": bool(in_check) if in_check is not None else None,
            "children": [],
            "board": board_codes,
            "reason": None,
            "cutReason": None,
            "bestChildMove": None,
        }
        tree["nodes"][node_id] = node
        if parent_id is None:
            tree["root"] = node_id
            try:
                tree.setdefault("roots", []).append(node_id)
            except Exception:
                pass
        else:
            parent = tree["nodes"].get(parent_id)
            if parent is not None:
                parent.setdefault("children", []).append(node_id)
        try:
            tree["meta"]["maxDepth"] = max(int(tree["meta"].get("maxDepth", 0)), int(depth))
        except Exception:
            pass
        return node_id
    except Exception as e:
        print(f"Error creating tree node: {e}")
        return -1


def _mark_pv_chain(tree: Dict[str, Any]) -> None:
    """
    Marks the principal variation chain(s) in-place by setting node["pv"] = True.

    Follows bestChildMove from each root and walks down by matching
    child.moveStr to parent.bestChildMove.
    """
    if not ENABLE_TREE_LOGGING:
        return
    try:
        if not tree or not isinstance(tree, dict):
            return
        nodes: Dict[int, Dict[str, Any]] = tree.get("nodes", {})
        if not nodes:
            return

        # normalize keys to int for safety
        def _get_node(nid):
            try:
                return nodes[nid]
            except KeyError:
                try:
                    return nodes[str(nid)]
                except Exception:
                    return None

        roots = tree.get("roots") or ([tree.get("root")] if tree.get("root") is not None else [])
        for root_id in roots:
            cur = root_id
            visited = set()
            while True:
                node = _get_node(cur)
                if node is None or cur in visited:
                    break
                node["pv"] = True
                visited.add(cur)
                bcm = node.get("bestChildMove")
                if not bcm:
                    break
                next_id = None
                for cid in node.get("children", []) or []:
                    cn = _get_node(cid)
                    if cn is not None and cn.get("moveStr") == bcm:
                        next_id = cn.get("id", cid)
                        break
                if next_id is None:
                    break
                cur = next_id
    except Exception as e:
        print(f"Error marking PV chain: {e}")


def _tree_finalize_node(
    var: Dict[str, Any],
    node_id: int,
    *,
    score: Optional[float] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    reason: Optional[str] = None,
    cut_reason: Optional[str] = None,
    best_child_move: Optional[str] = None,
) -> None:
    """
    Sets final values on a node; no-op if tree is missing.
    """
    if not ENABLE_TREE_LOGGING:
        return
    try:
        tree = var.get("search_tree")
        if not tree:
            return
        node = tree["nodes"].get(node_id)
        if not node:
            return
        if score is not None:
            node["score"] = _json_safe_number(score)
        if alpha is not None:
            node["alpha1"] = _json_safe_number(alpha)
        if beta is not None:
            node["beta1"] = _json_safe_number(beta)
        if reason is not None:
            node["reason"] = str(reason)
        if cut_reason is not None:
            node["cutReason"] = str(cut_reason)
        if best_child_move is not None:
            node["bestChildMove"] = str(best_child_move)
    except Exception as e:
        print(f"Error finalizing tree node: {e}")


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


def _get_rook_attacks(sq: int, occ: int) -> int:
    """
    Computes rook ray attacks from a square on a 5x5 board.

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


def _get_bishop_attacks(sq: int, occ: int) -> int:
    """
    Computes bishop diagonal ray attacks from a square on a 5x5 board.

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
                cap_piece = board[Square(pos.x, pos.y)].piece
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

    def __init__(self, entry_count: int = 262144):  # 1048576):
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

    scored_moves = [
        (
            score_move(
                board, p, m, var, ply, hash_move_tuple, prev_move_tuple, bbpos, all_bbs_list
            ),
            (p, m),
        )
        for p, m in legal_moves
    ]

    scored_moves.sort(key=lambda x: x[0], reverse=True)
    max_moves = 12 if var.get("_time_pressure") else 20
    trimmed = scored_moves[:max_moves]
    return [move for score, move in trimmed]


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
        staged.append(((stg, -scr), (p, m)))
    staged.sort(key=lambda x: x[0])
    max_moves = 12 if var.get("_time_pressure") else 20
    return [mv for _, mv in staged[:max_moves]]


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
                        [
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
                        ],
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


def find_move_from_tuple(
    board, player, move_tuple: Optional[Tuple[int, int, int, int]]
) -> Optional[Tuple[Any, Any]]:
    """
    Resolves a (sx, sy, dx, dy) tuple to an actual (piece, move_opt) pair.

    Args:
        board: Current board.
        player: Player who is supposed to make the move.
        move_tuple: Tuple (sx, sy, dx, dy) or None.

    Returns:
        (piece, move_opt) if the move is legal for `player`; otherwise None.

    How it works:
        - Reads the piece from the source square and, if it belongs to `player`,
          iterates `piece.get_move_options()` to match the destination.
    """
    if move_tuple is None:
        return None
    sx, sy, dx, dy = move_tuple

    try:
        piece = board[Square(sx, sy)].piece
    except Exception:
        piece = None

    if piece and piece.player.name == player.name:
        try:
            for m in piece.get_move_options():
                if m.position.x == dx and m.position.y == dy:
                    return (piece, m)
        except Exception as e:
            print(f"Error getting move options: {e}")
            pass

    return None


def is_in_check(board, player, bbpos: Optional["BBPos"] = None) -> bool:
    """
    Detects whether `player`'s king is currently in check using bitboards.

    Args:
        board: Current board.
        player: Player whose king is tested for check.

    Returns:
        True if the king is attacked by any opposing piece; False otherwise.

    How it works:
        - Builds `BBPos` and locates the king square.
        - Tests attacks from opponent knights, bishops/queens (diagonals),
          rooks/queens/custom Right (ranks/files), king, and pawn diagonals.
    """
    bbpos = bb_from_board(board) if bbpos is None else bbpos
    occ = bbpos.occ_all
    if player.name == "white":
        king_bb = bbpos.WK
        their_knights, their_bishops = bbpos.BN, bbpos.BB
        their_rooks, their_queens = bbpos.BR, bbpos.BQ
        their_rights, their_king = bbpos.BRi, bbpos.BK
        their_pawns = bbpos.BP
        color_white = False
    else:
        king_bb = bbpos.BK
        their_knights, their_bishops = bbpos.WN, bbpos.WB
        their_rooks, their_queens = bbpos.WR, bbpos.WQ
        their_rights, their_king = bbpos.WRi, bbpos.WK
        their_pawns = bbpos.WP
        color_white = True
    if king_bb == 0:
        return False
    ksq = _pop_lsb_njit(king_bb)
    if _KNIGHT_MOVES[ksq] & their_knights:
        return True
    if _KING_MOVES[ksq] & their_king:
        return True
    if _get_bishop_attacks(ksq, occ) & (their_bishops | their_queens):
        return True
    if _get_rook_attacks(ksq, occ) & (their_rooks | their_queens | their_rights):
        return True
    r, c = ksq // 5, ksq % 5
    if color_white:
        a1 = (r - 1, c - 1) if r > 0 and c > 0 else None
        a2 = (r - 1, c + 1) if r > 0 and c < 4 else None
    else:
        a1 = (r + 1, c - 1) if r < 4 and c > 0 else None
        a2 = (r + 1, c + 1) if r < 4 and c < 4 else None
    for a in (a1, a2):
        if a is not None and (their_pawns & (1 << (a[0] * 5 + a[1]))):
            return True
    return False


# =====================================================================
# === Main Search (MODIFIED)
# =====================================================================


def negamax(
    board,
    player,
    depth,
    alpha,
    beta,
    var,
    ply,
    current_hash,
    *,
    tree_parent_id: Optional[int] = None,
    move_str: Optional[str] = None,
    node_type: str = "negamax",
):
    """
    Core negamax search with alpha-beta pruning and modern heuristics.

    Implements repetition detection, TT probing/storage, aspiration windows
    (from caller), futility/razoring, null-move pruning (configurable), late
    move reductions (LMR), history/killers/countermoves ordering, and PV
    search re-searching. Uses incremental make/unmake with Zobrist hashing.

    Check extension:
        If a child move gives check to the opponent side, the child search
        depth is extended by 1 ply to improve mate net detection in endgames.

    Args:
        board: Current board.
        player: Side to move at this node.
        depth: Remaining depth in plies (integer >= 0).
        alpha: Alpha bound.
        beta: Beta bound.
        var: Mutable search state (TT, zobrist, heuristics, timers, stacks).
        ply: Root-relative ply index (0 at root).
        current_hash: Zobrist hash for the current node.

    Returns:
        Tuple (score, best_move_pair) where best_move_pair is (piece, move_opt)
        or None when no legal moves or at leaf.
    """

    var["_nodes"] = var.get("_nodes", 0) + 1
    var.setdefault("_rep_stack", [])
    var.setdefault("_cur_plies_irreversible", 0)
    var.setdefault("_cur_plies_from_null", 10**9)
    if ply == 0:
        # Seed repetition stack/counts from game history without double-counting the current hash.
        _rep_init = list(var.get("game_hash_history", []))
        if _rep_init and _rep_init[-1] == current_hash:
            _rep_init.pop()
        var["_rep_stack"] = _rep_init
        # Build counts for O(1) repetition detection
        rc = {}
        try:
            for h in _rep_init:
                rc[h] = rc.get(h, 0) + 1
        except Exception:
            pass
        var["_rep_counts"] = rc
        print(
            f"[ROOT] Starting depth {depth}, hash={current_hash}",
            file=sys.stderr,
        )

    # Precompute bitboards for this node
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

    # Determine if side to move is in check
    try:
        nm_in_check = is_in_check(board, player, bbpos=bbpos)
    except Exception:
        nm_in_check = False
    tree_node_id = None

    # Node-managed repetition list: push this node's hash

    var["_rep_stack"].append(current_hash)
    try:
        var["_rep_counts"][current_hash] = var.get("_rep_counts", {}).get(current_hash, 0) + 1
    except Exception:
        pass

    try:
        # O(1) repetition check using counts on the current path
        if var.get("_rep_counts", {}).get(current_hash, 0) >= 2:
            contempt = float(var.get("contempt", 0.0))
            sc = contempt if player.name == "white" else -contempt
            return sc, None
        # Game history repetition (also O(1) via global counts when available)
        if var.get("_game_rep_counts", {}).get(current_hash, 0) >= 2:
            contempt = float(var.get("contempt", 0.0))
            sc = contempt if player.name == "white" else -contempt
            return sc, None
    except Exception as e:
        print(f"Error in game hash history: {e}")
        pass
    try:
        # Check time at every node to avoid overshooting budgets
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

        pieces = list(board.get_pieces())
        game_phase = calculate_game_phase(pieces)

 

        board_hash = current_hash
        tt: TranspositionTable = var["transposition_table"]
        tt_entry = tt.probe(board_hash)
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
                except Exception as e:
                    print(f"Error handling mate value: {e}")
                    pass
                flag = tt_entry.flag
                best_move = find_move_from_tuple(board, player, hash_move_tuple)

                if flag == TT_FLAG_EXACT:
                    return score, best_move
                elif flag == TT_FLAG_LOWER:
                    alpha = max(alpha, score)
                elif flag == TT_FLAG_UPPER:
                    beta = min(beta, score)

                if alpha >= beta:
                    return score, best_move

        if hash_move_tuple is None and depth >= 5:
            try:
                seed_depth = 2
                _seed_score, _seed_move = negamax(
                    board,
                    player,
                    seed_depth,
                    alpha,
                    beta,
                    var,
                    ply,
                    current_hash,
                    tree_parent_id=tree_node_id,
                    move_str="(seed d=2)",
                    node_type="seed",
                )
                if _seed_move:
                    sp, sm = _seed_move
                    hash_move_tuple = (
                        sp.position.x,
                        sp.position.y,
                        sm.position.x,
                        sm.position.y,
                    )
            except Exception as e:
                print(f"Error seeding move: {e}")
                pass

        is_pv_node = (alpha + 1) < beta
        in_check = nm_in_check

        if ply == 0:
            pass

        if alpha > WIN_VALUE - 200:
            return alpha, None
        if beta < -WIN_VALUE + 200:
            return beta, None

        if depth == 0:
            leaf_score = evaluate_position_static(board, player, pieces, game_phase)
            if leaf_score >= beta:
                return beta, None
            if leaf_score > alpha:
                alpha = leaf_score

            quiescence_score = quiescence_search(
                board,
                player,
                8,
                alpha,
                beta,
                pieces,
                game_phase,
                current_hash,
                var,
            )
            return quiescence_score, None

        if (
            var.get("flags", {}).get("futility", True)
            and depth <= 1
            and not is_pv_node
            and not in_check
        ):
            static_eval = evaluate_position_static(board, player, pieces, game_phase)
            futility_margin = 120 * depth
            if static_eval + futility_margin <= alpha:
                return static_eval, None
            if depth == 1:
                razor_margin = 200
                if static_eval + razor_margin <= alpha:
                    quiescence_score = quiescence_search(
                        board,
                        player,
                        1,
                        alpha,
                        beta,
                        pieces,
                        game_phase,
                        current_hash,
                        var,
                    )
                    if quiescence_score <= alpha:
                        return alpha, None

        has_non_pawn_piece = False
        try:
            for pc in pieces:
                if getattr(pc.player, "name", "white") == player.name and pc.name not in (
                    "Pawn",
                    "King",
                ):
                    has_non_pawn_piece = True
                    break
        except Exception as e:
            print(f"Error checking non-pawn piece: {e}")
            has_non_pawn_piece = True
        if (
            var.get("flags", {}).get("nmp", True)
            and ply > 0
            and depth >= 3
            and (not is_pv_node)
            and game_phase > 2
            and not in_check
            and has_non_pawn_piece
        ):
            # if list_legal_moves_for(board, player):
                R = 2 + (depth // 6)
                next_player = board.players[1] if player.name == "white" else board.players[0]
                null_move_hash = current_hash ^ var["zobrist"].black_to_move_hash
                _saved_null_plies = var.get("_cur_plies_from_null", 10**9)
                var["_cur_plies_from_null"] = 0
                try:
                    score_nmp, _ = negamax(
                        board,
                        next_player,
                        depth - 1 - R,
                        -beta,
                        -beta + 1,
                        var,
                        ply + 1,
                        null_move_hash,
                    )
                finally:
                    var["_cur_plies_from_null"] = _saved_null_plies
                if -score_nmp >= beta:
                    if depth - 2 >= 1:
                        v_score, _ = negamax(
                            board,
                            next_player,
                            depth - 2,
                            -beta,
                            -beta + 1,
                            var,
                            ply + 1,
                            null_move_hash,
                        )
                        if -v_score >= beta:
                            return beta, None
                    else:
                        return beta, None

        moves = get_ordered_moves_optimized(
            board,
            player,
            var,
            ply,
            hash_move_tuple,
            var.get("_prev_move_tuple"),
            captures_only=False,
            bbpos=bbpos,
            all_bbs_list=all_bbs_list,
            board_hash=current_hash,
        )

        if not moves:
            mate_score = -1_000_000 + ply
            if in_check:
                return mate_score, None
            else:
                c = float(var.get("contempt", 0.0))
                sc2 = c if player.name == "white" else -c
                return sc2, None

        best_score = -INF
        best_move_in_this_node = None
        original_alpha = alpha
        best_pre_tuple = None

        for i, (piece, move_opt) in enumerate(moves):
            try:
                pre_tuple = (
                    piece.position.x,
                    piece.position.y,
                    move_opt.position.x,
                    move_opt.position.y,
                )
            except Exception as e:
                print(f"Error getting pre tuple: {e}")
                continue

            is_capture = getattr(move_opt, "captures", None)
            if (not is_capture) and (not is_pv_node) and depth <= 3 and i >= 12 and not in_check:
                continue
            new_board, mapped_piece, mapped_move = clone_and_apply_move(board, piece, move_opt)
            if new_board is None:
                continue

            next_player = new_board.players[1] if player.name == "white" else new_board.players[0]
            try:
                zob = var.get("zobrist")
                if zob is not None and mapped_piece is not None:
                    next_hash = compute_incremental_hash_after_move(
                        zob, current_hash, board, piece, move_opt, mapped_piece
                    )
                else:
                    next_hash = var["zobrist"].compute_full_hash(new_board, next_player.name)
            except Exception:
                next_hash = var["zobrist"].compute_full_hash(new_board, next_player.name)

            _saved_irrev_plies = var.get("_cur_plies_irreversible", 0)
            is_capture = getattr(move_opt, "captures", None)
            is_pawn_move = getattr(piece, "name", "") == "Pawn"
            var["_cur_plies_irreversible"] = (
                0 if (is_capture or is_pawn_move) else (_saved_irrev_plies + 1)
            )
            _saved_null_plies = var.get("_cur_plies_from_null", 10**9)
            var["_cur_plies_from_null"] = _saved_null_plies + 1

            next_depth = depth - 1
            is_promo = False
            try:
                is_promo = bool(getattr(move_opt, "extra", {}).get("promote"))
            except Exception as e:
                print(f"Error checking promotion: {e}")
                is_promo = False
            is_killer = False
            try:
                if ply < len(var.get("killers", [])):
                    killers_here = var["killers"][ply]
                    is_killer = pre_tuple in killers_here
            except Exception as e:
                print(f"Error checking killer: {e}")
                is_killer = False
            if (
                var.get("flags", {}).get("lmr", True)
                and depth >= 3
                and i > 0
                and (not is_capture)
                and (not is_promo)
                and (not is_killer)
                and (not is_pv_node)
                and (not in_check)
            ):
                reduction_plies = int(math.log(depth) * math.log(i) / 1.9 + 0.5)
                if not is_pv_node:
                    reduction_plies += 1
                next_depth = max(1, next_depth - reduction_plies)

            # Check extension: if this move gives check, extend by 1 ply
            try:
                if depth >= 2 and is_in_check(new_board, next_player):
                    next_depth += 1
            except Exception:
                pass

            old_prev = var.get("_prev_move_tuple")
            var["_prev_move_tuple"] = pre_tuple

            if i == 0:
                child_score, _ = negamax(
                    new_board,
                    next_player,
                    next_depth,
                    -beta,
                    -alpha,
                    var,
                    ply + 1,
                    next_hash,
                )
                score = -child_score
            else:
                child_score, _ = negamax(
                    new_board,
                    next_player,
                    next_depth,
                    -(alpha + 1),
                    -alpha,
                    var,
                    ply + 1,
                    next_hash,
                )
                score = -child_score
                if score > alpha and score < beta:
                    child_score, _ = negamax(
                        new_board,
                        next_player,
                        next_depth,
                        -beta,
                        -alpha,
                        var,
                        ply + 1,
                        next_hash,
                    )
                    score = -child_score

            # Track best root move so far for timeout fallback
            if ply == 0:
                try:
                    if score > var.get("_root_best_score", -INF):
                        var["_root_best_score"] = score
                        var["_root_best_move_tuple"] = pre_tuple
                except Exception:
                    pass

            var["_cur_plies_irreversible"] = _saved_irrev_plies
            var["_cur_plies_from_null"] = _saved_null_plies

            if old_prev is None:
                var.pop("_prev_move_tuple", None)
            else:
                var["_prev_move_tuple"] = old_prev

            if score > best_score:
                best_score = score
                best_move_in_this_node = (piece, move_opt)
                best_pre_tuple = pre_tuple

            if score > alpha:
                alpha = score

            if alpha >= beta:
                var["_fail_highs"] = var.get("_fail_highs", 0) + 1
                if not is_capture:
                    if ply < len(var["killers"]):
                        killers = var["killers"][ply]
                        if pre_tuple != killers[0]:
                            killers[1] = killers[0]
                            killers[0] = pre_tuple
                if is_capture:
                    ch = var.setdefault("capture_history", {})
                    ch[pre_tuple] = ch.get(pre_tuple, 0) + depth**2
                else:
                    var["history"][pre_tuple] = var["history"].get(pre_tuple, 0) + depth**2
                if old_prev is not None:
                    var.setdefault("countermoves", {})[old_prev] = pre_tuple
                    cont = var.setdefault("cont_history", {})
                    cont_key = (old_prev, pre_tuple)
                    cont[cont_key] = cont.get(cont_key, 0) + depth
                break

        tt_flag = TT_FLAG_EXACT
        if best_score <= original_alpha:
            tt_flag = TT_FLAG_UPPER
        elif best_score >= beta:
            tt_flag = TT_FLAG_LOWER

        score_to_store = best_score
        try:
            if best_score > WIN_VALUE:
                score_to_store = best_score + ply
            elif best_score < -WIN_VALUE:
                score_to_store = best_score - ply
        except Exception as e:
            print(f"Error storing in TT: {e}")
            pass
        tt.store(board_hash, depth, score_to_store, tt_flag, best_pre_tuple)

        return best_score, best_move_in_this_node
    finally:
        try:
            if var.get("_rep_stack"):
                var["_rep_stack"].pop()
            if var.get("_rep_counts") is not None:
                cnt = var["_rep_counts"].get(current_hash, 0)
                if cnt > 1:
                    var["_rep_counts"][current_hash] = cnt - 1
                elif cnt == 1:
                    var["_rep_counts"].pop(current_hash, None)
        except Exception:
            pass


def _reconstruct_pv(board, player, var_dict, max_len=MAX_SEARCH_DEPTH + 8) -> List[str]:
    """
    Reconstructs the Principal Variation using TT best-move pointers.

    Args:
        board: The current board (will be temporarily modified and restored).
        player: Side to move at the PV root.
        var_dict: Search state containing `zobrist` and `transposition_table`.
        max_len: Safety cap on PV length to avoid infinite loops.

    Returns:
        A list of move strings representing the PV from the root.

    How it works:
        - Starting from the current hash, repeatedly probes the TT and finds
          the corresponding move on the board, applies it, and appends to PV.
        - Uses make/unmake to avoid cloning and restores board afterwards.
    """
    pv_moves: List[str] = []

    # Work on a clone so the original board is untouched
    try:
        work_board = board.clone()
    except Exception as e:
        print(f"Error cloning board: {e}")
        work_board = board

    cur_player = player
    steps = 0
    visited_hashes = set()

    zobrist: Zobrist = var_dict.get("zobrist")
    tt: TranspositionTable = var_dict.get("transposition_table")
    if not zobrist or not tt:
        return []

    cur_hash = var_dict.get("current_hash")
    if cur_hash is None:
        cur_hash = zobrist.compute_full_hash(work_board, cur_player.name)

    while steps < max_len:
        if cur_hash in visited_hashes:
            break
        visited_hashes.add(cur_hash)

        ent = tt.probe(cur_hash)
        if not ent or not ent.best_move_tuple:
            break

        move_tuple = ent.best_move_tuple
        move_pair = find_move_from_tuple(work_board, cur_player, move_tuple)
        if move_pair is None:
            break

        mapped_piece, mapped_move = move_pair
        pv_moves.append(move_to_str(mapped_piece, mapped_move))

        try:
            mapped_piece.move(mapped_move)
            cur_player = (
                work_board.players[1] if cur_player.name == "white" else work_board.players[0]
            )
            cur_hash = zobrist.compute_full_hash(work_board, cur_player.name)
        except Exception as e:
            print(f"Error applying PV move: {e}")
            break
        steps += 1

    return pv_moves


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

    # Track time across the game and per move
    current_move_start_time = time.perf_counter()
    var_state.setdefault("total_time_used_s", 0.0)
    var_state.setdefault("game_ply", 0)
    var_state["game_ply"] += 1

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
    MOVE_HARD_LIMIT_S = 13.0
    remaining_time_s = GAME_TIME_S - var_state["total_time_used_s"]
    ply_so_far = var_state["game_ply"]
    moves_remaining = max(10, 40 - (ply_so_far // 2))
    dynamic_budget_s = (remaining_time_s / moves_remaining) #* 0.9
    soft_time = max(0.3, min(12, dynamic_budget_s))
    hard_time = MOVE_HARD_LIMIT_S - 0.05
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

    best_piece, best_move_opt = None, None
    best_score = -INF

    try:
        legal = list_legal_moves_for(board, player)
    except Exception as e:
        print(f"Error listing legal moves: {e}")
        legal = []
    if not legal:
        legal = []
        try:
            for pc in board.get_player_pieces(player):
                try:
                    for opt in pc.get_move_options():
                        if _has_xy(pc) and _has_xy(opt):
                            legal.append((pc, opt))
                except Exception:
                    continue
        except Exception as e:
            print(f"Error getting player pieces: {e}")
            pass
    if not legal:
        print("No legal moves found")
        return None, None
    best_piece, best_move_opt = random.choice(legal)

    ASPIRATION_WINDOW_DELTA = 50

    try:
        zobrist_hasher: Zobrist = var_state["zobrist"]
        current_hash = zobrist_hasher.compute_full_hash(board, player.name)
        var_state["current_hash"] = current_hash
        try:
            if var_state.get("_last_root_hash") != current_hash:
                var_state.setdefault("game_hash_history", []).append(current_hash)
                var_state["_last_root_hash"] = current_hash
                # Track global game repetition counts for O(1) lookup
                gcc = var_state.setdefault("_game_rep_counts", {})
                gcc[current_hash] = gcc.get(current_hash, 0) + 1
        except Exception as e:
            print(f"Error setting last root hash: {e}")
            var_state.setdefault("game_hash_history", [])
    except Exception as e:
        print(f"ERROR: Hash computation failed: {e}", file=sys.stderr)
        return best_piece, best_move_opt

    try:
        var_state["_nodes"] = 0
        var_state["_qnodes"] = 0
        var_state.get("transposition_table").hits = 0
        var_state.get("transposition_table").probes = 0
    except Exception as e:
        print(f"Error initializing transposition table: {e}")
        pass

    try:
        for depth in range(START_SEARCH_DEPTH, MAX_SEARCH_DEPTH + 1):
            var_state["_id_depth"] = depth

            elapsed = time.perf_counter() - start_t
            if hard_time > 0 and elapsed >= hard_time:
                var_state["_hard_time_stop"] = True
                break
            if soft_time > 0 and elapsed >= soft_time:
                var_state["_soft_time_stop"] = True
                break

            if depth == START_SEARCH_DEPTH:
                alpha = -INF
                beta = INF
                delta = None
            else:
                delta = ASPIRATION_WINDOW_DELTA
                alpha = best_score - delta
                beta = best_score + delta

            attempt = 0
            max_attempts = 2
            used_full_window = False
            while True:
                score, move = negamax(
                    board=board,
                    player=player,
                    depth=depth,
                    alpha=alpha,
                    beta=beta,
                    var=var_state,
                    ply=0,
                    current_hash=current_hash,
                    tree_parent_id=None,
                    move_str="root",
                    node_type="negamax",
                )

                if delta is not None and score <= alpha:
                    if attempt < max_attempts:
                        attempt += 1
                        delta *= 2
                        alpha = best_score - delta
                        beta = best_score + delta
                        continue
                    if not used_full_window:
                        alpha = -INF
                        beta = INF
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
                        alpha = -INF
                        beta = INF
                        used_full_window = True
                        continue
                    break

                break

            if move:
                best_piece, best_move_opt = move
                best_score = score

            print(
                f"DEBUG: Depth {depth} finished. Nodes={var_state.get('_nodes', -1)}, QNodes={var_state.get('_qnodes', -1)}",
                file=sys.stderr,
            )
            # No post-search diagnostics to avoid overhead

    except Exception as e:
        import traceback

        print(f"[AGENT2_ERROR] Search failed at depth {depth}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # On timeout or error, try to use best root move found so far in this iteration
        try:
            best_tuple = var_state.get("_root_best_move_tuple")
            if best_tuple:
                fallback = find_move_from_tuple(board, player, best_tuple)
                if fallback:
                    best_piece, best_move_opt = fallback
                    best_score = var_state.get("_root_best_score", best_score)
        except Exception:
            pass
        var_state["_initialized_this_game"] = False

    # (Webview/tree export removed)

    # Update total time used
    try:
        elapsed_this_move = time.perf_counter() - current_move_start_time
        var_state["total_time_used_s"] += elapsed_this_move
        print(
            f"DEBUG: Move took {elapsed_this_move:.2f}s, Total used {var_state['total_time_used_s']:.2f}s"
        )
    except Exception as e:
        print(f"Error updating time usage: {e}")

    if not isinstance(var, dict):
        _PERSISTENT_VAR = var_state

    return best_piece, best_move_opt