"""
One-time offline build script: converts a Polyglot opening book (.bin) into
book.py, a plain-Python dict keyed by *our own* engine's Zobrist hash
(agent.py's `_ZOBRIST`), mapping each catalogued position to a single
recommended move.

Not imported by agent.py at runtime -- this is a build-time tool only.
Run it whenever you want to regenerate book.py (different source book,
different depth, or a different forced first move):

    python build_book.py

Requires the `chess` (python-chess) package, used only here for reading the
Polyglot book and walking legal chess positions -- agent.py itself has no
runtime dependency on it.
"""
from collections import deque

import chess
import chess.polyglot

import agent as A

# === Configuration ===
POLYGLOT_BOOK_PATH = "book_data/performance.bin"
OUTPUT_PATH = "book.py"
MAX_PLY = 12                    # 6 full moves per side
FORCED_FIRST_MOVE_UCI = "d2d4"  # White always opens with the Queen's Pawn (d4)

PC_PIECE_TO_OUR = {
    chess.PAWN: A.PAWN, chess.KNIGHT: A.KNIGHT, chess.BISHOP: A.BISHOP,
    chess.ROOK: A.ROOK, chess.QUEEN: A.QUEEN, chess.KING: A.KING,
}
PC_PIECE_TO_LETTER = {
    chess.PAWN: 'P', chess.KNIGHT: 'N', chess.BISHOP: 'B',
    chess.ROOK: 'R', chess.QUEEN: 'Q', chess.KING: 'K',
}


def pc_square_to_our(sq_pc: int) -> int:
    """
    Translate a python-chess square index (a1=0..h8=63, rank-major from
    White's side) into agent.py's square index (y*8+x, where y=0 is the
    *black* back rank -- see agent.py's square_index/board_to_bitboard).
    """
    file_idx = chess.square_file(sq_pc)  # 0=a..7=h
    rank_idx = chess.square_rank(sq_pc)  # 0=rank1..7=rank8
    x = file_idx
    y = 7 - rank_idx
    return y * 8 + x


def our_hash_for_board(board_pc: chess.Board) -> int:
    """Build an agent.py BitboardState purely from a python-chess Board and
    return agent.py's own Zobrist hash for it (the same hash used by the
    live engine's board_to_bitboard/transposition table)."""
    bbs = {k: 0 for k in ('WP', 'WN', 'WB', 'WQ', 'WK', 'WR', 'BP', 'BN', 'BB', 'BQ', 'BK', 'BR')}
    for sq_pc, piece in board_pc.piece_map().items():
        our_sq = pc_square_to_our(sq_pc)
        color_prefix = 'W' if piece.color == chess.WHITE else 'B'
        key = color_prefix + PC_PIECE_TO_LETTER[piece.piece_type]
        bbs[key] |= (1 << our_sq)

    occ_white = bbs['WP'] | bbs['WN'] | bbs['WB'] | bbs['WQ'] | bbs['WK'] | bbs['WR']
    occ_black = bbs['BP'] | bbs['BN'] | bbs['BB'] | bbs['BQ'] | bbs['BK'] | bbs['BR']

    castling_rights = 0
    if board_pc.has_kingside_castling_rights(chess.WHITE):
        castling_rights |= A.CASTLE_WK
    if board_pc.has_queenside_castling_rights(chess.WHITE):
        castling_rights |= A.CASTLE_WQ
    if board_pc.has_kingside_castling_rights(chess.BLACK):
        castling_rights |= A.CASTLE_BK
    if board_pc.has_queenside_castling_rights(chess.BLACK):
        castling_rights |= A.CASTLE_BQ

    state = A.BitboardState(
        WP=bbs['WP'], WN=bbs['WN'], WB=bbs['WB'], WQ=bbs['WQ'], WK=bbs['WK'], WR=bbs['WR'],
        BP=bbs['BP'], BN=bbs['BN'], BB=bbs['BB'], BQ=bbs['BQ'], BK=bbs['BK'], BR=bbs['BR'],
        occ_white=occ_white, occ_black=occ_black, occ_all=occ_white | occ_black,
        side_to_move=(0 if board_pc.turn == chess.WHITE else 1),
        zobrist_hash=0, en_passant_square=-1, castling_rights=castling_rights,
    )
    return A._ZOBRIST.compute_hash(state)


def move_to_book_entry(board_pc: chess.Board, move: chess.Move):
    """Convert a python-chess Move (for the given position) into our
    (from_sq, to_sq, promo, castle) tuple, using agent.py's own conventions."""
    from_sq = pc_square_to_our(move.from_square)
    to_sq = pc_square_to_our(move.to_square)
    castle = 0
    if board_pc.is_kingside_castling(move):
        castle = 1
    elif board_pc.is_queenside_castling(move):
        castle = 2
    promo = PC_PIECE_TO_OUR.get(move.promotion, 0) if move.promotion else 0
    return (from_sq, to_sq, promo, castle)


def build() -> dict:
    book: dict[int, tuple] = {}
    visited: set[int] = set()

    start = chess.Board()
    start_hash = our_hash_for_board(start)
    visited.add(start_hash)
    queue = deque([(start, 0)])

    forced_first_move = chess.Move.from_uci(FORCED_FIRST_MOVE_UCI)

    with chess.polyglot.open_reader(POLYGLOT_BOOK_PATH) as reader:
        while queue:
            board_pc, depth = queue.popleft()
            if depth >= MAX_PLY:
                continue

            entries = list(reader.find_all(board_pc))
            if not entries:
                continue

            h = our_hash_for_board(board_pc)

            # Pick the recommended move: highest Polyglot weight (matches
            # reader.find()'s own "main entry" semantics), except at the
            # very start of the game, where White always opens 1.d4.
            if depth == 0 and forced_first_move in board_pc.legal_moves:
                chosen_move = forced_first_move
            else:
                chosen_move = max(entries, key=lambda e: e.weight).move

            book[h] = move_to_book_entry(board_pc, chosen_move)

            # Explore every book-known reply as a child, so we have an
            # answer ready no matter which one the opponent actually plays.
            children = {entry.move for entry in entries}
            children.add(forced_first_move if depth == 0 else chosen_move)
            for mv in children:
                if mv not in board_pc.legal_moves:
                    continue
                child_board = board_pc.copy()
                child_board.push(mv)
                child_hash = our_hash_for_board(child_board)
                if child_hash not in visited:
                    visited.add(child_hash)
                    queue.append((child_board, depth + 1))

    return book


def write_book(book: dict) -> None:
    lines = [
        '"""',
        'Auto-generated opening book -- DO NOT EDIT BY HAND.',
        f'Built by build_book.py from {POLYGLOT_BOOK_PATH} (max {MAX_PLY} plies).',
        '',
        "Keyed by agent.py's own Zobrist hash (same hash used by the live",
        'engine and its transposition table). Each value is',
        '(from_sq, to_sq, promo, castle) in agent.py square/BBMove conventions.',
        '"""',
        '',
        'OPENING_BOOK = {',
    ]
    for h, entry in book.items():
        lines.append(f'    {h}: {entry!r},')
    lines.append('}')
    lines.append('')
    with open(OUTPUT_PATH, 'w') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    book = build()
    write_book(book)
    print(f'Built {len(book)} book positions (max {MAX_PLY} plies) -> {OUTPUT_PATH}')
