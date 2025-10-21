"""
COMP2321 Chess Agent — Alpha-Beta (Depth = 2)
==============================================
A simple 2-ply minimax agent with alpha-beta pruning.

Search depth:
-------------
Depth 1 = Agent's move
Depth 2 = Opponent's best reply

No iterative deepening, no quiescent search — just plain alpha-beta.
"""

import random
import time
from extension.board_utils import list_legal_moves_for
from chessmaker.chess.pieces import King, Queen, Bishop, Knight, Pawn
from extension.piece_right import Right
from extension.board_rules import get_result

# ============================================================================
# CONSTANTS
# ============================================================================

PIECE_VALUES = {
    'pawn': 100,
    'knight': 320,
    'bishop': 330,
    'right': 500,
    'queen': 900,
    'king': 20000
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _other_player(board, player_name):
    return next(p for p in board.players if p.name != player_name)

def _this_player(board, player_name):
    return next(p for p in board.players if p.name == player_name)

def _piece_value(piece):
    return PIECE_VALUES.get(piece.name.lower(), 0)

def _position_key(pos):
    # Turns a Position-like object into a hashable key
    return (pos.x, pos.y)

def _piece_value_at(board, pos_key):
    for t in board.get_pieces():
        if _position_key(t.position) == pos_key:
            return _piece_value(t)
    return 0

def _has_mate_in_one(board, player_obj, player_name):
    """Return +∞ if player_obj can mate in one, else 0."""
    moves = list_legal_moves_for(board, player_obj)
    for piece, move in moves:
        b2 = board.clone()
        # find equivalent piece/move on clone
        np = next((p for p in b2.get_player_pieces(player_obj)
                   if type(p) == type(piece) and _position_key(p.position) == _position_key(piece.position)), None)
        if not np:
            continue
        nm = next((m for m in np.get_move_options()
                   if hasattr(m, "position") and _position_key(m.position) == _position_key(move.position)), None)
        if not nm:
            continue
        try:
            np.move(nm)
        except Exception:
            continue
        res = get_result(b2)
        if res:
            low = res.lower()
            if "win" in low and player_name in low:
                return 10_000_000  # immediate win
            if "win" in low and player_name not in low:
                return -10_000_000  # immediate loss (shouldn’t hit when checking our own moves)
    return 0

def _best_capture_value(board, player_obj):
    """Return the value of the best immediately capturable enemy piece for player_obj."""
    best = 0
    moves = list_legal_moves_for(board, player_obj)
    for piece, move in moves:
        if hasattr(move, "captures") and move.captures:
            for cap_pos in move.captures:
                v = _piece_value_at(board, _position_key(cap_pos))
                if v > best:
                    best = v
    return best

def _center_control_score(board, player_name):
    """
    Small bonus for being in/near the 3x3 center.
    Manhattan distance from center (2,2); reward falls with distance.
    """
    bonus = 0
    for piece in board.get_pieces():
        px, py = piece.position.x, piece.position.y
        dist = abs(px - 2) + abs(py - 2)       # 0..4 on 5x5
        # map distance to a tiny bonus: 4->0, 3->1, 2->2, 1->3, 0->4
        center_bump = max(0, 4 - dist)
        if piece.player.name == player_name:
            bonus += center_bump
        else:
            bonus -= center_bump
    # keep this deliberately tiny
    return bonus * 2   # scale the whole center effect lightly

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def evaluate_board(board, player_name):
    """
    Improved evaluation:
      1) Terminal outcomes (win/loss/draw)
      2) Material balance
      3) Mate-in-one detection (us vs them)
      4) Positive captures (best immediate capture each side)
      5) Slight center control
    Positive scores are good for 'player_name'.
    """
    # 1) Terminal check
    res = get_result(board)
    if res:
        low = res.lower()
        if "draw" in low:
            return 0
        if "win" in low:
            return 10000000 if player_name in low else -10000000

    # 2) Material
    material = 0
    for piece in board.get_pieces():
        v = _piece_value(piece)
        material += v if piece.player.name == player_name else -v

    # 3) Mate-in-one (check both sides from *current* position)
    me = _this_player(board, player_name)
    opp = _other_player(board, player_name)
    mate_me = _has_mate_in_one(board, me, player_name)               # huge + if I can mate now
    mate_opp = _has_mate_in_one(board, opp, player_name)             # huge - if they can mate now

    if mate_me != 0:
        return mate_me   # immediate win trumps everything
    if mate_opp != 0:
        return mate_opp  # immediate loss trumps everything

    # 4) Positive captures (tactical one-move swing)
    my_best_cap = _best_capture_value(board, me)
    opp_best_cap = _best_capture_value(board, opp)
    capture_swing = (my_best_cap - opp_best_cap) * 0.5  # modest weight

    # 5) Slight center control
    center = _center_control_score(board, player_name)

    # Final score
    return material + capture_swing + center

# ============================================================================
# MOVE ORDERING (simple version)
# ============================================================================

def order_moves(board, moves):
    """
    Orders moves to improve alpha-beta pruning.
    Prioritizes:
    1. Checkmates
    2. Captures of high-value pieces
    3. Center control
    """
    scored = []
    for piece, move in moves:
        score = 0

        # 1️⃣ Checkmate bonus (if known before move)
        if hasattr(move, "checkmate") and move.checkmate:
            score += 1000000

        # 2️⃣ Captures: scale by value of captured piece
        if hasattr(move, "captures") and move.captures:
            for cap_pos in move.captures:
                for t in board.get_pieces():
                    if t.position == cap_pos:
                        score += PIECE_VALUES.get(t.name.lower(), 0)
                        break
        # 3️⃣ Central preference (simple heuristic)
        center_distance = abs(move.position.x - 2) + abs(move.position.y - 2)
        score += (8 - center_distance)

        scored.append((score, piece, move))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [(p, m) for _, p, m in scored]


# ============================================================================
# MINIMAX WITH ALPHA-BETA PRUNING (Depth limited to 2)
# ============================================================================
def minimax(board, depth, alpha, beta, player_name, is_max_turn):
    """
    Standard minimax with alpha-beta pruning.
    Searches exactly 2 plies deep: agent move + opponent response.
    Detects immediate checkmates after each move.
    """
    from extension.board_rules import get_result

    # Terminal condition
    if get_result(board) is not None or depth == 0:
        return evaluate_board(board, player_name)

    current_player = board.current_player
    moves = list_legal_moves_for(board, current_player)
    if not moves:
        return evaluate_board(board, player_name)

    moves = order_moves(board, moves)

    if is_max_turn:
        value = float('-inf')
        for piece, move in moves:
            new_board = board.clone()
            try:
                # Locate corresponding piece on cloned board
                new_piece = next((p for p in new_board.get_player_pieces(current_player)
                                  if type(p) == type(piece) and p.position == piece.position), None)
                if not new_piece:
                    continue
                new_move = next((m for m in new_piece.get_move_options()
                                 if hasattr(m, 'position') and m.position == move.position), None)
                if not new_move:
                    continue

                # Execute move
                new_piece.move(new_move)

                # ✅ Check if this move caused checkmate
                result = get_result(new_board)
                if result is not None and "win" in result.lower() and player_name in result.lower():
                    return 999999  # instant winning move

                # Switch current player
                new_board.current_player = [p for p in new_board.players if p != current_player][0]

                # Opponent's turn
                score = minimax(new_board, depth - 1, alpha, beta, player_name, is_max_turn=False)
                value = max(value, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            except Exception:
                continue
        return value

    else:
        value = float('inf')
        for piece, move in moves:
            new_board = board.clone()
            try:
                new_piece = next((p for p in new_board.get_player_pieces(current_player)
                                  if type(p) == type(piece) and p.position == piece.position), None)
                if not new_piece:
                    continue
                new_move = next((m for m in new_piece.get_move_options()
                                 if hasattr(m, 'position') and m.position == move.position), None)
                if not new_move:
                    continue

                new_piece.move(new_move)

                # ✅ Check if this move caused checkmate (opponent winning)
                result = get_result(new_board)
                if result is not None and "win" in result.lower() and player_name not in result.lower():
                    return -999999  # opponent wins — terrible for us

                # Switch current player
                new_board.current_player = [p for p in new_board.players if p != current_player][0]

                # Agent's turn again
                score = minimax(new_board, depth - 1, alpha, beta, player_name, is_max_turn=True)
                value = min(value, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            except Exception:
                continue
        return value


# ============================================================================
# BEST MOVE FUNCTION
# ============================================================================

def find_best_move(board, player):                               
    """
    Evaluates all legal moves at depth 2 using minimax with alpha-beta pruning.
    """
    legal_moves = list_legal_moves_for(board, player)            
    if not legal_moves:                                         
        return None, None                                        
    if len(legal_moves) == 1:                                    
        return legal_moves[0]                                   

    best_move = None                                             # Will store the best (piece, move) found so far.
    best_score = float('-inf')                                   # Best score seen so far (we maximize this).
    alpha, beta = float('-inf'), float('inf')                    # Initialize alpha-beta bounds (alpha = lower bound, beta = upper bound).

    ordered_moves = order_moves(board, legal_moves)              # Reorder moves (e.g., checkmates/captures first) to improve pruning.

    for piece, move in ordered_moves:                            # Loop through each candidate move in the chosen order...
        new_board = board.clone()                                # Work on a fresh copy so we don't mutate the real game state.
        try:
            new_piece = next(                                    # On the cloned board, find the "same" piece object as `piece`...
                (p for p in new_board.get_player_pieces(player)  
                 if type(p) == type(piece)                       
                 and p.position == piece.position),              
                None                                             
            )
            if not new_piece:                                    
                continue                                         

            new_move = next(                                     # On that cloned piece, find the "same" move as `move`...
                (m for m in new_piece.get_move_options()         # ...by scanning its legal options on the clone...
                 if hasattr(m, 'position')                       # ...making sure the move has a destination...
                 and m.position == move.position),               # ...and matching the destination square.
                None                                             # If none found, return None.
            )
            if not new_move:                                     
                continue                                         

            # Apply move
            new_piece.move(new_move)                             # Execute the move on the cloned board (updates board state).

            # Switch turn
            new_board.current_player = [p for p in new_board.players if p != player][0]
                                                                   # Hand the turn to the opponent on the cloned board.

            # Search exactly one more level (opponent’s response)
            score = minimax(                                      # Evaluate the position assuming opponent now plays best.
                new_board,                                        # The cloned, updated board after our candidate move.
                1,                                                # Depth = 1: search exactly one opponent reply (total depth 2).
                alpha,                                            # Current alpha bound (best guaranteed score for us so far).
                beta,                                             # Current beta bound (opponent's bound).
                player.name,                                      # Evaluate from our (root) player's perspective.
                is_max_turn=False                                 # It’s opponent’s turn now → minimizing side.
            )

            if score > best_score:                                # If this candidate line scores higher than any seen before...
                best_score = score                                
                best_move = (piece, move)                         

            alpha = max(alpha, score)                             # Tighten alpha (our lower bound) since we found a better line.
            if beta <= alpha:                                     # Alpha-beta cutoff: opponent can force us away from worse lines.
                break                                             # Stop exploring further moves — they can't beat current best.

        except Exception:                                         # If anything goes wrong skip this candidate and keep going.
            continue                                              
    if best_move:                                                 # If we found at least one valid, evaluated move...
        return best_move                                          # ...return the best one.
    else:                                                         # Otherwise (very rare; e.g., all mapping failed)...
        # Fallback: random legal move
        return random.choice(legal_moves)                         # ...return a random legal move to stay robust.


# ============================================================================
# AGENT ENTRY POINT
# ============================================================================

def agent(board, player, var):
    """
    Called by COMP2321 environment.
    Always searches depth = 2 using alpha-beta pruning.
    """
    piece, move = find_best_move(board, player)
    if piece is None or move is None:
        legal = list_legal_moves_for(board, player)
        if legal:
            piece, move = random.choice(legal)
    return piece, move
