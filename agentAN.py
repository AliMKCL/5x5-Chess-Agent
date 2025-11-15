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
# ENDGAME EVALUATION FUNCTIONS (for agentE.py)
# ============================================================================

def classify_endgame_type(board, player_name):
    """
    Classifies the current position into specific endgame categories.

    Args:
        board: Current board state
        player_name: Name of the player to evaluate for

    Returns:
        str: Endgame type classification:
            - 'none': Middlegame (>8 pieces)
            - 'pawn_race': Pawn endgame → triggers opposition + promotion race
            - 'mating_attack': Queen/Right vs King → triggers mobility restriction
            - 'minor_piece_endgame': Only Knights/Bishops → triggers king activity
            - 'complex_endgame': 5-8 pieces → general improvements
    """
    from chessmaker.chess.pieces import King, Queen, Bishop, Knight, Pawn
    from extension.piece_right import Right

    # board.get_pieces() returns a generator; convert once to list for reuse
    all_pieces = list(board.get_pieces())
    total_pieces = len(all_pieces)

    # Middlegame: more than 8 pieces
    if total_pieces > 8:
        return 'none'

    # Get player and opponent
    player = next(p for p in board.players if p.name == player_name)
    opponent = next(p for p in board.players if p.name != player_name)

    # Categorize pieces by type and player
    player_pieces = {'pawns': [], 'queens': [], 'rights': [], 'knights': [], 'bishops': [], 'king': None}
    opponent_pieces = {'pawns': [], 'queens': [], 'rights': [], 'knights': [], 'bishops': [], 'king': None}

    for piece in all_pieces:
        piece_name = piece.name.lower()
        is_player = piece.player == player

        if piece_name == 'pawn':
            if is_player:
                player_pieces['pawns'].append(piece)
            else:
                opponent_pieces['pawns'].append(piece)
        elif piece_name == 'queen':
            if is_player:
                player_pieces['queens'].append(piece)
            else:
                opponent_pieces['queens'].append(piece)
        elif piece_name == 'right':
            if is_player:
                player_pieces['rights'].append(piece)
            else:
                opponent_pieces['rights'].append(piece)
        elif piece_name == 'knight':
            if is_player:
                player_pieces['knights'].append(piece)
            else:
                opponent_pieces['knights'].append(piece)
        elif piece_name == 'bishop':
            if is_player:
                player_pieces['bishops'].append(piece)
            else:
                opponent_pieces['bishops'].append(piece)
        elif piece_name == 'king':
            if is_player:
                player_pieces['king'] = piece
            else:
                opponent_pieces['king'] = piece

    # Count non-king pieces for each player
    player_non_king = len(player_pieces['pawns']) + len(player_pieces['queens']) + len(player_pieces['rights']) + len(player_pieces['knights']) + len(player_pieces['bishops'])
    opponent_non_king = len(opponent_pieces['pawns']) + len(opponent_pieces['queens']) + len(opponent_pieces['rights']) + len(opponent_pieces['knights']) + len(opponent_pieces['bishops'])

    # MATING ATTACK: We have Queen/Right, opponent has only King (or King + minor piece)
    if (len(player_pieces['queens']) >= 1 or len(player_pieces['rights']) >= 1) and opponent_non_king <= 1:
        # Opponent has at most 1 minor piece (knight or bishop)
        if opponent_non_king == 0 or (len(opponent_pieces['knights']) + len(opponent_pieces['bishops']) == opponent_non_king):
            return 'mating_attack'

    # Opponent has Queen/Right, we have only King (or King + minor) - also mating attack
    if (len(opponent_pieces['queens']) >= 1 or len(opponent_pieces['rights']) >= 1) and player_non_king <= 1:
        if player_non_king == 0 or (len(player_pieces['knights']) + len(player_pieces['bishops']) == player_non_king):
            return 'mating_attack'

    # PAWN RACE: At least one pawn exists, no major pieces (queens/rights)
    total_pawns = len(player_pieces['pawns']) + len(opponent_pieces['pawns'])
    total_majors = len(player_pieces['queens']) + len(opponent_pieces['queens']) + len(player_pieces['rights']) + len(opponent_pieces['rights'])

    if total_pawns >= 1 and total_majors == 0:
        return 'pawn_race'

    # MINOR PIECE ENDGAME: Only knights and/or bishops remain (no pawns, no majors)
    if total_pawns == 0 and total_majors == 0:
        return 'minor_piece_endgame'

    # COMPLEX ENDGAME: 5-8 pieces total, doesn't fit other categories
    return 'complex_endgame'


def count_mobility(board, player):
    """
    Counts the number of legal moves available to a player.

    This is critical for mobility restriction strategy in endgames.

    Args:
        board: Current board state
        player: The player to count moves for

    Returns:
        int: Number of legal moves available
    """
    from extension.board_utils import list_legal_moves_for

    legal_moves = list_legal_moves_for(board, player)
    return len(legal_moves)


def evaluate_king_opposition(board, player_name):
    """
    Evaluates king opposition in pawn endgames.

    Opposition is a critical concept where kings face each other with odd squares between.
    Having the opposition determines who wins in many pawn endgames.

    Args:
        board: Current board state
        player_name: Name of the player to evaluate for

    Returns:
        int: Bonus (+) if we have opposition, penalty (-) if opponent has it, 0 otherwise
    """
    from chessmaker.chess.pieces import King

    all_pieces = board.get_pieces()
    player = next(p for p in board.players if p.name == player_name)
    opponent = next(p for p in board.players if p.name != player_name)

    # Find both kings
    player_king = None
    opponent_king = None

    for piece in all_pieces:
        if piece.name.lower() == 'king':
            if piece.player == player:
                player_king = piece
            else:
                opponent_king = piece

    if not player_king or not opponent_king:
        return 0

    # Calculate distance between kings
    dx = abs(player_king.position.x - opponent_king.position.x)
    dy = abs(player_king.position.y - opponent_king.position.y)

    # Direct opposition: kings on same file/rank with exactly 2 squares apart (1 square between)
    # Or diagonal opposition with 2 squares distance
    has_direct_opposition = False

    if (dx == 0 and dy == 2) or (dy == 0 and dx == 2):
        # Vertical or horizontal opposition
        has_direct_opposition = True
    elif dx == 2 and dy == 2:
        # Diagonal opposition
        has_direct_opposition = True

    # Distant opposition: same file/rank, even number of squares apart (>2)
    has_distant_opposition = False

    if dx == 0 and dy > 2 and dy % 2 == 0:
        has_distant_opposition = True
    elif dy == 0 and dx > 2 and dx % 2 == 0:
        has_distant_opposition = True

    # Determine whose turn it is (who has the opposition)
    # If it's our turn and we have direct/distant opposition, that's good
    current_player_name = board.current_player.name

    if has_direct_opposition:
        if current_player_name == opponent.name:
            # It's opponent's turn, so WE have the opposition (good for us)
            return 100
        else:
            # It's our turn, opponent has the opposition (bad for us)
            return -100

    if has_distant_opposition:
        if current_player_name == opponent.name:
            # We have distant opposition
            return 50
        else:
            # Opponent has distant opposition
            return -50

    return 0


def evaluate_pawn_promotion_race(board, player_name):
    """
    Evaluates pawn promotion races with proper king protection analysis.

    Key improvements:
    1. Checks if opponent king can capture the pawn at its CURRENT position
    2. Only rewards pawn advancement if:
       - Opponent king cannot catch it before promotion, OR
       - Our king is close enough (≤1 square) to protect it
    3. Penalizes exposed pawns heavily

    Args:
        board: Current board state
        player_name: Name of the player to evaluate for

    Returns:
        int: Bonus for favorable pawn races, penalty for unfavorable ones
    """
    from chessmaker.chess.pieces import King, Pawn

    all_pieces = board.get_pieces()
    player = next(p for p in board.players if p.name == player_name)
    opponent = next(p for p in board.players if p.name != player_name)

    # Find kings and pawns
    player_king = None
    opponent_king = None
    player_pawns = []
    opponent_pawns = []

    for piece in all_pieces:
        piece_name = piece.name.lower()
        if piece_name == 'king':
            if piece.player == player:
                player_king = piece
            else:
                opponent_king = piece
        elif piece_name == 'pawn':
            if piece.player == player:
                player_pawns.append(piece)
            else:
                opponent_pawns.append(piece)

    if not player_king or not opponent_king:
        return 0

    score = 0

    # Evaluate each of our pawns
    for pawn in player_pawns:
        # Calculate promotion square (row 0 for white, row 4 for black)
        promo_y = 0 if player.name == "white" else 4
        promo_x = pawn.position.x

        # Distance from pawn to promotion (pawns move in straight line)
        pawn_to_promo = abs(pawn.position.y - promo_y)

        # FIXED: Use Chebyshev distance (king moves) = max(dx, dy)
        # Kings can move diagonally, so actual distance is the maximum of x and y differences

        # Distance from our king to the pawn's CURRENT position
        our_king_to_pawn = max(abs(player_king.position.x - pawn.position.x),
                                abs(player_king.position.y - pawn.position.y))

        # Distance from opponent king to the pawn's CURRENT position
        opp_king_to_pawn = max(abs(opponent_king.position.x - pawn.position.x),
                                abs(opponent_king.position.y - pawn.position.y))

        # Distance from opponent king to promotion square
        opp_king_to_promo = max(abs(opponent_king.position.x - promo_x),
                                 abs(opponent_king.position.y - promo_y))

        # === CRITICAL CHECK 1: Can opponent capture the pawn RIGHT NOW? ===
        # If opponent king is adjacent (distance 1), pawn is under immediate threat
        if opp_king_to_pawn == 1:
            # Pawn is next to opponent king - DANGER!
            # Only safe if our king is also adjacent (protecting it)
            if our_king_to_pawn == 1:
                # Our king protects the pawn - neutral position
                score += 1000
            else:
                # Pawn is exposed and will be captured!
                score -= 800  # Massive penalty for exposed pawn
                continue  # Skip further evaluation - this pawn is lost

        # === CRITICAL CHECK 2: Rule of the Square ===
        # Can the opponent king catch the pawn before it promotes?
        # Opponent needs to reach promotion square in <= pawn_to_promo moves
        can_opponent_catch = opp_king_to_promo <= pawn_to_promo

        if not can_opponent_catch:
            # Opponent CANNOT catch the pawn - it's unstoppable!
            score += 500  # Huge bonus for unstoppable pawn
            # Still give bonus for being close to promotion
            score += (4 - pawn_to_promo) * 50
        else:
            # Opponent CAN catch the pawn - need our king's protection
            # === CONDITION: Our king must be close (≤1 square) to protect ===
            if our_king_to_pawn <= 1:
                # Our king is adjacent - pawn is protected
                score += 200
                # Bonus for advancement if protected
                score += (4 - pawn_to_promo) * 30
            elif our_king_to_pawn == 2:
                # Our king is 2 squares away - marginal protection
                score += 50
                # Smaller bonus for advancement
                score += (4 - pawn_to_promo) * 10
            else:
                # Our king is too far - pawn is unsupported and catchable
                score -= 400  # Big penalty
                # Don't give bonus for advancement - this pawn is in danger

    # Evaluate opponent's pawns (penalty if they're dangerous)
    for pawn in opponent_pawns:
        promo_y = 0 if opponent.name == "white" else 4
        promo_x = pawn.position.x

        pawn_to_promo = abs(pawn.position.y - promo_y)

        # FIXED: Use Chebyshev distance for king moves
        # Distance from our king to opponent's pawn
        our_king_to_pawn = max(abs(player_king.position.x - pawn.position.x),
                                abs(player_king.position.y - pawn.position.y))

        # Distance from opponent king to their pawn
        opp_king_to_pawn = max(abs(opponent_king.position.x - pawn.position.x),
                                abs(opponent_king.position.y - pawn.position.y))

        # Distance from our king to opponent's promotion square
        our_king_to_promo = max(abs(player_king.position.x - promo_x),
                                 abs(player_king.position.y - promo_y))

        # Check if we can capture opponent's pawn immediately
        if our_king_to_pawn == 1:
            # We're adjacent to opponent's pawn
            if opp_king_to_pawn == 1:
                # Their king protects it
                score -= 0
            else:
                # We can capture it next move!
                score += 600  # Bonus for capturable opponent pawn

        # Rule of the square for opponent pawn
        can_we_catch = our_king_to_promo <= pawn_to_promo

        if not can_we_catch:
            # We CANNOT catch their pawn - it's unstoppable!
            score -= 500  # Huge penalty
            score -= (4 - pawn_to_promo) * 50
        else:
            # We CAN catch their pawn
            if opp_king_to_pawn <= 1:
                # Their king protects it - dangerous
                score -= 200
                score -= (4 - pawn_to_promo) * 30
            elif opp_king_to_pawn == 2:
                # Marginal support
                score -= 50
                score -= (4 - pawn_to_promo) * 10
            else:
                # Unsupported opponent pawn - we can catch it
                score += 300

    return score


def evaluate_mating_net(board, player_name):
    """
    Evaluates mating attack positions (Queen/Right vs King).

    Implements THREE key strategies:
    1. MOBILITY RESTRICTION: Reduce opponent king moves to 2-4 (avoid stalemate at 0-1)
    2. EDGE DRIVE: Force opponent king to board edge
    3. KING COOPERATION: Our king should be 2-3 squares away to support

    This is the KEY INNOVATION suggested by the user.

    Args:
        board: Current board state
        player_name: Name of the player to evaluate for

    Returns:
        int: Bonus for good mating position, penalty for stalemate risk
    """
    from chessmaker.chess.pieces import King, Queen
    from extension.piece_right import Right

    all_pieces = board.get_pieces()
    player = next(p for p in board.players if p.name == player_name)
    opponent = next(p for p in board.players if p.name != player_name)

    # Find pieces
    player_king = None
    opponent_king = None
    has_mating_piece = False

    for piece in all_pieces:
        piece_name = piece.name.lower()
        if piece_name == 'king':
            if piece.player == player:
                player_king = piece
            else:
                opponent_king = piece
        elif piece_name in ['queen', 'right']:
            if piece.player == player:
                has_mating_piece = True

    # Only apply if we have a mating piece (Queen or Right)
    if not has_mating_piece or not opponent_king or not player_king:
        return 0

    score = 0

    # === 1. MOBILITY RESTRICTION ===
    # Count opponent's legal moves
    opponent_mobility = count_mobility(board, opponent)

    if opponent_mobility == 0:
        # STALEMATE! This is very bad
        score -= 10000
    elif opponent_mobility == 1:
        # Risky - very close to stalemate
        score -= 5000
    elif 2 <= opponent_mobility <= 4:
        # PERFECT restriction! Opponent has limited options but not stalemate
        score += 300
    else:
        # Too much freedom - need to restrict more
        score -= opponent_mobility * 20

    # === 2. EDGE DRIVE ===
    # Calculate opponent king's distance to nearest edge
    edge_dist = min(
        opponent_king.position.x,           # Distance to left edge
        4 - opponent_king.position.x,       # Distance to right edge
        opponent_king.position.y,           # Distance to bottom edge
        4 - opponent_king.position.y        # Distance to top edge
    )

    # Bonus for king closer to edge (negative distance = bonus)
    # When edge_dist = 0 (king on edge), we get maximum bonus
    # When edge_dist = 2 (king in center), we get penalty
    score += (2 - edge_dist) * 100

    # === 3. KING COOPERATION ===
    # Our king should be 2-3 squares away (close enough to help, not blocking)
    # FIXED: Use Chebyshev distance for king moves
    king_distance = max(abs(player_king.position.x - opponent_king.position.x),
                        abs(player_king.position.y - opponent_king.position.y))

    if king_distance == 2 or king_distance == 3:
        # Ideal cooperation distance
        score += 150
    elif king_distance == 1:
        # Too close - might be blocking our queen/right
        score -= 50
    elif king_distance >= 4:
        # Too far - not helping
        score -= (king_distance - 3) * 30

    return score


def evaluate_passed_pawns(board, player_name):
    """
    Identifies and evaluates passed pawns.

    A passed pawn is one with no opposing pawns:
    - On the same file
    - On adjacent files

    Passed pawns are extremely dangerous in endgames.

    Args:
        board: Current board state
        player_name: Name of the player to evaluate for

    Returns:
        int: Bonus for passed pawns
    """
    from chessmaker.chess.pieces import Pawn

    all_pieces = board.get_pieces()
    player = next(p for p in board.players if p.name == player_name)
    opponent = next(p for p in board.players if p.name != player_name)

    # Collect pawns
    player_pawns = []
    opponent_pawns = []

    for piece in all_pieces:
        if piece.name.lower() == 'pawn':
            if piece.player == player:
                player_pawns.append(piece)
            else:
                opponent_pawns.append(piece)

    score = 0

    # Check each of our pawns
    for pawn in player_pawns:
        is_passed = True
        pawn_file = pawn.position.x
        pawn_rank = pawn.position.y

        # Direction of advancement (white: toward row 0, black: toward row 4)
        if player.name == "white":
            # Check if any opponent pawns block this pawn's path to row 0
            for opp_pawn in opponent_pawns:
                opp_file = opp_pawn.position.x
                opp_rank = opp_pawn.position.y

                # Check if opponent pawn is on same file or adjacent file
                if abs(opp_file - pawn_file) <= 1:
                    # Check if opponent pawn is ahead of our pawn (lower rank for white)
                    if opp_rank < pawn_rank:
                        is_passed = False
                        break
        else:
            # Black pawn - advancing toward row 4
            for opp_pawn in opponent_pawns:
                opp_file = opp_pawn.position.x
                opp_rank = opp_pawn.position.y

                if abs(opp_file - pawn_file) <= 1:
                    if opp_rank > pawn_rank:
                        is_passed = False
                        break

        if is_passed:
            # Bonus for passed pawn (higher bonus if closer to promotion)
            distance_to_promotion = abs(pawn_rank - (0 if player.name == "white" else 4))
            bonus = 200 - (distance_to_promotion * 30)
            score += bonus

    # Penalty for opponent's passed pawns
    for pawn in opponent_pawns:
        is_passed = True
        pawn_file = pawn.position.x
        pawn_rank = pawn.position.y

        if opponent.name == "white":
            for our_pawn in player_pawns:
                our_file = our_pawn.position.x
                our_rank = our_pawn.position.y

                if abs(our_file - pawn_file) <= 1:
                    if our_rank < pawn_rank:
                        is_passed = False
                        break
        else:
            for our_pawn in player_pawns:
                our_file = our_pawn.position.x
                our_rank = our_pawn.position.y

                if abs(our_file - pawn_file) <= 1:
                    if our_rank > pawn_rank:
                        is_passed = False
                        break

        if is_passed:
            distance_to_promotion = abs(pawn_rank - (0 if opponent.name == "white" else 4))
            penalty = 200 - (distance_to_promotion * 30)
            score -= penalty

    return score


def evaluate_key_squares(board, player_name):
    """
    Evaluates control of key squares in pawn endgames.

    In pawn endgames, certain squares in front of pawns (especially on the
    promotion path) are critical. Controlling them with the king determines victory.

    Args:
        board: Current board state
        player_name: Name of the player to evaluate for

    Returns:
        int: Bonus for controlling key squares
    """
    from chessmaker.chess.pieces import King, Pawn

    all_pieces = board.get_pieces()
    player = next(p for p in board.players if p.name == player_name)
    opponent = next(p for p in board.players if p.name != player_name)

    # Find kings and pawns
    player_king = None
    player_pawns = []

    for piece in all_pieces:
        piece_name = piece.name.lower()
        if piece_name == 'king' and piece.player == player:
            player_king = piece
        elif piece_name == 'pawn' and piece.player == player:
            player_pawns.append(piece)

    if not player_king or not player_pawns:
        return 0

    score = 0

    # For each pawn, identify key squares (promotion square and squares leading to it)
    for pawn in player_pawns:
        promo_y = 0 if player.name == "white" else 4
        promo_x = pawn.position.x

        # Key squares: promotion square and 1-2 squares in front of pawn
        key_squares = []

        # Promotion square
        key_squares.append((promo_x, promo_y))

        # Squares 1-2 ranks in front of pawn (toward promotion)
        if player.name == "white":
            if pawn.position.y - 1 >= 0:
                key_squares.append((promo_x, pawn.position.y - 1))
            if pawn.position.y - 2 >= 0:
                key_squares.append((promo_x, pawn.position.y - 2))
        else:
            if pawn.position.y + 1 <= 4:
                key_squares.append((promo_x, pawn.position.y + 1))
            if pawn.position.y + 2 <= 4:
                key_squares.append((promo_x, pawn.position.y + 2))

        # Check if our king is on or near key squares
        # FIXED: Use Chebyshev distance for king moves
        for kx, ky in key_squares:
            king_dist = max(abs(player_king.position.x - kx), abs(player_king.position.y - ky))
            if king_dist == 0:
                # King is ON a key square
                score += 80
            elif king_dist == 1:
                # King is adjacent to key square
                score += 40

    return score


def evaluate_king_activity(board, player_name):
    """
    Evaluates general king activity/centralization in endgames.

    In endgames, the king becomes a fighting piece and should be centralized.

    Args:
        board: Current board state
        player_name: Name of the player to evaluate for

    Returns:
        int: Bonus for active king positioning
    """
    from chessmaker.chess.pieces import King

    all_pieces = board.get_pieces()
    player = next(p for p in board.players if p.name == player_name)

    # Find our king
    player_king = None
    for piece in all_pieces:
        if piece.name.lower() == 'king' and piece.player == player:
            player_king = piece
            break

    if not player_king:
        return 0

    # Calculate centralization bonus
    # Center squares (2,2) get highest bonus, edges get penalty
    x, y = player_king.position.x, player_king.position.y

    # Distance from center (2, 2)
    center_dist = abs(x - 2) + abs(y - 2)

    # Bonus decreases with distance from center
    centralization_bonus = (4 - center_dist) * 25

    return centralization_bonus


def evaluate_mobility_advantage(board, player_name):
    """
    Evaluates mobility advantage (difference in legal moves).

    Having more legal moves than the opponent is a positional advantage in endgames.

    Args:
        board: Current board state
        player_name: Name of the player to evaluate for

    Returns:
        int: Bonus based on mobility difference
    """
    player = next(p for p in board.players if p.name == player_name)
    opponent = next(p for p in board.players if p.name != player_name)

    our_mobility = count_mobility(board, player)
    opp_mobility = count_mobility(board, opponent)

    mobility_diff = our_mobility - opp_mobility

    # Each extra move is worth a small bonus
    return mobility_diff * 10

# ============================================================================
# CONSTANTS & HEURISTICS
# ============================================================================

# Global file handle for logging
LOG_FILE = None
MOVE_COUNTER = 0

# Quiescence search configuration
QUIESCENCE_ENABLED = True  # Toggle quiescence search on/off
MAX_QUIESCENCE_DEPTH = 7  # Maximum quiescence search depth

def init_log_file():
    """Initialize the log file for this game session."""
    global LOG_FILE
    LOG_FILE = open("game_log.txt", "a")
    LOG_FILE.write("=== GAME LOG (agentE - Endgame Enhanced with Quiescence Search) ===\n\n")

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

def build_move_cache(board):
    """
    Build a cache of move options for all pieces on the board.

    This is CRITICAL-2 optimization: Instead of calling piece.get_move_options()
    multiple times for the same piece in the same board state, we cache all moves
    once and reuse them.

    Time Complexity:
    - WITHOUT cache: O(n²) - n pieces × n calls per piece = n²
    - WITH cache: O(n) - one call per piece

    Returns:
        dict: {(piece.position.x, piece.position.y, piece.name, piece.player.name): [moves]}
    """
    cache = {}
    for piece in board.get_pieces():
        # Create a unique key for this piece (position + type + player)
        key = (piece.position.x, piece.position.y, piece.name, piece.player.name)
        cache[key] = piece.get_move_options()
    return cache

def create_position_map(board):
    """
    Create a position-to-piece lookup table for O(1) piece lookups by coordinates.

    This is CRITICAL-3 optimization: Instead of scanning board.get_pieces() linearly
    (O(n) per lookup), we build a dictionary mapping (x, y) -> piece (O(1) lookups).

    Time Complexity:
    - Building the map: O(n) where n = number of pieces
    - Lookups: O(1) instead of O(n)

    Args:
        board: The board to build the position map for

    Returns:
        dict: {(x, y): piece} mapping coordinates to piece objects
    """
    pos_map = {}
    for piece in board.get_pieces():
        pos_map[(piece.position.x, piece.position.y)] = piece
    return pos_map

def get_cached_moves(piece, cache):
    """
    Retrieve cached move options for a piece.

    Args:
        piece: The piece to get moves for
        cache: The move cache dictionary

    Returns:
        list: Cached move options, or fresh moves if not cached
    """
    key = (piece.position.x, piece.position.y, piece.name, piece.player.name)
    if key in cache:
        return cache[key]
    # Fallback: if somehow not in cache, compute fresh
    print("Warning: Piece moves not found in cache, computing fresh.")
    return piece.get_move_options()


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

# Estimates how good or bad a board position is for the given player.
def evaluate_board(board, player_name, pos_map=None):
    """
    ENDGAME-ENHANCED EVALUATION FUNCTION (agentE.py)

    Evaluates the board state based on:
      1) Material balance
      2) Positional values from piece-square tables (endgame-aware)
      3) King safety (bonus if ≥ 2 friendly pieces within 1-cell radius)
      4) ENDGAME-SPECIFIC EVALUATION (Phase B):
         - Pawn race: Opposition, promotion race, passed pawns, key squares
         - Mating attack: Mobility restriction, edge drive, king cooperation
         - Minor piece endgame: King activity
         - Complex endgame: Mobility advantage + king activity

    Args:
        board: Current board state
        player_name: Name of the player to evaluate for
        pos_map: Optional position-to-piece lookup dict for O(1) piece lookups

    Returns:
        Positive score → good for 'player_name'
        Negative score → good for opponent
    """
    score = 0

    # Get lists of all pieces and players
    # board.get_pieces() returns a generator; convert to list for reuse
    all_pieces = list(board.get_pieces())
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

        # 1. Material + Positional Value (using endgame-aware tables)
        base_value = get_piece_value(piece)
        is_white = piece.player.name == "white"
        pos_value = get_positional_value(piece, is_white, board, use_endgame_tables=True)
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

    # --- 3. KING SAFETY (reduced importance in endgames) -------------------
    # Calculate king safety for both kings (less relevant in endgames)
    total_pieces = len(all_pieces)

    if total_pieces > 8:  # Only apply in middlegame
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

    # --- 4. ENDGAME-SPECIFIC EVALUATION (NEW IN PHASE B) -------------------
    # Classify endgame type and apply specialized evaluation
    endgame_type = classify_endgame_type(board, player_name)

    if endgame_type == 'pawn_race':
        # Pawn endgame bonuses
        #log_message("PAWN RACE")
        # print("PAWN RACE")
        score += evaluate_king_opposition(board, player_name)
        score += evaluate_pawn_promotion_race(board, player_name)
        score += evaluate_passed_pawns(board, player_name)
        score += evaluate_key_squares(board, player_name)

    elif endgame_type == 'mating_attack':
        # print("MATING ATTACK")
        # Mating attack bonuses (mobility restriction + edge drive + cooperation)
        score += evaluate_mating_net(board, player_name)

    elif endgame_type == 'minor_piece_endgame':
        # Minor piece endgame - king activity is crucial
        score += evaluate_king_activity(board, player_name)
        score += evaluate_mobility_advantage(board, player_name)

    elif endgame_type == 'complex_endgame':
        # General endgame improvements
        score += evaluate_king_activity(board, player_name)
        score += evaluate_mobility_advantage(board, player_name)
        # Also check for passed pawns if any exist
        score += evaluate_passed_pawns(board, player_name)

    return score

# ============================================================================
# MOVE ORDERING
# ============================================================================

def order_moves(board, moves, move_cache=None, pos_map=None):
    """
    Orders moves by tactical and positional importance.

    Priority:
      1. Checkmate moves
      2. Valuable or safe captures (victim >= attacker OR target is undefended)
      3. Positional improvement (piece-square tables)

    Args:
        board: Current board state
        moves: List of (piece, move) tuples to order
        move_cache: Optional cache of piece move options to avoid redundant calculations
        pos_map: Optional position-to-piece lookup dict for O(1) piece lookups
    """
    scored_moves = []

    # Build move cache if not provided (for backwards compatibility)
    if move_cache is None:
        move_cache = build_move_cache(board)

    # CRITICAL-3 OPTIMIZATION: Create position-to-piece lookup for O(1) access
    if pos_map is None:
        pos_map = create_position_map(board)

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
            #log_message(f"Move ordering: Checkmate move detected - {piece.name} to ({move.position.x},{move.position.y})")

        #=====  2 Valuable captures ===============================================
        # Prioritize high-value captures (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)

        if hasattr(move, "captures") and move.captures:
            for capture_pos in move.captures:
                # CRITICAL-3 OPTIMIZATION: O(1) lookup instead of O(n) scan
                target = pos_map.get((capture_pos.x, capture_pos.y))
                if target:
                    victim_value = get_piece_value(target)

                    # Get opponent player for exchange evaluation
                    opponent = next(p for p in board.players if p != piece.player)

                    # FIX: Use capture_pos (where victim is), not move.position (where attacker moves to)
                    # In most cases these are the same, but for clarity use the actual capture location
                    num_diff, val_diff = attacker_defender_ratio(board, capture_pos, opponent, piece.player, move_cache, pos_map)

                    # Base MVV-LVA score: prefer low-value attackers capturing high-value victims
                    # Using (victim * 10) ensures victim value is prioritized
                    base_mvv_lva = (victim_value * 10) - attacker_value

                    # Case 1: More or equal defenders than attackers
                    if not val_diff or val_diff < 0:
                        score += base_mvv_lva    # Check if weakest attacker < victim_value
                        ##log_message(f"Equal/losing numbers but potentially favorable: {piece.name} captures {target.name}, num_diff={num_diff}, score = {score} move {piece.name} to ({move.position.x},{move.position.y})")

                    # Case 2: More attackers than defenders, and positive trade
                    elif val_diff > 0:
                        score += base_mvv_lva + 1000    # Capture with the weakest attacker and prefer the capture
                        ##log_message(f"Winning exchange: {piece.name} captures {target.name}, net={val_diff}, score={score}, move {piece.name} to ({move.position.x},{move.position.y})")



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
# QUIESCENCE SEARCH (PHASE 1)
# ============================================================================

def quiescence_search(board, alpha, beta, player_name, time_limit, start_time, is_max_turn=True, q_depth=0):
    """
    Quiescence search: extend search for tactical moves only.
    Prevents horizon effect by resolving capture sequences to quiet positions.
    # ADD CHECKS as well

    Uses STANDARD MINIMAX (not negamax) for clarity and correctness.
    - is_max_turn=True: Maximizing player (agent) tries to maximize score
    - is_max_turn=False: Minimizing player (opponent) tries to minimize score
    - Evaluation always from player_name's perspective (positive = good for player_name)

    Based on game_log.txt analysis:
    - Quiet positions: scores typically in range [-100, +100]
    - Captures can swing: [-500, +500]
    - We search until no more captures improve the position

    Args:
        board: Current board state
        alpha, beta: Alpha-beta bounds
        player_name: The agent's perspective (fixed throughout search tree)
        time_limit: Max time allowed
        start_time: When search started
        is_max_turn: True if current turn belongs to the agent (maximizing)
        q_depth: Current quiescence depth (for limiting)

    Returns:
        Evaluation score (int) from player_name's perspective
    """
    global nodes_explored
    nodes_explored += 1

    # Check timeout
    if time.time() - start_time >= time_limit:
        eval_score = evaluate_board(board, player_name)
        indent = "  " * (q_depth + 2)
        #log_message(f"{indent}[Q-depth {q_depth}] TIMEOUT - returning static eval={eval_score}")
        return eval_score

    # Limit quiescence depth to prevent infinite loops
    if q_depth >= MAX_QUIESCENCE_DEPTH:
        eval_score = evaluate_board(board, player_name)
        indent = "  " * (q_depth + 2)
        #log_message(f"{indent}[Q-depth {q_depth}] MAX Q-DEPTH REACHED - returning static eval={eval_score}")
        return eval_score

    indent = "  " * (q_depth + 2)

    # Check for terminal game states
    result = get_result(board)
    if result is not None:
        res = result.lower()
        if "checkmate" in res:
            # Return high/low scores based on who won
            if player_name in res or (player_name == "white" and "black loses" in res) or (player_name == "black" and "white loses" in res):
                #log_message(f"{indent}[Q-depth {q_depth}] CHECKMATE (we win) - returning +999999")
                return 999999  # We win
            else:
                #log_message(f"{indent}[Q-depth {q_depth}] CHECKMATE (we lose) - returning -999999")
                return -999999  # We lose
        elif "draw" in res or "stalemate" in res:
            #log_message(f"{indent}[Q-depth {q_depth}] DRAW/STALEMATE - returning 0")
            return 0

    # Stand-pat score: evaluate current position without any moves
    # This is the "quiet" baseline - if we don't make a capture, this is what we get
    stand_pat = evaluate_board(board, player_name)
    turn_type = "MAX" if is_max_turn else "MIN"
    #log_message(f"{indent}[Q-depth {q_depth}, {turn_type}] Stand-pat evaluation = {stand_pat} (alpha={alpha}, beta={beta})")

    # STANDARD MINIMAX LOGIC:
    # - MAX turn: if stand_pat >= beta, this position is too good, MIN player won't allow it (beta cutoff)
    #            if stand_pat > alpha, update our guaranteed minimum (alpha)
    # - MIN turn: if stand_pat <= alpha, this position is too bad, MAX player won't allow it (alpha cutoff)
    #            if stand_pat < beta, update opponent's guaranteed maximum (beta)

    if is_max_turn:
        # Maximizing player
        if stand_pat >= beta:
            #log_message(f"{indent}[Q-depth {q_depth}] BETA CUTOFF (stand-pat {stand_pat} >= beta {beta}) - pruning")
            return beta
        if stand_pat > alpha:
            original_alpha = alpha
            alpha = stand_pat
            #log_message(f"{indent}[Q-depth {q_depth}] Stand-pat improves alpha: {original_alpha} -> {alpha}")
    else:
        # Minimizing player
        if stand_pat <= alpha:
            #log_message(f"{indent}[Q-depth {q_depth}] ALPHA CUTOFF (stand-pat {stand_pat} <= alpha {alpha}) - pruning")
            return alpha
        if stand_pat < beta:
            original_beta = beta
            beta = stand_pat
            #log_message(f"{indent}[Q-depth {q_depth}] Stand-pat improves beta: {original_beta} -> {beta}")

    # Get all legal moves
    current_player = board.current_player
    all_moves = list_legal_moves_for(board, current_player)

    if not all_moves:
        #log_message(f"{indent}[Q-depth {q_depth}] No legal moves available - returning stand-pat={stand_pat}")
        return stand_pat

    # Filter for TACTICAL moves only (captures)
    # In quiescence, we only consider forcing moves that might change evaluation
    tactical_moves = []
    for piece, move in all_moves:
        is_capture = hasattr(move, "captures") and move.captures
        if is_capture:
            tactical_moves.append((piece, move))

    # If no tactical moves, position is "quiet" - return stand-pat
    if not tactical_moves:
        #log_message(f"{indent}[Q-depth {q_depth}] No captures available - QUIET POSITION (stand-pat={stand_pat})")
        # print(f"{'  ' * (q_depth + 1)}[Q-depth {q_depth}] Quiet position, eval={stand_pat}")
        return stand_pat

    #log_message(f"{indent}[Q-depth {q_depth}] Found {len(tactical_moves)} captures to analyze from {len(all_moves)} total moves")
    # print(f"{'  ' * (q_depth + 1)}[Q-depth {q_depth}] Analyzing {len(tactical_moves)} captures...")

    # Order tactical moves by MVV-LVA for better pruning
    move_cache = build_move_cache(board)
    pos_map = create_position_map(board)
    ordered_tactical = order_moves(board, tactical_moves, move_cache, pos_map)

    # Initialize best score based on who's turn it is (standard minimax)
    # MAX player starts at -infinity, MIN player starts at +infinity
    best_q_score = stand_pat  # Start with stand-pat as baseline
    moves_tried = 0

    # Search only capture moves
    for piece, move in ordered_tactical:
        if time.time() - start_time >= time_limit:
            #log_message(f"{indent}[Q-depth {q_depth}] Timeout during capture search - returning best so far: {best_q_score}")
            break

        moves_tried += 1
        new_board = board.clone()
        try:
            # Apply move (same as minimax)
            new_piece = next((p for p in new_board.get_player_pieces(current_player)
                             if type(p) == type(piece) and p.position == piece.position), None)
            if not new_piece:
                continue

            cached_moves = get_cached_moves(new_piece, move_cache)
            new_move = next((m for m in cached_moves
                            if hasattr(m, "position") and m.position == move.position), None)
            if not new_move:
                continue

            # Get capture target info for logging
            capture_target = None
            if hasattr(new_move, "captures") and new_move.captures:
                for cap_pos in new_move.captures:
                    target = pos_map.get((cap_pos.x, cap_pos.y))
                    if target:
                        capture_target = target.name
                        break

            new_piece.move(new_move)
            new_board.current_player = [p for p in new_board.players if p != current_player][0]

            #log_message(f"{indent}[Q-depth {q_depth}] Try #{moves_tried}: {piece.name}x{capture_target or '?'} at ({move.position.x},{move.position.y})")

            # STANDARD MINIMAX: Recursive call with flipped is_max_turn
            # NO negation, NO alpha/beta flip - just like regular minimax
            score = quiescence_search(new_board, alpha, beta, player_name, time_limit, start_time, not is_max_turn, q_depth + 1)

            #log_message(f"{indent}[Q-depth {q_depth}]   -> {piece.name}x{capture_target or '?'} returned score={score} (alpha={alpha}, beta={beta})")

            # STANDARD MINIMAX ALPHA-BETA UPDATE (same as main minimax function)
            if is_max_turn:
                # Maximizing player
                best_q_score = max(best_q_score, score)
                if score > alpha:
                    old_alpha = alpha
                    alpha = score
                    #log_message(f"{indent}[Q-depth {q_depth}]   -> NEW BEST! Alpha improved: {old_alpha} -> {alpha}")
                    # print(f"{'  ' * (q_depth + 1)}[Q-depth {q_depth}] New best: {piece.name}x{capture_target}, score={score}")
                if beta <= alpha:
                    #log_message(f"{indent}[Q-depth {q_depth}]   -> BETA CUTOFF! (beta={beta} <= alpha={alpha}) - pruning remaining {len(ordered_tactical) - moves_tried} captures")
                    # print(f"{'  ' * (q_depth + 1)}[Q-depth {q_depth}] Beta cutoff at {piece.name}x{capture_target}, score={score}")
                    break
            else:
                # Minimizing player
                best_q_score = min(best_q_score, score)
                if score < beta:
                    old_beta = beta
                    beta = score
                    #log_message(f"{indent}[Q-depth {q_depth}]   -> NEW BEST! Beta improved: {old_beta} -> {beta}")
                    # print(f"{'  ' * (q_depth + 1)}[Q-depth {q_depth}] New best: {piece.name}x{capture_target}, score={score}")
                if beta <= alpha:
                    #log_message(f"{indent}[Q-depth {q_depth}]   -> ALPHA CUTOFF! (beta={beta} <= alpha={alpha}) - pruning remaining {len(ordered_tactical) - moves_tried} captures")
                    # print(f"{'  ' * (q_depth + 1)}[Q-depth {q_depth}] Alpha cutoff at {piece.name}x{capture_target}, score={score}")
                    break

        except Exception as e:
            #log_message(f"{indent}[Q-depth {q_depth}] Exception during {piece.name} capture: {e}")
            continue

    #log_message(f"{indent}[Q-depth {q_depth}] Finished analyzing {moves_tried}/{len(ordered_tactical)} captures - best score={best_q_score}")
    # print(f"{'  ' * (q_depth + 1)}[Q-depth {q_depth}] Done, best score={best_q_score}")

    # Return the best score found (stand-pat if no captures improved position)
    return best_q_score


# ============================================================================
# SEARCH
# ============================================================================

import time
import random

def find_best_move(board, player, max_depth=10, time_limit=40.0):
    """
    Iterative deepening search using alpha-beta minimax with quiescence search.

    Args:
        board: Current game board object
        player: The player whose move we're finding
        max_depth: Maximum depth to search (default = 10)
        time_limit: Max search time in seconds (default = 40.0)

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

    # CRITICAL-2 OPTIMIZATION: Build move cache ONCE for the root position
    # This avoids redundant get_move_options() calls in order_moves
    root_move_cache = build_move_cache(board)

    # CRITICAL-3 OPTIMIZATION: Build position map ONCE for the root position
    # This avoids redundant O(n) scans for piece lookups
    root_pos_map = create_position_map(board)

    # Iterative deepening loop
    for depth in range(1, max_depth + 1):
        depth_start_time = time.time()
        depth_nodes_before = nodes_explored
        print(f"\n=== Depth {depth} (Quiescence {'ON' if QUIESCENCE_ENABLED else 'OFF'}) ===")

        # Time check before each new depth
        if time.time() - start_time >= time_limit:
            break

        # alpha: Best value that maximizing player can guarantee.
        # beta: Best value that the minimizing player can guarantee.

        alpha, beta = float('-inf'), float('inf')
        current_best_move = None
        current_best_score = float('-inf')

        ordered_moves = order_moves(board, legal_moves, root_move_cache, root_pos_map)

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

                # CRITICAL-2 OPTIMIZATION: Use cached moves instead of calling get_move_options()
                cached_moves = get_cached_moves(new_piece, root_move_cache)
                new_move = next((m for m in cached_moves
                                 if hasattr(m, "position") and m.position == move.position), None)
                if not new_move:
                    continue

                new_piece.move(new_move)
                # print(f"Testing move at depth {depth}: {piece} to ({move.position.x},{move.position.y})")
                #log_message(f"  Testing move at depth {depth}: {piece.name} from ({piece.position.x},{piece.position.y}) to ({move.position.x},{move.position.y})")

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
                        #log_message(f"  *** CHECKMATE FOUND: {piece.name} to ({move.position.x},{move.position.y}) - Instant win! ***")
                        return (piece, move)  # Return immediately, this is the best move possible

                # Check if this move causes a stalemate
                if is_stalemate(new_board):
                    current_eval = evaluate_board(board, player.name)

                    # If we're winning (eval > -200), skip this stalemate move
                    # Threshold: -200 means we need to be clearly losing to accept a draw
                    if current_eval > 0:
                        print(f"  -> Skipping stalemate move (current eval: {current_eval:.0f})")
                        #log_message(f"  -> Skipping stalemate move (current eval: {current_eval:.0f})")
                        continue
                    else:
                        # If we're losing badly (eval < -200), stalemate is acceptable
                        print(f"  -> Accepting stalemate move (current eval: {current_eval:.0f})")
                        #log_message(f"  -> Accepting stalemate move (current eval: {current_eval:.0f})")


                # Run minimax for OPPONENT's reply
                # This runs after our hypothetical move (that we are checking) so it works.
                #log_message(f"  >> Starting minimax search for opponent's reply (depth={depth-1})")
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
                #log_message(f"    -> Minimax returned score: {score} for {player.name}")
                #log_message(f"    -> Current best score: {current_best_score}, Testing if {score} > {current_best_score}")
                # print(f"      Score: {score}")

                # If the current move is better than the previous best at this depth, update new best move.
                if score > current_best_score and score != float('inf') and score != float('-inf'):
                    current_best_score = score
                    current_best_move = (piece, move)
                    #log_message(f"    -> New best move! Score: {score} (player: {player.name}, piece: {piece.name}, move: ({move.position.x},{move.position.y}))")
                    # Debug: Show when we find a very good move (potential checkmate)
                    # Check for inf/nan and treat as non-checkmate
                    if score >= 999999 and score < float('inf'):
                        print(f"  *** FOUND FORCED CHECKMATE: {piece} to ({move.position.x},{move.position.y}) with score {score}")
                        #log_message(f"    *** FOUND FORCED CHECKMATE: {piece.name} to ({move.position.x},{move.position.y}) with score {score}")
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
                #log_message(f"Depth {depth} FULLY COMPLETED ({moves_evaluated}/{total_moves} moves) - Updated best move")
            else:
                # Partial search - only use if significantly better than previous depth
                improvement = current_best_score - best_score
                if improvement > 0:  # Improvement threshold
                    best_move = current_best_move
                    best_score = current_best_score
                    print(f"Depth {depth} PARTIAL ({moves_evaluated}/{total_moves} moves): Using move {best_move[0].name} with score {best_score} (improvement: +{improvement})")
                    #log_message(f"Depth {depth} PARTIAL but IMPROVED - Updated best move (searched {moves_evaluated}/{total_moves})")
                else:
                    print(f"Depth {depth} INCOMPLETE ({moves_evaluated}/{total_moves} moves): Found {current_best_move[0].name} with score {current_best_score}, keeping previous best (score {best_score}, improvement only +{improvement})")
                    #log_message(f"Depth {depth} INCOMPLETE - Keeping previous depth's best move (searched {moves_evaluated}/{total_moves})")
        else:
            # No moves were fully evaluated at this depth (likely due to timeout or all moves failed)
            if moves_evaluated == 0:
                print(f"Depth {depth} TIMEOUT: No moves completed ({total_moves} moves available) - using previous depth's best")
                #log_message(f"Depth {depth} TIMEOUT before completing first move - keeping previous best")
            else:
                print(f"Depth {depth} ERROR: Moves evaluated but no valid move found - using previous depth's best")
                #log_message(f"Depth {depth} ERROR: {moves_evaluated} moves evaluated but current_best_move is None")

        # Print statistics for this depth
        depth_nodes = nodes_explored - depth_nodes_before
        depth_time = time.time() - depth_start_time
        nodes_per_sec = depth_nodes / depth_time if depth_time > 0 else 0

        # print(f"\n{'='*60}")
        # print(f"Depth {depth} Summary:")
        # print(f"  Nodes explored: {depth_nodes} ({nodes_per_sec:.0f} nodes/sec)")
        # print(f"  Time taken: {depth_time:.3f}s")
        # print(f"  Moves evaluated: {moves_evaluated}/{total_moves}")
        # print(f"  Total nodes so far: {nodes_explored}")
        # if current_best_move:
        #     print(f"  Best move: {current_best_move[0].name} -> ({current_best_move[1].position.x},{current_best_move[1].position.y})")
        #     print(f"  Best score: {current_best_score}")
        # print(f"{'='*60}\n")

        #log_message(f"\n{'='*60}")
        #log_message(f"DEPTH {depth} SUMMARY:")
        #log_message(f"  Nodes explored: {depth_nodes} in {depth_time:.3f}s ({nodes_per_sec:.0f} nodes/sec)")
        #log_message(f"  Total nodes: {nodes_explored}")
        #log_message(f"  Moves evaluated: {moves_evaluated}/{total_moves}")
        if current_best_move:
            #log_message(f"  Best move at depth {depth}: {current_best_move[0].name} to ({current_best_move[1].position.x},{current_best_move[1].position.y}) with score {current_best_score}")
            pass
        else:
            pass
            #log_message(f"  No valid move found at depth {depth}, keeping previous best")
        #log_message(f"{'='*60}\n")

        # EARLY TERMINATION: If we found a forced checkmate, stop iterative deepening immediately
        # No need to search deeper - we already have a guaranteed winning sequence!
        if found_checkmate or current_best_score >= 999999:
            print(f"*** FORCED CHECKMATE SEQUENCE FOUND AT DEPTH {depth} - Stopping search immediately ***")
            print(f"*** Playing: {best_move[0]} to ({best_move[1].position.x},{best_move[1].position.y}) ***")
            #log_message(f"*** FORCED CHECKMATE SEQUENCE FOUND AT DEPTH {depth} - Stopping search immediately ***")
            break

        # Stop if time runs out mid-search
        if time.time() - start_time >= time_limit:
            print(f"Time limit reached after depth {depth}")
            #log_message(f"Time limit reached after depth {depth}")
            break

    # Log final decision
    total_time = time.time() - start_time
    print(f"\n{'#'*60}")
    print(f"### SEARCH COMPLETE ###")
    print(f"{'#'*60}")
    print(f"Final Move: {best_move[0].name} -> ({best_move[1].position.x},{best_move[1].position.y})")
    print(f"Final Score: {best_score}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Total Nodes: {nodes_explored}")
    print(f"Nodes/Second: {nodes_explored / total_time:.0f}")
    print(f"Quiescence: {'ENABLED' if QUIESCENCE_ENABLED else 'DISABLED'}")
    print(f"{'#'*60}\n")

    #log_message(f"\n{'#'*60}")
    #log_message(f"### FINAL MOVE SELECTION ###")
    #log_message(f"{'#'*60}")
    #log_message(f"Move: {best_move[0].name} from ({best_move[0].position.x},{best_move[0].position.y}) to ({best_move[1].position.x},{best_move[1].position.y})")
    #log_message(f"Score: {best_score}")
    #log_message(f"Total search time: {total_time:.2f}s")
    #log_message(f"Total nodes explored: {nodes_explored}")
    #log_message(f"Average nodes/second: {nodes_explored / total_time:.0f}")
    #log_message(f"Quiescence search: {'ENABLED' if QUIESCENCE_ENABLED else 'DISABLED'}")
    #log_message(f"{'#'*60}\n")

    return best_move

# ============================================================================
# MINIMAX + ALPHA-BETA + QUIESCENCE
# ============================================================================

import time

def minimax(board, depth, alpha, beta, player_name, time_limit, start_time, is_max_turn=True, indent_level=0) -> float:
    """
    Minimax search with alpha-beta pruning, time awareness, and quiescence search.

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
        # FIX: Check for ANY checkmate, not just "checkmate - black loses"
        if "checkmate" in res:
            # Prefer faster checkmates: add bonus for shallower depth
            # If we're at depth 5, checkmate is 5 moves away → bonus = (10-5) = 5
            # If we're at depth 1, checkmate is 1 move away → bonus = (10-1) = 9
            mate_bonus = (10 - depth) * 1000

            # Check if we (player_name) won or the opponent won
            # Example results: "checkmate - black loses" or "checkmate - white loses"
            if player_name in res or (player_name == "white" and "black loses" in res) or (player_name == "black" and "white loses" in res):
                return 999999 + mate_bonus  # We win: prefer faster mate
            else:
                return -999999 - mate_bonus  # We lose: delay mate as long as possible
        elif "draw" in res:
            return 0

    # --- QUIESCENCE SEARCH AT LEAF NODES ---
    if depth == 0:
        if QUIESCENCE_ENABLED:
            # Instead of static evaluation, call quiescence search
            # This resolves tactical sequences to quiet positions
            # Pass is_max_turn to maintain consistent perspective
            indent = "  " * indent_level
            #log_message(f"{indent}[Depth 0] Entering QUIESCENCE SEARCH (alpha={alpha}, beta={beta}, is_max_turn={is_max_turn})")
            q_score = quiescence_search(board, alpha, beta, player_name, time_limit, start_time, is_max_turn, q_depth=0)
            #log_message(f"{indent}[Depth 0] Quiescence returned: {q_score}")
            return q_score
        else:
            static_eval = evaluate_board(board, player_name)
            indent = "  " * indent_level
            #log_message(f"{indent}[Depth 0] Static evaluation (Q-search OFF): {static_eval}")
            return static_eval

    # --- 2 Get legal moves ----------------------------------------------
    current_player = board.current_player
    legal_moves = list_legal_moves_for(board, current_player)
    if not legal_moves:
        return evaluate_board(board, player_name)

    # --- 3 Initialize best value ----------------------------------------
    best_value = float('-inf') if is_max_turn else float('inf')

    # --- 4 Order moves for better pruning -------------------------------
    # CRITICAL-2 OPTIMIZATION: Build move cache once per board state
    move_cache = build_move_cache(board)

    # CRITICAL-3 OPTIMIZATION: Build position map once per board state
    pos_map = create_position_map(board)

    ordered_moves = order_moves(board, legal_moves, move_cache, pos_map)

    # --- 5 Explore moves ------------------------------------------------
    for piece, move in ordered_moves:
        # Check time inside loop too
        if time.time() - start_time >= time_limit:
            break

        # DEBUG: Print what's being explored (comment out for production)
        indent = "  " * indent_level
        turn_type = "MAX" if is_max_turn else "MIN"
        # print(f"{indent}[Depth {depth}, {turn_type}] Testing: {piece.name} to ({move.position.x},{move.position.y})")
        #log_message(f"{indent}[Depth {depth}, {turn_type}] Testing: {piece.name} to ({move.position.x},{move.position.y})")

        new_board = board.clone()
        try:
            # Locate piece & move equivalents on cloned board
            new_piece = next((p for p in new_board.get_player_pieces(current_player)
                              if type(p) == type(piece) and p.position == piece.position), None)
            if not new_piece:
                continue

            # CRITICAL-2 OPTIMIZATION: Use cached moves instead of calling get_move_options()
            cached_moves = get_cached_moves(new_piece, move_cache)
            new_move = next((m for m in cached_moves
                             if hasattr(m, "position") and m.position == move.position), None)

            if not new_move:
                continue

            new_piece.move(new_move)
            # Switch turn
            new_board.current_player = [p for p in new_board.players if p != current_player][0]

            # Check for stalemate using our helper function
            if is_stalemate(new_board):
                # Stalemate/Draw in the middle of minimax tree:
                # Evaluate the position BEFORE the stalemate move to determine if draw is good/bad
                # Use the board state before this move (from parent node)
                current_eval = evaluate_board(board, player_name)

                # If we're winning (eval > 200), a draw is BAD
                # If we're losing (eval < -200), a draw is GOOD
                # Use conservative thresholds to avoid accepting draws in equal positions

                if current_eval > 200:
                    # We're winning - draw is terrible, we're throwing away a win
                    return -50000
                elif current_eval < -200:
                    # We're losing - draw is great, we're escaping a loss
                    return 50000
                else:
                    # Position is roughly equal - draw is neutral
                    return 0

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
            #log_message(f"{indent}  -> Returned: {value} (alpha={alpha}, beta={beta})")

            # --- 8️⃣ Alpha-beta updates -----------------------------------
            if is_max_turn:
                best_value = max(best_value, value)
                alpha = max(alpha, value)
                if value > best_value or best_value == float('-inf'):
                    pass
                    #log_message(f"{indent}  -> New MAX best: {value}")
            else:
                best_value = min(best_value, value)
                beta = min(beta, value)
                if value < best_value or best_value == float('inf'):
                    #log_message(f"{indent}  -> New MIN best: {value}")
                    pass
            if beta <= alpha:
                #log_message(f"{indent}  -> PRUNED! (beta={beta} <= alpha={alpha})")
                break  # Prune rest of branch

        except Exception:
            continue

    return best_value if best_value != float('inf') and best_value != float('-inf') else evaluate_board(board, player_name)

# ============================================================================
# AGENT ENTRY POINT
# ============================================================================

def agent(board, player, var):
    """
    Main agent entry point for COMP2321 system.

    agentE.py - Endgame-Enhanced Agent (Phase B)

    Features:
    - Quiescence Search (from agentQ.py)
    - Endgame classification (pawn_race, mating_attack, etc.)
    - Specialized endgame evaluation functions:
      * Mobility restriction (2-4 moves ideal)
      * King opposition detection
      * Pawn promotion race analysis
      * Passed pawn evaluation
      * Mating net evaluation (edge drive + king cooperation)
    - Endgame-specific piece-square tables

    Time limit: 12.5 seconds (conservative for 14s per-move limit)
    Max depth: 10 (same as agentQ - dynamic depth reserved for Phase C)
    """
    piece, move = find_best_move(board, player, time_limit=12.5)
    if piece is None or move is None:
        legal = list_legal_moves_for(board, player)
        if legal:
            piece, move = random.choice(legal)
            #log_message(f"No best move found, playing random move: {piece.name} to ({move.position.x},{move.position.y})")
    else:
        pass
        #log_message(f"\n*** FINAL DECISION: Playing {piece.name} from ({piece.position.x},{piece.position.y}) to ({move.position.x},{move.position.y}) ***")

    return piece, move
