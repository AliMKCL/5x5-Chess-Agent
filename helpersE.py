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