# COMP2321: Chess Fragments - Project Summary

## Project Overview

**Chess Fragments** is a 5×5 chess variant game where you develop an AI agent to compete against predefined opponent algorithms. The game features traditional chess pieces plus a custom "Right" piece.

### Game Rules
- **Board**: 5×5 grid
- **Pieces**: Right, Knight, Bishop, Queen, King, Pawn
- **Special Piece - Right**: Combines rook movement (horizontal/vertical) with knight movement (L-shape), can jump over pieces
- **Pawn Promotion**: Pawns promote to Queens when reaching the back rank
- **Win Conditions**: Checkmate, stalemate, timeout (opponent loses), illegal move (player loses)
- **Draw Conditions**: Two kings only, fivefold repetition, overall game timeout

---

## Your Algorithmic Strategy (agent4.py)

### Core Algorithm: **Minimax with Alpha-Beta Pruning + Iterative Deepening**

---

### 1. **Evaluation Function** 

Your position evaluation considers three key factors:

#### A. Material Balance
Base piece values defined in `PIECE_VALUES`:
```python
'pawn':   100
'knight': 330
'bishop': 320
'right':  500
'queen':  900
'king':   20000
```

**Note**: The Right piece (500) is valued between a Bishop/Knight (~320) and a Queen (900), reflecting its hybrid rook+knight capabilities.

#### B. Positional Value - Piece-Square Tables

Each piece has a 5×5 table that awards bonuses/penalties for square occupation (from White's perspective):

**PAWN_TABLE** - Encourages forward advancement:
```
Row 0 (back):   [10,  10,  10,  10,  10]  ← Promotion imminent
Row 1:          [ 5,   5,   5,   5,   5]
Row 2:          [ 5,   5,   5,   5,   5]
Row 3:          [-5,   5,  -5,   5,   0]
Row 4 (start):  [ 0,   0,   0,   0,   0]
```

**KNIGHT_TABLE** - Prefers center, avoids edges:
```
[-5,  -5,  -5,  -5,  -5]
[-5,   0,   0,   0,  -5]
[-5,   0,   0,   0,  -5]  ← Center squares valued
[-5,   0,   0,   0,  -5]
[-5,  -5,  -5,  -5,  -5]
```

**BISHOP_TABLE** - Values center control:
```
[-10,  -5,  -5,  -5, -10]
[ -5,   0,   0,   0,  -5]
[ -5,   0,   5,   0,  -5]  ← Center peak
[ -5,   0,   0,   0,  -5]
[-10,  -5,  -5,  -5, -10]
```

**RIGHT_TABLE** - Prefers aggressive back-rank positioning:
```
[-5,   5,   5,   5,  -5]  ← Back rank advantage
[ 0,   5,   5,   5,   0]
[ 0,   0,   0,   0,   0]
[ 0,   0,   0,   0,   0]
[-5,   0,   0,   0,  -5]
```

**QUEEN_TABLE** - Center control focus:
```
[-5,   0,   0,   0,  -5]
[ 0,   5,   5,   5,   0]
[ 0,   5,   5,   5,   0]  ← Strong center presence
[ 0,   5,   5,   5,   0]
[-5,   0,   0,   0,  -5]
```

**KING_TABLE (Middlegame)** - Stay on back rank for safety:
```
[-20, -20, -20, -20, -20]  ← Heavily penalized exposure
[-15, -15, -15, -15, -15]
[-10, -10, -10, -10, -10]
[ -5,  -5,  -5,  -5,  -5]
[  5,   5,   5,   5,   5]  ← Safe home rank
```

**KING_TABLE_ENDGAME (≤4 pieces)** - Neutral positioning:
```
All zeros - allows king to activate freely
```

#### C. King Safety 
Dynamic evaluation based on nearby friendly pieces:
- **≥2 allies within 1 square**: +50 bonus
- **1 ally within 1 square**: +20 bonus
- **0 allies (isolated king)**: -50 penalty

---

### 2. **Move Ordering** 

Optimizes alpha-beta efficiency by prioritizing moves in descending order of tactical importance:

#### Priority Hierarchy:
1. **Checkmate moves** (score: +100,000,000) - Instant win detection
2. **Valuable captures** using **MVV-LVA** (Most Valuable Victim - Least Valuable Attacker)
3. **Positional improvements** based on piece-square table gains

#### Capture Evaluation with Exchange Analysis

The `attacker_defender_ratio()` function performs sophisticated tactical analysis:

**Function Logic**:
1. Identifies all pieces attacking the target square
2. Identifies all pieces defending the target square (excluding the target piece itself)
3. Calculates `num_diff = attackers - defenders`
4. If `num_diff ≤ 0`: Defenders hold → return `(num_diff, None)`
5. If `num_diff > 0`: Simulate optimal exchange sequence:
   - Sort attackers/defenders by value (cheapest first)
   - Exclude the `num_diff` most valuable attackers (they won't be traded)
   - Simulate alternating captures until one side runs out
   - Return `(num_diff, val_diff)` where `val_diff` is net material outcome

**Example Exchange Calculation**:
```
Target square: Enemy Pawn (100)
Attackers: Pawn (100), Knight (330), Queen (900)  → 3 attackers
Defenders: Pawn (100)                              → 1 defender

num_diff = 3 - 1 = 2 (excess attackers)
Exchange simulation (exclude 2 strongest: Knight, Queen):
  1. Pawn captures Pawn: +100 (gain target)
  2. Defender Pawn recaptures: -100 (lose attacking pawn)
  3. Attacking piece recaptures (now an undefended pawn)
  3. No more defenders → exchange ends
val_diff = +100 - 100 + 100 = 100 (pozitive trade, and we keep Knight & Queen)
```

**Move Ordering Integration** :
- **More attackers than defenders + positive trade**: `base_mvv_lva + 1000` bonus
- **Equal/fewer attackers but favorable victim value**: `base_mvv_lva` score only
- Base MVV-LVA formula: `(victim_value × 10) - attacker_value`

---

### 3. **Iterative Deepening Search** 

Progressively deepens search with time-aware fallback mechanism:

#### Configuration:
- **Depth range**: 1 → 10
- **Time limit**:  Flexible (defined in agent() in agent4.py)
- **Fallback strategy**: Uses previous depth's best move if timeout occurs

#### Key Features:
1. **Partial Depth Handling** :
   - Only trusts incomplete depth if score improves significantly
   - Tracks `moves_evaluated` vs `total_moves` to assess completeness

2. **Early Termination** :
   - Stops immediately upon finding forced checkmate (score ≥999,999)
   - No need to search deeper when mate sequence is guaranteed

3. **Statistics Tracking**:
   - Nodes explored per depth
   - Time spent per depth
   - Cumulative totals

#### Decision Logic:
```python
if all_moves_searched:
    # Trust complete depth unconditionally
    best_move = current_best_move
elif improvement > 0:
    # Accept partial depth if better than previous
    best_move = current_best_move
else:
    # Keep previous depth's result
    # (partial depth not sufficiently better)
```

---

### 4. **Minimax with Alpha-Beta Pruning**

Recursive search algorithm with aggressive optimizations:

#### Terminal Conditions:
1. **Checkmate**: ±999,999 + mate_bonus
   - `mate_bonus = (10 - depth) × 1000`
   - Prefers faster mates, delays losses
2. **Draw/Stalemate**: 0
3. **Depth limit**: Return static evaluation
4. **Time limit**: Return immediate evaluation

#### Stalemate Handling:

**Detection** (`is_stalemate()` function checks for):
- "stalemate (no more possible moves) - black/white loses"
- "draw - only 2 kings left"
- "draw - fivefold repetition"

**Strategic Response**:
- **In root search**:
  - If `eval > -500` (winning): Skip stalemate move
  - If `eval ≤ -500` (losing): Accept stalemate as escape

- **In minimax tree**:
  - Maximizer causes stalemate while winning: -50,000 penalty
  - Minimizer causes stalemate (opponent forced draw): +50,000 bonus

#### Alpha-Beta Pruning:
```python
if is_max_turn:
    best_value = max(best_value, value)
    alpha = max(alpha, value)
else:
    best_value = min(best_value, value)
    beta = min(beta, value)

if beta <= alpha:
    break  # Prune remaining branches
```

#### Time Management:
- Checks `time.time() - start_time >= time_limit` at:
  - Entry to function (line 507)
  - Inside move loop (line 548)
- Returns static evaluation immediately on timeout

---

## Performance Optimizations

### Evaluation Function
✓ **Single-pass iteration**: Calculates material, positional, and king safety in one loop
✓ **Piece categorization**: Simultaneously collects player/opponent pieces during evaluation

### Move Ordering
✓ **Position-to-piece lookup**: O(1) capture target identification
✓ **Cached piece counts**: Computed once instead of per-move for endgame detection
✓ **Early MVV-LVA calculation**: Sorts captures by expected value

### Search
✓ **Move ordering**: Better pruning through tactical prioritization
✓ **Iterative deepening**: Goes deeper as long as there is time.
✓ **Alpha-beta pruning**: Eliminates branches determined "useless".
✓ **Early mate termination**: Stops search when forced win found

---

## Implementation Architecture

### File Structure
- **agent4.py**: Main agent logic (search, minimax, move ordering)
- **helpers.py**: Evaluation functions (piece values, tables, exchange analysis)
- **extension/**: Custom game components (Right piece, board utilities)
- **chessmaker/**: Base chess framework

### Helper Functions (helpers.py)

| Function | Purpose | Lines |
|----------|---------|-------|
| `get_piece_value(piece)` | Returns base material value | 208-209 |
| `get_positional_value(piece, is_white, board)` | Returns piece-square table bonus | 211-232 |
| `attacker_defender_ratio(board, pos, attacker, defender)` | Calculates exchange outcomes | 74-206 |
| `is_stalemate(board)` | Detects stalemate/draw conditions | 234-258 |

### Logging System (agent4.py:14-82)
- **File**: `game_log.txt`
- **Contents**: Board states, move evaluations, minimax traces, search statistics
- **Format**: Readable text with indentation showing search depth

---

## Algorithm Summary

Your implementation combines **classical game-tree search** with **modern optimizations**:

1. **Minimax**: Exhaustively explores game tree assuming optimal play
2. **Alpha-Beta Pruning**: Eliminates provably suboptimal branches
3. **Iterative Deepening**: Provides anytime algorithm behavior (always has a valid move)
4. **Move Ordering**: MVV-LVA + positional heuristics maximize pruning efficiency
5. **Sophisticated Evaluation**: Material + Position + King Safety + Tactical awareness
6. **Time Management**: Graceful degradation under time pressure
7. **Tactical Awareness**: Checkmate detection, exchange evaluation, stalemate avoidance

This represents a **strong classical AI approach** suitable for competitive play in the 5×5 Chess Fragments variant!
