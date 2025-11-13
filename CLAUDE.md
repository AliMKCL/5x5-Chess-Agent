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

## Agent Implementations

### agentS.py - Standard Minimax Agent
**Core Algorithm**: Minimax with Alpha-Beta Pruning + Iterative Deepening

The foundational agent implementing classical game-tree search with:
- Material + positional evaluation
- MVV-LVA move ordering
- Alpha-beta pruning
- Iterative deepening (depth 1-10)
- Time management

### agentQ.py - Quiescence Search Agent
**Core Algorithm**: agentS.py + Quiescence Search

Extends agentS.py with tactical horizon extension:
- Quiescence search at leaf nodes (max depth 7)
- Resolves capture sequences to "quiet" positions
- Prevents horizon effect in tactical positions
- Better tactical awareness in complex positions

### agentE.py - Endgame-Enhanced Agent
**Core Algorithm**: agentQ.py + Advanced Endgame Handling

Builds upon agentQ.py with sophisticated endgame capabilities:
- **Endgame classification**: Detects pawn races, mating attacks, minor piece endings
- **Mobility restriction**: Limits opponent moves to 2-4 (avoiding stalemate)
- **Specialized evaluation**: Pawn promotion races, king opposition, mating nets
- **Dynamic search depth**: Searches deeper (up to depth 20) with fewer pieces
- **Endgame-specific piece-square tables**: Active king, aggressive pawn advancement

---

## Core Algorithmic Strategy (Shared by All Agents)

### 1. **Evaluation Function**

Position evaluation considers three key factors:

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

**KING_TABLE_ENDGAME (≤4 pieces, agentS/Q)** - Neutral positioning:
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

**Move Ordering Integration**:
- **More attackers than defenders + positive trade**: `base_mvv_lva + 1000` bonus
- **Equal/fewer attackers but favorable victim value**: `base_mvv_lva` score only
- Base MVV-LVA formula: `(victim_value × 10) - attacker_value`

---

### 3. **Iterative Deepening Search**

Progressively deepens search with time-aware fallback mechanism:

#### Configuration:
- **agentS.py**: Depth range 1 → 10, time limit ~13s
- **agentQ.py**: Depth range 1 → 10, time limit ~12.5s
- **agentE.py**: Depth range 1 → 20 (dynamic), time limit ~12.5s
- **Fallback strategy**: Uses previous depth's best move if timeout occurs

#### Key Features:
1. **Partial Depth Handling**:
   - Only trusts incomplete depth if score improves significantly
   - Tracks `moves_evaluated` vs `total_moves` to assess completeness

2. **Early Termination**:
   - Stops immediately upon finding forced checkmate (score ≥999,999)
   - No need to search deeper when mate sequence is guaranteed

3. **Statistics Tracking**:
   - Nodes explored per depth
   - Time spent per depth
   - Cumulative totals

---

### 4. **Minimax with Alpha-Beta Pruning**

Recursive search algorithm with aggressive optimizations:

#### Terminal Conditions:
1. **Checkmate**: ±999,999 + mate_bonus
   - `mate_bonus = (10 - depth) × 1000`
   - Prefers faster mates, delays losses
2. **Draw/Stalemate**: 0
3. **Depth limit**: Return static evaluation (or quiescence search in agentQ/E)
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

---

## Endgame Enhancements (agentE.py Only)

### Endgame Classification System

**`classify_endgame_type(board, player_name)`** detects specific material configurations:

- **'pawn_race'**: Pawn endgames → Triggers opposition + promotion race evaluation
- **'mating_attack'**: Queen/Right vs King → Triggers mobility restriction + edge drive
- **'minor_piece_endgame'**: Only Knights/Bishops → Triggers king activity bonus
- **'complex_endgame'**: 5-8 pieces → General endgame improvements
- **'none'**: Middlegame (>8 pieces)

### Endgame-Specific Piece-Square Tables

**KING_TABLE_ENDGAME_ACTIVE** - Centralized king (used in agentE.py):
```
[-10, -5,  0, -5, -10]
[ -5, 10, 15, 10,  -5]
[  0, 15, 20, 15,   0]  ← Strong center control
[ -5, 10, 15, 10,  -5]
[-10, -5,  0, -5, -10]
```

**PAWN_TABLE_ENDGAME** - Aggressive promotion drive:
```
[200, 200, 200, 200, 200]  ← Promotion imminent!
[ 80,  80,  80,  80,  80]  ← Very close
[ 40,  40,  40,  40,  40]  ← Halfway
[ 15,  15,  15,  15,  15]
[  0,   0,   0,   0,   0]
```

**QUEEN_TABLE_ENDGAME** - Centralized control for mating:
```
[-20,  -5,   0,  -5, -20]
[ -5,  15,  20,  15,  -5]
[  0,  20,  25,  20,   0]  ← Maximum control
[ -5,  15,  20,  15,  -5]
[-20,  -5,   0,  -5, -20]
```

**RIGHT_TABLE_ENDGAME** - Similar centralized control:
```
[-20,  -5,   0,  -5, -20]
[ -5,  15,  20,  15,  -5]
[  0,  20,  25,  20,   0]
[ -5,  15,  20,  15,  -5]
[-20,  -5,   0,  -5, -20]
```

### Specialized Endgame Evaluation Functions

#### 1. Mobility Restriction (`evaluate_mating_net`)
**Key Innovation**: Restricts opponent king mobility to ideal range (2-4 moves)

```python
opponent_mobility = count_legal_moves(opponent)

if opponent_mobility == 0:
    penalty = -10000  # Stalemate!
elif opponent_mobility == 1:
    penalty = -5000   # Risky, close to stalemate
elif 2 <= opponent_mobility <= 4:
    bonus = +300      # Perfect restriction!
else:
    penalty = -(opponent_mobility * 20)  # Too much freedom
```

Also includes:
- **Edge drive**: Penalty based on king's distance to nearest edge
- **King cooperation**: Bonus when our king is 2-3 squares away (supporting distance)

#### 2. Pawn Promotion Race (`evaluate_pawn_promotion_race`)
Analyzes race to promotion:
- Calculates squares to promotion for each pawn
- Compares our king's distance vs opponent's king distance to promotion square
- Awards bonus if our king is closer (can support pawn)
- Checks if pawn path is clear

#### 3. King Opposition (`evaluate_king_opposition`)
Detects opposition patterns:
- **Direct opposition**: Kings facing with 1 square between (same file/rank)
- **Distant opposition**: Kings 2+ squares apart on same file/rank
- Awards bonus if we have opposition

#### 4. Passed Pawns (`evaluate_passed_pawns`)
Identifies pawns with no opposing pawns on:
- Same file
- Adjacent files (no blockers in path)
Awards large bonuses for passed pawns

#### 5. Key Squares (`evaluate_key_squares`)
In pawn endgames, certain squares near promotion are critical
- Identifies promotion square and squares ±1 rank
- Awards bonus for controlling these with our king

#### 6. King Activity (`evaluate_king_activity`)
General endgame king centralization bonus

#### 7. Mobility Advantage (`evaluate_mobility_advantage`)
Compares total mobility (legal moves) between players
- Having more moves = positional advantage

### Dynamic Search Depth (agentE.py)

Adjusts max depth based on piece count:

```python
total_pieces <= 4:  max_depth = 20  # Very deep endgame
total_pieces <= 6:  max_depth = 15  # Deep search
total_pieces <= 8:  max_depth = 12
Otherwise:          max_depth = 10  # Standard middlegame
```

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
✓ **Iterative deepening**: Goes deeper as long as there is time
✓ **Alpha-beta pruning**: Eliminates branches determined "useless"
✓ **Early mate termination**: Stops search when forced win found
✓ **Quiescence search** (agentQ/E): Resolves tactical sequences
✓ **Dynamic depth** (agentE): Searches deeper with fewer pieces

---

## Implementation Architecture

### File Structure
- **agentS.py**: Standard minimax agent (search, minimax, move ordering)
- **agentQ.py**: Quiescence search agent (builds on agentS.py)
- **agentE.py**: Endgame-enhanced agent (builds on agentQ.py) 🆕
- **helpers.py**: Evaluation functions (piece values, tables, exchange analysis, endgame evaluation)
- **extension/**: Custom game components (Right piece, board utilities)
- **chessmaker/**: Base chess framework

### Helper Functions (helpers.py)

#### Basic Evaluation Functions
| Function | Purpose |
|----------|---------|
| `get_piece_value(piece)` | Returns base material value |
| `get_positional_value(piece, is_white, board)` | Returns piece-square table bonus |
| `attacker_defender_ratio(board, pos, attacker, defender)` | Calculates exchange outcomes |
| `is_stalemate(board)` | Detects stalemate/draw conditions |

#### Endgame Evaluation Functions 🆕
| Function | Purpose |
|----------|---------|
| `classify_endgame_type(board, player_name)` | Detects endgame type classification |
| `count_mobility(board, player)` | Counts legal moves for mobility restriction |
| `evaluate_king_opposition(board, player_name)` | Opposition detection for pawn endgames |
| `evaluate_pawn_promotion_race(board, player_name)` | Analyzes race to promotion |
| `evaluate_mating_net(board, player_name)` | Mobility restriction + edge drive + cooperation |
| `evaluate_passed_pawns(board, player_name)` | Detects and values passed pawns |
| `evaluate_key_squares(board, player_name)` | Control of critical squares |
| `evaluate_king_activity(board, player_name)` | General endgame king centralization |
| `evaluate_mobility_advantage(board, player_name)` | Mobility-based evaluation |

### Logging System
- **File**: `game_log.txt`
- **Contents**: Board states, move evaluations, minimax traces, search statistics
- **Format**: Readable text with indentation showing search depth

---

## Algorithm Summary

### agentS.py - Classical Approach
1. Minimax with alpha-beta pruning
2. Iterative deepening (depth 1-10)
3. MVV-LVA move ordering
4. Material + position + king safety evaluation
5. Time management

### agentQ.py - Tactical Enhancement
All of agentS.py, plus:
6. Quiescence search (resolves capture sequences)
7. Better tactical awareness

### agentE.py - Endgame Mastery 🆕
All of agentQ.py, plus:
8. **Endgame classification** (pawn race, mating attack, etc.)
9. **Mobility restriction** (limits opponent to 2-4 moves)
10. **Specialized endgame evaluation** (opposition, promotion race, mating nets)
11. **Dynamic search depth** (20 ply in simple endgames)
12. **Endgame-specific piece-square tables** (active king, aggressive pawns)

This represents a **sophisticated AI system** combining classical game-tree search with modern tactical awareness and expert endgame knowledge - suitable for competitive play in the 5×5 Chess Fragments variant!
