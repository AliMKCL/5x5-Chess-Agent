# Checkmate Score Issue Analysis

## The Problem

Position:
```
  0 1 2 3 4
0 . q k . r
1 . . p . b
2 . p R p P
3 . P . Q .
4 . B K . N
```

Agent (White) is evaluating Right (2,2) to (0,1).

**Observed Behavior:**
- Depths 1-5: Right (2,2) to (0,1) scores 1,003,999 (checkmate score)
- Depth 6: King (2,4) to (3,4) scores 1,215 (normal eval)
- Agent plays King to (3,4)

**Score breakdown:**
- 1,003,999 = CHECKMATE_SCORE (999,999) + mate_bonus (4,000)
- mate_bonus = depth * 1000 = 4 * 1000 = 4,000
- This means checkmate was detected at depth=4 remaining

**Reality:**
- Right to (0,1) is NOT checkmate
- Right to (0,1) loses the Right piece
- Best move is Right to (1,0) capturing the Queen

## Hypothesis: Phantom Checkmate

The agent found a checkmate score where no forced checkmate exists. This suggests:

### Possible Causes:

1. **Bug in legal move generation after Right to (0,1)**
   - After Right to (0,1), the move generator might be incorrectly returning no legal moves for the opponent
   - This would trigger the checkmate detection at line 441-463 in minimax
   - Check: `generate_legal_moves()` might have a bug with the specific position after Right captures

2. **Bug in check detection**
   - The `is_in_check()` function might incorrectly report that Black king is in check
   - Combined with (1), this would create a phantom checkmate

3. **Bug in apply_move() for Right piece**
   - The bitboard state after applying Right to (0,1) might be corrupted
   - This could cause move generation to fail

4. **Alpha-beta pruning issue**
   - At shallow depths (1-5), the alpha-beta might be pruning actual escape moves
   - This would make it appear as checkmate when there are actually legal moves
   - BUT: Alpha-beta should never affect the root move's evaluation

5. **Quiescence search phantom mate**
   - Quiescence search detects checkmate (lines 294-312)
   - Returns 939,999 (not 1,003,999), so this is NOT the source

## The Score Propagation

Looking at the scores:
- Depths 1-4: 21 nodes, instant completion, score 1,003,999
- Depth 5: 5,866 nodes, 1.98s, score 1,003,999
- Depth 6: 10,078 nodes, 6.12s, score 1,215

**Why did depth 6 refute it?**
- At depth 6, the agent searched one more ply
- This extra ply revealed that the "checkmate" is not forced
- The opponent has an escape move that was only found at depth 6

**This suggests:** At depths 1-5, the opponent's escape move was being pruned or not generated.

## Investigation Steps

1. **Add logging after Right to (0,1):**
   - Print the board state after applying Right to (0,1)
   - Print all legal moves for Black
   - Check if move count is 0 (which would trigger checkmate)

2. **Check alpha-beta pruning:**
   - Ensure pruning doesn't affect root node evaluation
   - Verify that all root moves are fully searched

3. **Verify move generation:**
   - Test `generate_legal_moves()` on the position after Right to (0,1)
   - Ensure it returns all legal moves for Black

4. **Check Right piece move logic:**
   - The Right piece combines Rook + Knight moves
   - Verify `_get_right_attacks()` is correct
   - Verify `apply_move()` correctly updates bitboards for Right pieces

## Recommended Fix

The immediate issue (playing worse moves from incomplete depths) is ALREADY FIXED by the timeout check at line 637. The agent correctly keeps the previous depth's best move when timeout occurs.

The deeper issue is the phantom checkmate. This requires investigation into:
- Move generation correctness
- Check detection correctness
- Apply move correctness for Right pieces
