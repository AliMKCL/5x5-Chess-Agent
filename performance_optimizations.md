# Performance Optimizations for agentBitboard_gemini.py and helpersBitboard.py

These optimizations improve speed WITHOUT changing move selection logic.

---

## HIGH IMPACT Optimizations

### 1. **Disable Logging in Production** (agentBitboard_gemini.py:188)
```python
LOGGING_ENABLED = True  # Change to False
```
**Impact:** Each `log_message()` call opens a file, writes, and closes. This happens thousands of times per search.
**Fix:** Set `LOGGING_ENABLED = False` or remove logging calls entirely.

---

### 2. **Remove SEE from Move Ordering** (agentBitboard_gemini.py:225)
```python
see_score = static_exchange_eval(bb_state, move)  # EXPENSIVE!
```
**Impact:** `static_exchange_eval()` is called for EVERY capture move during ordering. It iterates all pieces, builds attacker lists, sorts them, and simulates exchanges. This is very slow.
**Fix:** Remove SEE entirely and rely only on MVV-LVA:
```python
def score_move(move: BBMove, bb_state: BitboardState, tt_best_move: Optional[Tuple] = None) -> int:
    if tt_best_move and (move.from_sq, move.to_sq) == tt_best_move:
        return 10_000_000
    if move.captured_type != -1:
        return (MVV_LVA_VALUES[move.captured_type] * 10) - MVV_LVA_VALUES[move.piece_type]
    return 0
```

---

### 3. **Use `__slots__` for BBMove and BitboardState** (helpersBitboard.py:214, 844)
Dataclasses create `__dict__` for each instance, which is slow.
**Fix:** Add `__slots__` to prevent dict creation:
```python
@dataclass(slots=True)
class BitboardState:
    ...

@dataclass(slots=True)
class BBMove:
    ...
```
**Note:** Requires Python 3.10+. For older Python, manually define `__slots__`.

---

### 4. **Cache Zobrist Keys as Tuple Instead of Dict** (helpersBitboard.py:466-471)
```python
self.piece_keys[(piece_type, color, sq)] = key  # Dict lookup is slow
```
**Fix:** Use a flat list with index calculation:
```python
# In __init__:
self.piece_keys = [0] * (6 * 2 * 25)  # 300 entries
for piece_type in range(6):
    for color in range(2):
        for sq in range(25):
            idx = piece_type * 50 + color * 25 + sq
            self.piece_keys[idx] = rng.getrandbits(64)

# In usage:
def get_key(piece_type, color, sq):
    return self.piece_keys[piece_type * 50 + color * 25 + sq]
```

---
DONE
### 5. **Inline `index_to_xy` and `square_index`** (helpersBitboard.py:110, 128)
These are called thousands of times per search.
```python
def index_to_xy(sq): return (sq % 5, sq // 5)
def square_index(x, y): return y * 5 + x
```
**Fix:** Inline these calculations directly where used, or use `divmod()`:
```python
x, y = divmod(sq, 5)  # Note: returns (quotient, remainder) = (y, x)
# So use: y, x = divmod(sq, 5)
```

---
DONE
### 6. **Precompute Sliding Attack Tables** (helpersBitboard.py:337-436)
`_get_rook_attacks()` and `_get_bishop_attacks()` are called repeatedly with the same squares.
**Fix:** Precompute attack tables for all 25 squares × all possible occupancies (or use magic bitboards for larger boards). For 5x5, simple caching works:
```python
# Precompute rook attacks for each square (ignoring occupancy for rough estimate)
ROOK_RAYS = [_get_rook_attacks(sq, 0) for sq in range(25)]
BISHOP_RAYS = [_get_bishop_attacks(sq, 0) for sq in range(25)]
```
Then use rays as masks for quick checks before full calculation.

---

### 7. **Avoid Creating New Lists in Hot Paths** (helpersBitboard.py:904, 1071)
```python
own_pieces = [
    (bb_state.WP, PAWN), (bb_state.WN, KNIGHT), ...
]
```
**Fix:** Define these as module-level constants or use tuples:
```python
WHITE_PIECE_ATTRS = ('WP', 'WN', 'WB', 'WQ', 'WK', 'WR')
BLACK_PIECE_ATTRS = ('BP', 'BN', 'BB', 'BQ', 'BK', 'BR')
PIECE_TYPES = (PAWN, KNIGHT, BISHOP, QUEEN, KING, RIGHT)
```

---

### 8. **Use Local Variables in Tight Loops** (helpersBitboard.py)
Python looks up global/module variables slower than local variables.
**Fix:** At the start of hot functions, assign globals to locals:
```python
def generate_legal_moves(bb_state, captures_only=False):
    _KNIGHT_ATTACKS = KNIGHT_ATTACKS  # Local reference
    _KING_ATTACKS = KING_ATTACKS
    _iter_bits = iter_bits
    ...
```

---

### 9. **Reduce Object Creation in `apply_move`** (helpersBitboard.py:1092)
Every move creates a new `BitboardState` object.
**Fix:** Consider using mutable state with undo functionality instead of immutable copies. This requires significant refactoring but can be 2-3x faster.

---
DONE
### 10. **Remove Redundant TT Format Conversion** (agentBitboard_gemini.py:470-477)
```python
if isinstance(tt_move, tuple) and len(tt_move) == 2:
    from_pos, to_pos = tt_move
    if isinstance(from_pos, tuple) and isinstance(to_pos, tuple):
        ...
```
The TT stores `(from_sq, to_sq)` as integers (line 517), but retrieval expects tuples. This check always fails!
**Fix:** Store and retrieve consistently:
```python
# In minimax, already stores correctly:
tt_move_tuple = (best_move.from_sq, best_move.to_sq)  # Two integers

# In probe usage, use directly:
if tt_move:
    tt_best_move_tuple = tt_move  # Already (from_sq, to_sq) integers
```

---

## MEDIUM IMPACT Optimizations

DONE
### 11. **Remove `time.time()` Calls from Inner Loop** (agentBitboard_gemini.py:283, 421)
```python
if time.time() - start_time > TIME_LIMIT:
```
**Fix:** Check time less frequently (every N nodes):
```python
if stats['nodes_searched'] % 1000 == 0 and time.time() - start_time > TIME_LIMIT:
```

---
DONE (?)
### 12. **Use `@lru_cache` for `is_in_check`** (helpersBitboard.py:520)
Check detection is called multiple times for the same position.
**Fix:** Cache results based on zobrist hash:
```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def is_in_check_cached(zobrist_hash, check_white, WK, BK, ...):
    ...
```
**Caveat:** Need to pass all relevant bitboards as hashable arguments.

---
DONE
### 13. **Avoid `sorted()` - Use In-Place Sort** (agentBitboard_gemini.py:253)
```python
return sorted(moves, key=lambda m: score_move(m, bb_state, tt_best_move), reverse=True)
```
**Fix:** Sort in-place to avoid creating new list:
```python
moves.sort(key=lambda m: score_move(m, bb_state, tt_best_move), reverse=True)
return moves
```

---
DONE
### 14. **Precompute King Safety Neighbor Masks** (helpersBitboard.py:1474-1483)
The nested loop checking neighbors is repeated for every evaluation.
**Fix:** Use precomputed KING_ATTACKS for neighbor squares:
```python
# Already have KING_ATTACKS[sq] which gives all adjacent squares!
nearby_allies = count_bits(KING_ATTACKS[white_king_sq] & bb_state.occ_white)
```

---

### 15. **Remove Mate-in-1 Check at Root** (agentBitboard_gemini.py:564-579)
```python
for move in moves:
    new_state = apply_move(bb_state, move)
    opponent_moves = generate_legal_moves(new_state)
    ...
```
This pre-check iterates all moves and generates opponent moves for each. The minimax search will find mate-in-1 anyway at depth 1.
**Fix:** Remove this loop entirely. The search will find it.

---

## LOW IMPACT (But Easy) Optimizations

DONE
### 16. **Use `float('inf')` Constants** (agentBitboard_gemini.py)
```python
best_score = -float('inf')  # Creates new float object each time
```
**Fix:** Define constants:
```python
NEG_INF = float('-inf')
POS_INF = float('inf')
```

---

### 17. **Remove Print Statements** (agentBitboard_gemini.py:576, 631, 651, etc.)
Print statements are slow due to I/O.
**Fix:** Remove or guard with a flag.

---

### 18. **Use Tuple Unpacking in Comprehensions** (helpersBitboard.py)
```python
for piece_bb, piece_type in pieces:  # Already good
```
This is already efficient, but ensure no unnecessary intermediate structures.

---

## Summary: Priority Order

1. **LOGGING_ENABLED = False** - Immediate, huge impact
2. **Remove SEE from move ordering** - Major speedup
3. **Fix TT move retrieval format** - TT ordering currently broken!
4. **Remove mate-in-1 pre-check** - Removes redundant work
5. **Use `__slots__` on dataclasses** - Reduces memory allocation
6. **Inline `index_to_xy`/`square_index`** - Many calls per search
7. **Local variable caching in hot functions** - Python-specific speedup
8. **Reduce time checks** - Check every N nodes instead of every node
9. **Precompute neighbor masks for king safety** - Uses existing KING_ATTACKS

---

## Expected Performance Gains

| Optimization | Estimated Speedup |
|--------------|-------------------|
| Disable logging | 10-20% |
| Remove SEE | 20-40% |
| Fix TT ordering | 10-30% (better pruning) |
| Remove mate-in-1 check | 5-10% |
| `__slots__` | 5-15% |
| Inline functions | 5-10% |
| Local variables | 5-10% |
| Reduce time checks | 2-5% |

**Combined potential improvement: 50-100%+ faster search**
