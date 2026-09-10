# 8x8 Chess Agent

A high-performance chess engine and AI agent for standard 8×8 chess. Built from the ground up in Python with a 64-square bitboard architecture, precomputed ray attacks, Zobrist hashing, iterative deepening, transposition tables, late move reduction (LMR), move ordering heuristics, and quiescence search.

* Check the file "project_description.md" for the original rules reference (written for the 5×5 variant this engine was ported from; core piece movement rules are unchanged for standard chess).

---

## Performance Overview

- **Search Depth:** Depth reached in the standard 10-second thinking budget depends on position complexity, with an additional **quiescence search up to depth 7**.
- **Node Throughput:** High-speed node evaluation facilitated by native 64-bit integer bitboards and hardware-mapped bit manipulation (`POPCNT`, LSB bit extraction).
- **Branch Pruning:** Alpha-Beta pruning augmented with Transposition Table move ordering, MVV-LVA, and Late Move Reductions (LMR).

---

## Architecture & Codebase Overview

```
├── agent.py               # Core AI Agent: Bitboard engine, search algorithms, evaluation & heuristics
├── test_fullgame.py       # Game execution harness: Simulates complete matches with time controls
├── log_test_fullgame.py   # Instrumented match runner: Full game tracking, state export, and logging
├── samples.py             # Preconfigured 8x8 board scenarios and piece layouts (sample0 - sample7)
├── opponent.py            # Baseline opponent player implementation for benchmarking
├── extension/             # Framework utilities
│   ├── board_rules.py     # Move timeouts, repetition counters, and terminal condition detectors
│   ├── board_utils.py     # ASCII board visualization and framework-to-agent mapping helpers
│   └── piece_pawn.py      # Custom pawn promotion configurations
└── README.md              # Project documentation
```

---

## Technical Algorithms & Implementation Details

The AI engine is implemented entirely in [`agent.py`](file:///Users/alimuratkeceli/Desktop/Projects/Python/8x8-Chess-Agent/agent.py). It operates via a dedicated bitboard representation isolated from framework overhead during search operations.

```
       Chessmaker Framework Board State
                      │
                      ▼
             board_to_bitboard()
                      │
                      ▼
         ┌─────────────────────────┐
         │   Iterative Deepening   │◄─── Time Management & Move Stability
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │  Minimax + Alpha-Beta   │◄─── Transposition Table (Zobrist Hash)
         │   with LMR Reductions   │◄─── Move Ordering (TT + MVV-LVA)
         └────────────┬────────────┘
                      │ (depth <= 0)
         ┌────────────▼────────────┐
         │    Quiescence Search    │◄─── Stand-Pat & Tactical Check Resolution
         └────────────┬────────────┘
                      │
         ┌────────────▼────────────┐
         │ Static Evaluation (PST) │◄─── Material + Piece-Square Tables + King Safety
         └────────────┬────────────┘
                      │
                      ▼
          bbmove_to_framework_move()
                      │
                      ▼
           Executed Engine Move
```

### 1. 64-Square Bitboard Representation
- **Bitboard Structure (`BitboardState`):** Uses 64-bit integers to represent square occupancy on a standard 8×8 grid ($[0 \dots 63]$).
- **12 Discrete Piece Bitboards:** `WP`, `WN`, `WB`, `WQ`, `WK`, `WR` for White; `BP`, `BN`, `BB`, `BQ`, `BK`, `BR` for Black, along with combined occupancy masks `occ_white`, `occ_black`, and `occ_all`.
- **Fast Bit Manipulation:**
  - `pop_lsb(bb)`: Isolates the least significant bit using `bb & -bb` and computes the index via `.bit_length() - 1`.
  - `iter_bits(bb)`: Fast generator iterating through all active bit indices.
  - `count_bits(bb)`: Leverages Python’s hardware-accelerated `int.bit_count()` (`POPCNT`).
- **Precomputed Attack Tables & Sliding Ray Masks:**
  - Lookup tables for jumping pieces: `KNIGHT_ATTACKS` and `KING_ATTACKS`.
  - Precomputed ray masks: `ROOK_RAYS` and `BISHOP_RAYS` enable early exits when sliding rays contain no blocking pieces.
- **Reverse Attack Generation & Cached Check Detection:**
  - `is_in_check` casts reverse attacks outward from the king's coordinate to determine threats without simulating the opponent's full move generation.
  - Results are memoized with an `@lru_cache` keyed on individual bitboards.

### 2. Zobrist Hashing
- **Incremental State Fingerprinting (`ZobristHasher`):** Precomputes 64-bit random integers for all $(PieceType \times Color \times Square)$ permutations ($6 \times 2 \times 64 = 768$ keys) plus a side-to-move key.
- **$O(1)$ Hash Updates:** Inside `apply_move()`, the board state hash is updated incrementally by XORing out origin squares/captured pieces and XORing in destinations and promotion pieces.

### 3. Iterative Deepening & Root Management
- **Progressive Search (`find_best_move`):** Searches depths $1, 2, 3, \dots, N$ up to `MAX_DEPTH` (default 50) within the allocated time window.
- **Root Mate-in-1 Detection:** Scans immediate moves for forced wins before initiating tree search.
- **Move Stability Early Exit:** Halts deepening early if the identical best move is selected across 4 consecutive depths (starting at depth $\ge 4$).
- **Forced Win Early Exit:** Stops search when a decisive checkmate sequence is confirmed ($\ge$ depth 6).
- **Graceful Timeout Fallback:** Recovers the best move from the last fully completed search depth, or selects the highest-scoring move from the last three completed depths when searching $\ge 7$ plies deep.

### 4. Minimax with Alpha-Beta Pruning
- **Minimax Search (`minimax`):** Explores the game tree to compute optimal moves while bounding the search window with $[\alpha, \beta]$.
- **Late Move Reduction (LMR):**
  - Non-tactical (quiet) moves appearing later in the ordered move list (index $\ge 3$) at search depths $\ge 3$ are searched with a reduced depth (1 to 3 ply reduction).
  - If a reduced-depth move improves $\alpha$ or $\beta$, a full-depth re-search is triggered.
- **Transposition Table Integration:** Probes the table before move expansion; exact scores or scores causing cutoffs terminate search branches immediately.
- **Distance-to-Mate Scoring:** Scales checkmate scores by remaining depth (`mate_bonus = depth * 1000`) to prefer shorter, faster checkmate paths.

### 5. Quiescence Search
- **Tactical Horizon Resolution (`quiescence_search`):** Evaluates capture sequences beyond the nominal search depth up to `QUIESCENCE_MAX_DEPTH = 7` to avoid the horizon effect.
- **Stand-Pat Bounds:** Uses static evaluation as a baseline score; pruning captures that fail to improve alpha/beta.
- **In-Check Escalation:** When the king is in check within quiescence, evasion moves are generated to resolve forced tactical sequences and checkmate states.

### 6. Move Ordering Heuristics
Moves are scored and sorted in-place before branching (`order_moves`, `score_move`):
1. **Transposition Table Best Move:** Prioritized first ($+10,000,000$) to maximize early beta cutoffs.
2. **MVV-LVA (Most Valuable Victim – Least Valuable Attacker):** Captures are prioritized by the value of the victim piece versus the attacker piece (`(victim_value * 10) - attacker_value`).
3. **Quiet Moves:** Neutral scoring for non-capturing moves.

### 7. Transposition Table
- **Memory-Bounded Cache (`BitboardTranspositionTable`):** Stores depth, score, bound type (`TT_EXACT`, `TT_LOWER_BOUND`, `TT_UPPER_BOUND`), and best move hint.
- **Replacement Strategy:** Depth-preferred replacement with FIFO eviction once the configured memory limit (default 64MB) is reached.
- **Bound-Aware Probing:** Validates upper and lower bounds against the current search window.

### 8. Static Position Evaluation
- **Material Evaluation:** Base piece valuations:
  - Pawn: $100$
  - Knight: $330$
  - Bishop: $320$
  - Rook: $500$
  - Queen: $900$
  - King: $20000$
- **Piece-Square Tables (PST):** 8×8 position-specific matrices for each piece type encouraging center dominance, pawn advancement, and piece coordination.
- **Middlegame King Safety:** When more than 13 pieces remain on the board, the evaluator calculates adjacent friendly shield pieces using `KING_ATTACKS` bitboard masks, penalizing exposed or isolated kings.

### 9. Clock Management & Timeout Strategy
- **Time Limits:** Searches with an active time limit per turn (`TIME_LIMIT = 10` seconds) leaving safety margins within the 14-second engine move budget.
- **Tactical Clock Delay (`ENABLE_TIMEOUT_STRATEGY`):** In heavily losing positions (score $\le -600$), the agent can intentionally consume remaining turn time (up to 12.5 seconds) to apply clock pressure in timed matches.

---

## Configuration & Tuning Parameters

Key settings in [`agent.py`](file:///Users/alimuratkeceli/Desktop/Projects/Python/8x8-Chess-Agent/agent.py):

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `MAX_DEPTH` | `50` | Maximum iterative deepening depth limit |
| `QUIESCENCE_MAX_DEPTH` | `7` | Maximum search depth for quiescence capture resolution |
| `TIME_LIMIT` | `10` | Time budget in seconds for search execution per move |
| `ENABLE_TIMEOUT_STRATEGY` | `True` | Enables time-delay strategy in severely losing positions |
| `TIMEOUT_STRATEGY_THRESHOLD` | `-600` | Centipawn deficit threshold required to trigger time delay |
| `TIMEOUT_STRATEGY_DELAY` | `12.5` | Target duration in seconds to hold the clock when delay is active |
| `LOGGING_ENABLED` | `False` | Enables detailed search and statistics logging to `game_log.txt` |

---

## How to Run & Use

### Prerequisites
- Python 3.11+ (chessmaker's `@classmethod @property` piece-naming pattern requires Python ≤3.12; it is broken by the classmethod/property chaining removal in Python 3.13+)
- `chessmaker` framework installed / available in your Python environment

### Running a Match
To execute a game between the agent and an opponent or baseline player, run [`test_fullgame.py`](file:///Users/alimuratkeceli/Desktop/Projects/Python/8x8-Chess-Agent/test_fullgame.py):

```bash
python test_fullgame.py
```

### Configuring Match Settings
In [`test_fullgame.py`](file:///Users/alimuratkeceli/Desktop/Projects/Python/8x8-Chess-Agent/test_fullgame.py):
- **Select Board Setup:** Choose starting positions from `samples.py` (`sample0` through `sample7`):
  ```python
  testgame_timeout(p_white=agent, p_black=opponent, board_sample=sample0)
  ```
- **Agent Self-Play:** Test the agent against itself:
  ```python
  testgame_timeout(p_white=agent, p_black=agent, board_sample=sample1)
  ```

### Running with Full Logging & Analytics
To track per-move board states, search metrics, node statistics, and move histories:

```bash
python log_test_fullgame.py
```
This generates:
- `game_log.txt`: Detailed turn-by-turn game flow, evaluations, and search performance metrics.
- `moves_log.txt`: ASCII board states recorded after each move.
