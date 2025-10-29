# COMP2321: Coursework Overview – “CHESS FRAGMENTS”
*(Parts 1–5 only — up to but not including Section 6: Assessment)*

This game consists of two players: **Agent** (your designed algorithm) and **Opponent** (predefined baseline algorithms).  
Agent and Opponent compete on a **5×5 grid** in a turn-based strategy game inspired by chess.

---

## 1) Rule

The board consists of several pieces chosen from the set:
`{Right, Knight, Bishop, Queen, King, Pawn}`

Each piece has its own movement rules:

- **Right**: Moves any number of squares horizontally or vertically. It can also move in an **L-shape** (two squares in one direction, then one square perpendicular), and it **can jump over other pieces**.
- **Knight**: Moves in an **L-shape** (two in one direction, then one perpendicular), **jumping over other pieces**.
- **Bishop**: Moves any number of squares **diagonally**.
- **King**: Moves **one square** in any direction.
- **Queen**: Moves any number of squares **diagonally, horizontally, or vertically**.
- **Pawn**: Moves **one square forward**. A pawn **captures** by moving **one square diagonally forward** to an occupied square. When a pawn reaches the opponent’s **back rank**, it is **promoted to a Queen**.

**Setup notes**
- The **King** is always present on the board.
- Other pieces may or may not appear, depending on the board setup.
- In each game, Agent may play **white** and Opponent **black**, or vice versa.

---

## 2) Example of Piece Movements

Pieces are represented by single characters (uppercase for **white**, lowercase for **black**):

| Piece  | White/Black | Movement summary |
|--------|-------------|------------------|
| Right  | `R` / `r`   | Any squares horizontally/vertically; also L-shape; can jump |
| Knight | `N` / `n`   | L-shape; can jump |
| Bishop | `B` / `b`   | Any squares diagonally |
| King   | `K` / `k`   | One square in any direction |
| Queen  | `Q` / `q`   | Any squares diagonally/horizontally/vertically |
| Pawn   | `P` / `p`   | Forward 1 (first move may be 1 or 2 if clear); captures one square diagonally forward; promotes on back rank |

**Pawn movement details**
- **First move**: may move **1 or 2** squares forward if the path is clear.
- **Subsequent moves**: **1** square forward.
- **Captures**: one square **diagonally forward** to the occupied square.
- **Promotion**: upon reaching the opponent’s **back rank**, the pawn becomes a **Queen**.

---

## 3) Board Presentation in the Python Environment

In the coursework support package, the game board is represented as a **2D matrix** with shape **(5, 5)**.  
Each cell contains a single character denoting a piece (uppercase for white, lowercase for black) or is empty.

**Example matrix view**

```
  0 1 2 3 4
0 r b k q n
1 p p p p p
2 . . . . .
3 P P P P P
4 N Q K B R
```

> This is a conceptual illustration of a **5×5** board.  
> The provided support package includes the exact data structures and helpers.

---

## 4) Win, Lose, or Draw

### Win / Lose
The game ends in a **win** or **loss** if any of the following occurs:
- **Checkmate**: a player’s **King** is attacked with **no legal escape**.
- **Stalemate**: it is a player’s turn but they have **no legal moves** available.
- **Thinking timeout**: a player **fails to choose a move** within the predefined time limit (that player **loses**).
- **Illegal move**: a player makes an **invalid move** (that player **loses**).

### Draw
A **draw** occurs under any of the following:
- Only **two Kings** remain on the board.
- **Fivefold repetition**: the **exact board position** occurs **five times** (not necessarily consecutively).
- **Game timeout**: neither player achieves a win/loss **within the overall time limit**.

---

## 5) Preparing Your Agent (`agent.py`) for Submission

You must design an **Agent** (in `agent.py`) following the required format.  
Initial board positions and the **Opponent** are predefined; in Moodle there are multiple **difficulty levels** and **board setups**.  
Design your Agent to be **as strong and efficient as possible** across different settings.

**Recommended approach**
- Start with a simple baseline to validate the interface and rules.
- Progressively add stronger decision-making (e.g., **minimax**, **alpha–beta pruning**, improved evaluation).

**Performance & timing**
- Track **computational cost** (e.g., **nodes expanded**) and respect the **thinking time limit**.
- Control search by adjusting **depth**, using **pruning**, and other **optimizations** to avoid timeouts.

**Support package**
- Includes a **basic Agent** and **Opponent** for reference and testing the required format.
- The sample Agent is **not competitive**; you must implement your own and save it as:
  - `agent1.py` (final submission format).
