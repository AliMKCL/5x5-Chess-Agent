# Project Overview

This project implements a chess-playing agent for a chess-like game called "Chess Fragments". The agent is implemented in the `agent4.py` file and uses the minimax algorithm with alpha-beta pruning to play the game.

The core of the project is the `agent4.py` file, which contains a sophisticated chess engine with the following key components:

*   **Minimax with Alpha-Beta Pruning:** The agent uses a minimax search algorithm with alpha-beta pruning to explore the game tree and find the optimal move.
*   **Iterative Deepening:** The search is performed using iterative deepening, which allows the agent to find a good move within a given time limit.
*   **Move Ordering:** The agent uses move ordering to improve the efficiency of the alpha-beta pruning.
*   **Evaluation Function:** A sophisticated evaluation function is used to estimate the value of a given board position. The evaluation function considers material balance, positional advantages, and king safety.
*   **Logging:** The agent includes a logging functionality to record the game progress and the agent's decision-making process.

# Game Rules

This is a turn-based strategy game played on a 5x5 grid. The game is inspired by chess, but with some unique rules and pieces.

## Pieces and Movement

The game features the following pieces with unique movements:

*   **Right (R/r):** Moves any number of squares horizontally or vertically. It can also move in an L-shape (two squares in one direction, then one square perpendicular), and it can jump over other pieces.
*   **Knight (N/n):** Moves in an L-shape (two in one direction, then one perpendicular), jumping over other pieces.
*   **Bishop (B/b):** Moves any number of squares diagonally.
*   **King (K/k):** Moves one square in any direction.
*   **Queen (Q/q):** Moves any number of squares diagonally, horizontally, or vertically.
*   **Pawn (P/p):** Moves one square forward. A pawn captures by moving one square diagonally forward to an occupied square. When a pawn reaches the opponent’s back rank, it is promoted to a Queen.

## Win, Lose, or Draw

*   **Win/Lose:** A player wins if they checkmate the opponent's king, or if the opponent runs out of time or makes an illegal move. A player loses if their king is checkmated, or if they run out of time or make an illegal move.
*   **Draw:** A draw occurs if only two kings remain on the board, if a board position is repeated five times, or if the game times out.

# Code Solution

The `agent4.py` file contains the implementation of the chess-playing agent. The agent uses a combination of advanced techniques to play the game at a high level.

## Minimax with Alpha-Beta Pruning

The agent uses the minimax algorithm with alpha-beta pruning to search the game tree for the best move. The `minimax()` function in `agent4.py` implements this algorithm. The function takes the board state, the search depth, the alpha and beta values, and the player to move as input. It returns the evaluation of the board state.

## Iterative Deepening

The `find_best_move()` function uses iterative deepening to search the game tree. This allows the agent to find a good move within a given time limit. The function starts with a search depth of 1 and increases the depth until the time limit is reached. This ensures that the agent always has a move to play, even if it doesn't have enough time to complete a deep search.

## Move Ordering

The `order_moves()` function is used to order the moves before they are evaluated by the minimax algorithm. This improves the efficiency of the alpha-beta pruning by exploring the most promising moves first. The function prioritizes checkmate moves, valuable captures, and positional improvements.

## Evaluation Function

The `evaluate_board()` function is used to evaluate the board state. The function takes the board state and the player to move as input and returns a score that represents the evaluation of the board. The evaluation function considers the following factors:

*   **Material balance:** The value of the pieces on the board.
*   **Positional values:** The value of the position of the pieces on the board, based on piece-square tables.
*   **King safety:** A bonus is given if the king is well-protected by friendly pieces.

# Development Conventions

*   The main agent logic is implemented in the `agent()` function in `agent4.py`.
*   The `find_best_move()` function implements the iterative deepening search.
*   The `minimax()` function implements the minimax algorithm with alpha-beta pruning.
*   The `evaluate_board()` function implements the board evaluation logic.
*   The `order_moves()` function implements the move ordering logic.