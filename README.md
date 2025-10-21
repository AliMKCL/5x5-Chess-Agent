# COMP2321: Coursework “CHESS FRAGMENTS”



## 1. Environment Setup
### 1.1 Using Conda 
We prefer you to use [conda](https://anaconda.org/anaconda/conda) for installing a virtual environment with python 3.12.9
```
conda create -n comp2321 python=3.12.9
conda activate comp2321
```

Then install [chessmaker](https://pypi.org/project/chessmaker/) package 

```
(comp2321) pip install chessmaker
```
### 1.2 Without using Conda
It is also fine to set up the coursework without conda. If you do, make sure your environment includes:

* Python 3.12.9 (preferred for compatibility, though Python ≥ 3.11 should also work)  
* The [chessmaker](https://pypi.org/project/chessmaker/) package, installed via `pip install chessmaker`

### 1.3 CW-COMP2321 package:
From Moodle, download the CW supporting package CW-COMP2321.zip and extract it

---

## 2. File structure

```md
<CW-COMP2321>
├── extension
│   ├── board_rules.py
│   ├── board_utils.py
│   ├── piece_pawn.py
│   └── piece_right.py
├── agent.py
├── opponent.py
├── samples.py
├── test.py
└── README.md
```
---
## 3. Running a Test Game

1. Navigate to the package directory:

   ```
   cd CW-COMP2321/
   ```

2. Configure the game setup by choosing the white player, black player, and initial board. For example:

   ```python
   if __name__ == "__main__":
       testgame(p_white=agent, p_black=opponent, board_sample=sample0)
   ```

3. Run the game. Once the game finishes, the result will be displayed in the terminal. For example:

   ```
   === Game ended: Checkmate - black loses ===
   ```

4. If you want to interrupt the game partway through, press `Ctrl+C`:

   ```
   === Game ended by keyboard interruption ===
   ```

---

## 4. The `extension` Directory

- You are **not expected to modify any files here**. You may look at them to better understand the game setup, but any changes will have no effect on the version we run on Moodle.

- `board_utils.py`: contains several useful functions, such as printing the board to the terminal, checking legal moves, and creating copies of pieces with their corresponding moves on a cloned board.  
  If you want to use these functions in `agent.py`, make sure you import them correctly. For example:

  ```python
  from extension.board_utils import list_legal_moves_for
  ```

---

## 5. The Root Directory

You can safely modify the following Python files without affecting the game setup:

- **`agent.py`**: provides a very simple example of an Agent. This implementation is **only illustrative** and is not strong enough to compete against our assessment system. You must complete your own Agent and save it in the same format.  
  **Important:** in `agent.py`, you may import functions from `board_utils.py` (as shown in Section 4 above), if you want to create another function to support your Agent, you should define it and also put it in `agent.py`. 
  
- **`opponent.py`**: provides a very simple example of an Opponent. Again, this is for illustration only and will not be competitive in the assessment system. You must complete your own Agent and save it in the same format.  
  **Important:** similar to `agent.py`, in  `opponent.py`, you may import functions from `board_utils.py` (as shown in Section 4 above), if you want to create another function to support your Agent, you should define it and also put it in `opponent.py`. 

- **`samples.py`**: defines example board setups. You are encouraged to create additional board samples in the same format to test your Agent under different conditions.

- **`test.py`**: the main Python file for running the game. **Important:** only modify the white player, black player, and initial board. Do not try to change the player turns or the game logic. Such changes will have no effect on the version we run on Moodle.

---

## 6. Further Information about Chessmaker

You may wish to implement additional functions to support your agent in this game. For more details, please refer to the following resources:

Documentation: https://wolfdwyc.github.io/ChessMaker

Source Code: https://github.com/WolfDWyc/ChessMaker