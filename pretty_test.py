# Python 3.11+
import sys
from itertools import cycle
from chessmaker.chess.base import Board
from extension.board_utils import print_board_ascii, copy_piece_move
from extension.board_rules import get_result
from samples import white, black, sample0, sample1
from agent import agent
from opponent import opponent

def write_board_to_file(board, move_number, filename="game.txt"):
    """
    Writes the current board state to a file after each move.
    
    Args:
        board: The current board state
        move_number: The current move number
        filename: The output filename (default: "game.txt")
    """
    with open(filename, "a") as f:
        f.write(f"{move_number}\n")
        # Capture the board output
        import io
        from contextlib import redirect_stdout
        
        board_output = io.StringIO()
        with redirect_stdout(board_output):
            print_board_ascii(board)
        
        f.write(board_output.getvalue())
        f.write("\n")

def make_custom_board(board_sample):
    # player1: white vs player2: black
    players = [white, black]
    board = Board(
    squares = board_sample,
    players=players,
        turn_iterator=cycle(players),
    )
    return board, players

def testgame(p_white, p_black, board_sample):

    board, players = make_custom_board(board_sample)
    turn_order = cycle(players)
    var = None
    move_number = 0
    
    # Clear the game file at the start
    with open("game.txt", "w") as f:
        f.write("")
    
    print("=== Initial position ===")
    print_board_ascii(board)
    write_board_to_file(board, move_number)
    
    while True:
        try:
            move_number += 1
            player = next(turn_order)
            temp_board = board.clone()
            if player.name == "white":
                p_piece, p_move_opt = p_white(temp_board, player, var)
                board, piece, move_opt = copy_piece_move(board, p_piece, p_move_opt)
            else:
                p_piece, p_move_opt = p_black(temp_board, player, var)
                board, piece, move_opt = copy_piece_move(board, p_piece, p_move_opt)

            if (not piece) or (not move_opt):
                res = get_result(board)
                if res:
                    print(f"=== Game ended: {res} ===")
                else:
                    print(f"=== Game ended: {player.name} can not make a legal move ===")
                break

            else:
                try:
                    piece.move(move_opt)
                    print(f"{piece} move to: ({move_opt.position.x},{move_opt.position.y})")
                    if getattr(move_opt, "captures", None):
                        caps = ", ".join(f"({c.x},{c.y})" for c in move_opt.captures)
                        if caps:
                            print(f"{piece} captures at: {caps}")
                except Exception:
                    print(f"=== Game ended: {player.name} can not make a legal move ===")
                    break

            print_board_ascii(board)
            write_board_to_file(board, move_number)
            res = get_result(board)
            if res:
                print(f"=== Game ended: {res} ===")
                break

        except KeyboardInterrupt:
            print(f"=== Game ended by keyboard interuption ===")
            sys.exit()

if __name__ == "__main__":
    testgame(p_white=agent, p_black=agent, board_sample=sample0)