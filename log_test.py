# Python 3.11+
import sys
from itertools import cycle
from chessmaker.chess.base import Board
from extension.board_utils import print_board_ascii, copy_piece_move
from extension.board_rules import get_result
from samples import white, black, sample0, sample1
from agent import agent, log_message, log_board_state, init_log_file
from opponent import opponent
from agentS import agent, log_message, log_board_state, init_log_file
from agent_logless import agent as agent_logless
from agentQ import agent as agentQ

# Global move counter for logging
global_move_counter = 0

# Global file handle for moves log
MOVES_LOG_FILE = None

def init_moves_log():
    """Initialize the moves log file."""
    global MOVES_LOG_FILE
    MOVES_LOG_FILE = open("moves_log.txt", "w")
    MOVES_LOG_FILE.write("=== MOVES LOG ===\n\n")

def close_moves_log():
    """Close the moves log file."""
    global MOVES_LOG_FILE
    if MOVES_LOG_FILE:
        MOVES_LOG_FILE.close()
        MOVES_LOG_FILE = None

def log_board_to_moves_file(board, title=""):
    """Write the current board state to moves_log.txt."""
    global MOVES_LOG_FILE
    if not MOVES_LOG_FILE:
        return
    
    if title:
        MOVES_LOG_FILE.write(f"\n{title}\n")
    
    # Write board representation
    MOVES_LOG_FILE.write("  0 1 2 3 4\n")
    for y in range(5):
        MOVES_LOG_FILE.write(f"{y} ")
        for x in range(5):
            piece = None
            for p in board.get_pieces():
                if p.position.x == x and p.position.y == y:
                    piece = p
                    break
            if piece:
                # Map piece names to their correct symbols
                piece_name_lower = piece.name.lower()
                if piece_name_lower == "knight":
                    symbol = 'n'
                elif piece_name_lower == "king":
                    symbol = 'k'
                elif piece_name_lower == "queen":
                    symbol = 'q'
                elif piece_name_lower == "bishop":
                    symbol = 'b'
                elif piece_name_lower == "right":
                    symbol = 'r'
                elif piece_name_lower == "pawn":
                    symbol = 'p'
                else:
                    symbol = piece.name[0].lower()
                
                # Uppercase for white pieces
                if piece.player.name == "white":
                    symbol = symbol.upper()
                MOVES_LOG_FILE.write(f"{symbol} ")
            else:
                MOVES_LOG_FILE.write(". ")
        MOVES_LOG_FILE.write("\n")
    MOVES_LOG_FILE.write("\n")
    MOVES_LOG_FILE.flush()

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

    global global_move_counter
    global_move_counter = 0
    
    # Initialize the log files at the start of the game
    init_log_file()
    init_moves_log()

    board, players = make_custom_board(board_sample)
    turn_order = cycle(players)
    var = None
    print("=== Initial position ===")
    print_board_ascii(board)
    
    # Log initial board state to both files
    log_message("="*60)
    log_message("GAME START - Initial Position")
    log_message("="*60)
    log_board_state(board, "Initial Board State:")
    
    log_board_to_moves_file(board, "="*60 + "\nGAME START - Initial Position\n" + "="*60)
    
    while True:
        try:
            player = next(turn_order)
            
            # Log before the move
            log_message(f"\n{'='*60}")
            log_message(f"MOVE {global_move_counter} - {player.name.upper()}'s turn")
            log_message(f"{'='*60}")
            log_board_state(board, "Current Board State:")
            
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
                    move_desc = f"{piece} move to: ({move_opt.position.x},{move_opt.position.y})"
                    print(move_desc)
                    log_message(f"\n*** MOVE EXECUTED: {piece.name} from ({p_piece.position.x},{p_piece.position.y}) to ({move_opt.position.x},{move_opt.position.y}) ***")
                    
                    if getattr(move_opt, "captures", None):
                        caps = ", ".join(f"({c.x},{c.y})" for c in move_opt.captures)
                        if caps:
                            capture_desc = f"{piece} captures at: {caps}"
                            print(capture_desc)
                            log_message(f"*** CAPTURES at: {caps} ***")
                    
                    # Log board state after move to moves_log.txt
                    log_board_to_moves_file(board, f"Move {global_move_counter} - {player.name.upper()}: {piece.name} from ({p_piece.position.x},{p_piece.position.y}) to ({move_opt.position.x},{move_opt.position.y})")
                    
                    global_move_counter += 1
                except Exception:
                    print(f"=== Game ended: {player.name} can not make a legal move ===")
                    log_message(f"\n=== Game ended: {player.name} can not make a legal move ===")
                    break

            print_board_ascii(board)
            res = get_result(board)
            if res:
                print(f"=== Game ended: {res} ===")
                log_message(f"\n{'='*60}")
                log_message(f"=== GAME ENDED: {res} ===")
                log_message(f"{'='*60}")
                log_board_state(board, "Final Board State:")
                
                # Log final state to moves_log.txt
                log_board_to_moves_file(board, f"{'='*60}\nGAME ENDED: {res}\n{'='*60}\nFinal Board State:")
                close_moves_log()
                break

        except KeyboardInterrupt:
            print(f"=== Game ended by keyboard interuption ===")
            log_message(f"\n=== Game ended by keyboard interruption ===")
            close_moves_log()
            sys.exit()

if __name__ == "__main__":
    testgame(p_white=agentQ, p_black=agent, board_sample=sample0)