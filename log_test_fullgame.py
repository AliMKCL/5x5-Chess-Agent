# Python 3.11+
import sys, time
from itertools import cycle
from chessmaker.chess.base import Board
from extension.board_utils import print_board_ascii, copy_piece_move
from extension.board_rules import get_result, thinking_with_timeout, THINKING_TIME_BUDGET, GAME_TIME_BUDGET
from samples import white, black, sample0, sample1, sample2, sample3
import old_agents.agentS as agentS
import old_agents.agentQ as agentQ
import agentE
import agentT
from opponent import opponent
import agentBitboard as agentB
import agentP 
import agentFinal as agentF
import agentBitboard_optimized as agentNUMBERS
import agentBB_initial as agentBI
import agentBitboard_gemini as agentBG

# Global move counter for logging
global_move_counter = 0

# Global file handle for moves log
MOVES_LOG_FILE = None
# Global file handle for game log (shared across agents)
GAME_LOG_FILE = None

def log_message(message):
    """Write a message to the shared game log file."""
    global GAME_LOG_FILE
    if GAME_LOG_FILE:
        GAME_LOG_FILE.write(message + "\n")
        GAME_LOG_FILE.flush()

def log_board_state(board, title=""):
    """Log the current board state to the shared game log file."""
    global GAME_LOG_FILE
    if not GAME_LOG_FILE:
        return
    
    if title:
        GAME_LOG_FILE.write(f"\n{title}\n")
    
    # Write board representation
    GAME_LOG_FILE.write("  0 1 2 3 4\n")
    for y in range(5):
        GAME_LOG_FILE.write(f"{y} ")
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
                GAME_LOG_FILE.write(f"{symbol} ")
            else:
                GAME_LOG_FILE.write(". ")
        GAME_LOG_FILE.write("\n")
    GAME_LOG_FILE.write("\n")
    GAME_LOG_FILE.flush()

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

def close_all_logs():
    """Close both moves log and game log files."""
    global MOVES_LOG_FILE, GAME_LOG_FILE
    if MOVES_LOG_FILE:
        MOVES_LOG_FILE.close()
        MOVES_LOG_FILE = None
    if GAME_LOG_FILE:
        GAME_LOG_FILE.close()
        GAME_LOG_FILE = None
        # Clear agent log file handles
        agentS.LOG_FILE = None
        agentQ.LOG_FILE = None
        agentB.LOG_FILE = None

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

def testgame_timeout(p_white, p_black, board_sample):

    global global_move_counter, GAME_LOG_FILE
    global_move_counter = 0
    
    # Initialize the log files at the start of the game
    # CRITICAL: Create a shared log file handle for ALL agents
    GAME_LOG_FILE = open("game_log.txt", "w")
    GAME_LOG_FILE.write("=== GAME LOG (Multi-Agent) ===\n\n")
    GAME_LOG_FILE.flush()
    
    # Share the same file handle across all agent modules
    agentS.LOG_FILE = GAME_LOG_FILE
    agentQ.LOG_FILE = GAME_LOG_FILE
    agentE.LOG_FILE = GAME_LOG_FILE
    agentT.LOG_FILE = GAME_LOG_FILE
    agentB.LOG_FILE = GAME_LOG_FILE
    agentP.LOG_FILE = GAME_LOG_FILE
    agentF.LOG_FILE = GAME_LOG_FILE
    
    init_moves_log()

    board, players = make_custom_board(board_sample)
    turn_order = cycle(players)
    print(f"Time budget for each move {THINKING_TIME_BUDGET} seconds")
    print(f"Time budget for the game: {GAME_TIME_BUDGET} seconds")
    print("=== Initial position ===")
    print_board_ascii(board)
    
    # Log initial board state to both files
    print("DEBUG: About to log GAME START")  # DEBUG
    log_message("="*60)
    log_message("GAME START - Initial Position")
    log_message("="*60)
    log_board_state(board, "Initial Board State:")
    print("DEBUG: Finished logging initial board")  # DEBUG
    
    log_board_to_moves_file(board, "="*60 + "\nGAME START - Initial Position\n" + "="*60)
    
    t_start = time.perf_counter()
    t_game = t_start + GAME_TIME_BUDGET
    ply = 1
    while True:
        try:
            # Checking game timeout
            now = time.perf_counter()
            if now > t_game:
                print("=== Game ended: Draw - game timeout ===")
                log_message(f"\n{'='*60}")
                log_message(f"=== GAME ENDED: Draw - game timeout ===")
                log_message(f"{'='*60}")
                log_board_to_moves_file(board, f"{'='*60}\nGAME ENDED: Draw - game timeout\n{'='*60}")
                close_all_logs()
                break
            
            player = next(turn_order)
            
            # Log before the move
            log_message(f"\n{'='*60}")
            log_message(f"MOVE {global_move_counter} - {player.name.upper()}'s turn")
            log_message(f"{'='*60}")
            log_board_state(board, "Current Board State:")
            
            temp_board = board.clone()
            if player.name == "white":
                p_piece, p_move_opt = thinking_with_timeout(func=p_white, thinking_time=THINKING_TIME_BUDGET, board=temp_board, player=player, var=[ply, THINKING_TIME_BUDGET])
                board, piece, move_opt = copy_piece_move(board, p_piece, p_move_opt)
            else:
                p_piece, p_move_opt = thinking_with_timeout(func=p_black, thinking_time=THINKING_TIME_BUDGET, board=temp_board, player=player, var=[ply, THINKING_TIME_BUDGET])
                board, piece, move_opt = copy_piece_move(board, p_piece, p_move_opt)

            # Checking game timeout
            now = time.perf_counter()
            if now > t_game:
                print("=== Game ended: Draw - game timeout ===")
                log_message(f"\n{'='*60}")
                log_message(f"=== GAME ENDED: Draw - game timeout ===")
                log_message(f"{'='*60}")
                log_board_to_moves_file(board, f"{'='*60}\nGAME ENDED: Draw - game timeout\n{'='*60}")
                close_all_logs()
                break
            
            if (not piece) or (not move_opt):
                res = get_result(board)
                if res:
                    print(f"=== Game ended: {res} ===")
                    log_message(f"\n{'='*60}")
                    log_message(f"=== GAME ENDED: {res} ===")
                    log_message(f"{'='*60}")
                    log_board_state(board, "Final Board State:")
                    log_board_to_moves_file(board, f"{'='*60}\nGAME ENDED: {res}\n{'='*60}\nFinal Board State:")
                    close_all_logs()
                elif p_piece == 99:
                    print(f"=== Game ended: {player.name} thinking time out ===")
                    log_message(f"\n=== Game ended: {player.name} thinking time out ===")
                    close_all_logs()
                else:
                    print(f"=== Game ended: {player.name} can not make a legal move ===")
                    log_message(f"\n=== Game ended: {player.name} can not make a legal move ===")
                    close_all_logs()
                break

            else:
                try:
                    piece.move(move_opt)
                    ply = ply + 1
                    move_desc = f"{piece} move to: ({move_opt.position.x},{move_opt.position.y})"
                    print(move_desc)
                    log_message(f"\n*** MOVE EXECUTED: {piece.name} from ({p_piece.position.x},{p_piece.position.y}) to ({move_opt.position.x},{move_opt.position.y}) ***")
                    
                    if getattr(move_opt, "captures", None):
                        caps = ", ".join(f"({c.x},{c.y})" for c in move_opt.captures)
                        if caps:
                            capture_desc = f"{piece} captures at: {caps}"
                            print(capture_desc)
                            log_message(f"*** CAPTURES at: {caps} ***")
                    
                    # Log board state after move to both game_log.txt and moves_log.txt
                    log_message("")  # Add blank line for readability
                    log_board_state(board, f"Board State After Move {global_move_counter}:")
                    
                    log_board_to_moves_file(board, f"Move {global_move_counter} - {player.name.upper()}: {piece.name} from ({p_piece.position.x},{p_piece.position.y}) to ({move_opt.position.x},{move_opt.position.y})")
                    
                    global_move_counter += 1
                except Exception:
                    print(f"=== Game ended: {player.name} can not make a legal move ===")
                    log_message(f"\n=== Game ended: {player.name} can not make a legal move ===")
                    close_all_logs()
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
                close_all_logs()
                break

        except KeyboardInterrupt:
            print(f"=== Game ended by keyboard interuption ===")
            log_message(f"\n=== Game ended by keyboard interruption ===")
            close_all_logs()
            sys.exit()

if __name__ == "__main__":
    testgame_timeout(p_white=agentNUMBERS.agent, p_black=agentF.agent, board_sample=sample1)