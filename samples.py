from chessmaker.chess.base import Player
from chessmaker.chess.pieces import King, Bishop, Knight, Queen, Rook
from extension.piece_pawn import Pawn_Q
from chessmaker.chess.base import Square

white = Player("white")
black = Player("black")

# Standard 8x8 chess starting position.
# Back rank order: Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook.
sample0 = [
    [Square(Rook(black)), Square(Knight(black)), Square(Bishop(black)), Square(Queen(black)), Square(King(black)), Square(Bishop(black)), Square(Knight(black)), Square(Rook(black))],
    [Square(Pawn_Q(black)), Square(Pawn_Q(black)), Square(Pawn_Q(black)), Square(Pawn_Q(black)), Square(Pawn_Q(black)), Square(Pawn_Q(black)), Square(Pawn_Q(black)), Square(Pawn_Q(black))],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(Pawn_Q(white)), Square(Pawn_Q(white)), Square(Pawn_Q(white)), Square(Pawn_Q(white)), Square(Pawn_Q(white)), Square(Pawn_Q(white)), Square(Pawn_Q(white)), Square(Pawn_Q(white))],
    [Square(Rook(white)), Square(Knight(white)), Square(Bishop(white)), Square(Queen(white)), Square(King(white)), Square(Bishop(white)), Square(Knight(white)), Square(Rook(white))],
    ]

# Same standard starting position (kept as a distinct name for API parity with
# code that selects sample1 by default; there is only one "correct" standard
# chess starting arrangement).
sample1 = [
    [Square(Rook(black)), Square(Knight(black)), Square(Bishop(black)), Square(Queen(black)), Square(King(black)), Square(Bishop(black)), Square(Knight(black)), Square(Rook(black))],
    [Square(Pawn_Q(black)), Square(Pawn_Q(black)), Square(Pawn_Q(black)), Square(Pawn_Q(black)), Square(Pawn_Q(black)), Square(Pawn_Q(black)), Square(Pawn_Q(black)), Square(Pawn_Q(black))],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(Pawn_Q(white)), Square(Pawn_Q(white)), Square(Pawn_Q(white)), Square(Pawn_Q(white)), Square(Pawn_Q(white)), Square(Pawn_Q(white)), Square(Pawn_Q(white)), Square(Pawn_Q(white))],
    [Square(Rook(white)), Square(Knight(white)), Square(Bishop(white)), Square(Queen(white)), Square(King(white)), Square(Bishop(white)), Square(Knight(white)), Square(Rook(white))],
    ]

# King + Pawn vs lone King endgame: white king escorts its pawn toward promotion.
sample2 = [
    [Square(King(black)), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(Pawn_Q(white)), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(King(white)), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
]

# King + Rook vs lone King checkmate technique.
sample3 = [
    [Square(), Square(), Square(), Square(), Square(King(black)), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(King(white)), Square(), Square(), Square(), Square(Rook(white))],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
]

# King + Queen (black) vs lone King (white) checkmate technique.
sample4 = [
    [Square(), Square(), Square(), Square(), Square(King(black)), Square(), Square(), Square()],
    [Square(Queen(black)), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(King(white)), Square(), Square(), Square(), Square()],
]

# King and pawn endgame: kingside pawn race with both sides holding a passed pawn.
sample5 = [
    [Square(King(black)), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square(Pawn_Q(black))],
    [Square(King(white)), Square(), Square(), Square(), Square(), Square(), Square(), Square(Pawn_Q(white))],
    [Square(), Square(), Square(), Square(Pawn_Q(white)), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
]

# King and pawn endgame with a black bishop supporting, white up on pawns.
sample6 = [
    [Square(King(black)), Square(Bishop(black)), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square(Pawn_Q(black))],
    [Square(King(white)), Square(), Square(), Square(), Square(), Square(), Square(), Square(Pawn_Q(white))],
    [Square(), Square(Pawn_Q(white)), Square(), Square(Pawn_Q(white)), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
]

# Pawns on both flanks with kings centralizing.
sample7 = [
    [Square(King(black)), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(Pawn_Q(black)), Square(), Square(), Square(), Square(), Square(), Square(), Square(Pawn_Q(black))],
    [Square(Pawn_Q(white)), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(King(white)), Square(Pawn_Q(white)), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
    [Square(), Square(), Square(), Square(), Square(), Square(), Square(), Square()],
]
