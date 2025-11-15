"""
Quick test to verify Queen and Right piece move generation is correct.

This test creates simple board positions and verifies that:
1. Queen moves like rook + bishop (8 directions)
2. Right moves like rook + knight (horizontal/vertical + L-shapes)
"""

from helpersBitboard import (
    BitboardState, _get_queen_attacks, _get_right_attacks,
    _get_rook_attacks, _get_bishop_attacks, KNIGHT_ATTACKS,
    square_index, index_to_xy, test_bit, iter_bits
)


def print_attack_board(attacks: int, piece_name: str, sq: int):
    """Pretty-print attack bitboard."""
    x, y = index_to_xy(sq)
    print(f"\n{piece_name} attacks from ({x},{y}):")
    print("  0 1 2 3 4")
    for row in range(5):
        print(f"{row} ", end="")
        for col in range(5):
            sq_idx = square_index(col, row)
            if sq_idx == sq:
                print("X ", end="")  # Piece position
            elif test_bit(attacks, sq_idx):
                print("* ", end="")  # Attacked square
            else:
                print(". ", end="")
        print()


def test_queen_moves():
    """Test that queen moves in all 8 directions (rook + bishop)."""
    print("\n" + "="*60)
    print("TEST 1: Queen Movement (should move in 8 directions)")
    print("="*60)

    # Place queen in center of empty board
    sq = square_index(2, 2)  # Center square
    occupancy = 0  # Empty board

    queen_attacks = _get_queen_attacks(sq, occupancy)
    rook_attacks = _get_rook_attacks(sq, occupancy)
    bishop_attacks = _get_bishop_attacks(sq, occupancy)

    print_attack_board(queen_attacks, "QUEEN", sq)

    # Verify queen = rook + bishop
    expected = rook_attacks | bishop_attacks
    if queen_attacks == expected:
        print("\n✓ PASS: Queen correctly moves like rook + bishop")

        # Count squares
        queen_count = bin(queen_attacks).count('1')
        print(f"  Queen can attack {queen_count} squares from center (expected: 16)")

        if queen_count == 16:  # 4 horizontal + 4 vertical + 4 NE diag + 4 SE diag
            print("✓ PASS: Correct number of squares")
        else:
            print(f"✗ FAIL: Expected 16 squares, got {queen_count}")
    else:
        print("\n✗ FAIL: Queen attacks don't match rook + bishop")
        print(f"  Queen:   {bin(queen_attacks)}")
        print(f"  Expected: {bin(expected)}")


def test_right_moves():
    """Test that right moves like rook + knight."""
    print("\n" + "="*60)
    print("TEST 2: Right Piece Movement (should move like rook + knight)")
    print("="*60)

    # Place right piece in center of empty board
    sq = square_index(2, 2)  # Center square
    occupancy = 0  # Empty board

    right_attacks = _get_right_attacks(sq, occupancy)
    rook_attacks = _get_rook_attacks(sq, occupancy)
    knight_attacks = KNIGHT_ATTACKS[sq]

    print_attack_board(right_attacks, "RIGHT", sq)

    # Verify right = rook + knight
    expected = rook_attacks | knight_attacks
    if right_attacks == expected:
        print("\n✓ PASS: Right correctly moves like rook + knight")

        # Count squares
        right_count = bin(right_attacks).count('1')
        print(f"  Right can attack {right_count} squares from center")
        print(f"  (12 rook squares + 8 knight L-shapes = 20 total)")

        if right_count == 20:
            print("✓ PASS: Correct number of squares")
        else:
            print(f"✗ FAIL: Expected 20 squares, got {right_count}")
    else:
        print("\n✗ FAIL: Right attacks don't match rook + knight")
        print(f"  Right:    {bin(right_attacks)}")
        print(f"  Expected: {bin(expected)}")


def test_queen_with_blockers():
    """Test that queen is blocked by pieces."""
    print("\n" + "="*60)
    print("TEST 3: Queen with Blockers (rays should stop at pieces)")
    print("="*60)

    # Place queen at (2,2), blocker at (2,0) (blocks upward ray)
    sq = square_index(2, 2)
    blocker = square_index(2, 0)
    occupancy = 1 << blocker

    queen_attacks = _get_queen_attacks(sq, occupancy)

    print_attack_board(queen_attacks, "QUEEN (with blocker at 2,0)", sq)

    # Queen should attack (2,0) but not beyond
    blocked_sq = square_index(2, -1)  # Invalid, but conceptually beyond blocker

    # Verify blocker is attacked
    if test_bit(queen_attacks, blocker):
        print("\n✓ PASS: Queen can capture blocker at (2,0)")
    else:
        print("\n✗ FAIL: Queen should attack blocker position")

    # Count total attacks (should be less than 16 due to blocker)
    count = bin(queen_attacks).count('1')
    print(f"  Queen attacks {count} squares (less than 16 due to blocker)")
    if count < 16:
        print("✓ PASS: Blocker correctly limits queen movement")
    else:
        print("✗ FAIL: Blocker didn't limit movement")


def test_right_with_blockers():
    """Test that right piece rook-like moves are blocked, but knight moves jump."""
    print("\n" + "="*60)
    print("TEST 4: Right with Blockers (rook moves blocked, knight jumps)")
    print("="*60)

    # Place right at (2,2), blocker at (2,0)
    sq = square_index(2, 2)
    blocker = square_index(2, 0)
    occupancy = 1 << blocker

    right_attacks = _get_right_attacks(sq, occupancy)

    print_attack_board(right_attacks, "RIGHT (with blocker at 2,0)", sq)

    # Verify rook-like moves are blocked
    if test_bit(right_attacks, blocker):
        print("\n✓ PASS: Right can attack blocker (rook-like move)")
    else:
        print("\n✗ FAIL: Right should attack blocker with rook move")

    # Verify knight moves still work (can jump over blocker)
    knight_sq = square_index(1, 0)  # Knight L-shape from (2,2)
    if test_bit(right_attacks, knight_sq):
        print("✓ PASS: Right can still jump like knight (ignores blocker)")
    else:
        # Check if this square is actually a valid knight move
        if test_bit(KNIGHT_ATTACKS[sq], knight_sq):
            print(f"✗ FAIL: Right should be able to jump to ({index_to_xy(knight_sq)})")


def run_all_tests():
    """Run all piece movement tests."""
    print("\n" + "#"*60)
    print("# QUEEN AND RIGHT PIECE MOVEMENT TESTS")
    print("#"*60)

    test_queen_moves()
    test_right_moves()
    test_queen_with_blockers()
    test_right_with_blockers()

    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)
    print("\nIf all tests show ✓ PASS, the piece movements are correct!")
    print()


if __name__ == "__main__":
    run_all_tests()
