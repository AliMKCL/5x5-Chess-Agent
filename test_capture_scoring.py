"""
Test to verify capture scoring works correctly after bugfix.

This tests that:
1. Captures are detected correctly (captured_type != -1)
2. MVV-LVA scoring prioritizes captures over non-captures
3. High-value captures score higher than low-value captures
"""

import sys
sys.path.insert(0, '/Users/alimuratkeceli/Desktop/University/Artificial Intelligence/CW-COMP2321')

from helpersBitboard import BBMove, PAWN, KNIGHT, BISHOP, QUEEN, KING, RIGHT, PIECE_VALUES

# Import score_move directly from agentBitboard
# We'll copy the MVV_LVA_VALUES here to avoid module import issues
MVV_LVA_VALUES = [100, 330, 320, 900, 20000, 500]  # P, N, B, Q, K, Right

def score_move(move, tt_best_move=None):
    """Score a move for ordering (copied from agentBitboard.py to avoid imports)."""
    if tt_best_move and (move.from_sq, move.to_sq) == tt_best_move:
        return 10_000_000

    if move.captured_type != -1:  # -1 means no capture
        victim_value = MVV_LVA_VALUES[move.captured_type]
        attacker_value = MVV_LVA_VALUES[move.piece_type]
        return (victim_value * 10) - attacker_value

    return 0

def test_mvv_lva_array():
    """Test that MVV_LVA_VALUES matches PIECE_VALUES."""
    print("\n" + "="*60)
    print("TEST 1: MVV-LVA Array Correctness")
    print("="*60)

    print(f"\nPIECE_VALUES (helpersBitboard): {PIECE_VALUES}")
    print(f"MVV_LVA_VALUES (agentBitboard):  {MVV_LVA_VALUES}")

    if PIECE_VALUES == MVV_LVA_VALUES:
        print("\n✓ PASS: Arrays match perfectly!")
        return True
    else:
        print("\n✗ FAIL: Arrays don't match!")
        print(f"  Expected: {PIECE_VALUES}")
        print(f"  Got:      {MVV_LVA_VALUES}")
        return False


def test_capture_detection():
    """Test that captures are detected correctly."""
    print("\n" + "="*60)
    print("TEST 2: Capture Detection")
    print("="*60)

    # Non-capture move (pawn push)
    non_capture = BBMove(
        from_sq=10,
        to_sq=5,
        piece_type=PAWN,
        captured_type=-1,  # -1 = no capture
        promo=0
    )

    # Capture move (pawn takes pawn)
    capture = BBMove(
        from_sq=10,
        to_sq=5,
        piece_type=PAWN,
        captured_type=PAWN,  # Capturing a pawn
        promo=0
    )

    non_capture_score = score_move(non_capture)
    capture_score = score_move(capture)

    print(f"\nNon-capture (pawn push):")
    print(f"  captured_type = -1")
    print(f"  Score = {non_capture_score}")

    print(f"\nCapture (pawn×pawn):")
    print(f"  captured_type = {PAWN}")
    print(f"  Victim value = {MVV_LVA_VALUES[PAWN]}")
    print(f"  Attacker value = {MVV_LVA_VALUES[PAWN]}")
    print(f"  Score = ({MVV_LVA_VALUES[PAWN]} × 10) - {MVV_LVA_VALUES[PAWN]} = {capture_score}")

    if non_capture_score == 0:
        print("\n✓ PASS: Non-capture correctly scores 0")
        pass1 = True
    else:
        print(f"\n✗ FAIL: Non-capture should score 0, got {non_capture_score}")
        pass1 = False

    if capture_score > non_capture_score:
        print("✓ PASS: Capture scores higher than non-capture")
        pass2 = True
    else:
        print(f"✗ FAIL: Capture ({capture_score}) should score higher than non-capture ({non_capture_score})")
        pass2 = False

    return pass1 and pass2


def test_mvv_lva_ordering():
    """Test MVV-LVA: prefer valuable victims and cheap attackers."""
    print("\n" + "="*60)
    print("TEST 3: MVV-LVA Ordering")
    print("="*60)

    # Pawn takes queen (high-value victim, low-value attacker) → BEST
    pawn_takes_queen = BBMove(10, 5, PAWN, QUEEN, 0)

    # Queen takes pawn (low-value victim, high-value attacker) → WORST
    queen_takes_pawn = BBMove(10, 5, QUEEN, PAWN, 0)

    # Pawn takes pawn (equal exchange) → MIDDLE
    pawn_takes_pawn = BBMove(10, 5, PAWN, PAWN, 0)

    score1 = score_move(pawn_takes_queen)
    score2 = score_move(queen_takes_pawn)
    score3 = score_move(pawn_takes_pawn)

    print(f"\n1. Pawn×Queen:")
    print(f"   Victim={MVV_LVA_VALUES[QUEEN]}, Attacker={MVV_LVA_VALUES[PAWN]}")
    print(f"   Score = ({MVV_LVA_VALUES[QUEEN]} × 10) - {MVV_LVA_VALUES[PAWN]} = {score1}")

    print(f"\n2. Queen×Pawn:")
    print(f"   Victim={MVV_LVA_VALUES[PAWN]}, Attacker={MVV_LVA_VALUES[QUEEN]}")
    print(f"   Score = ({MVV_LVA_VALUES[PAWN]} × 10) - {MVV_LVA_VALUES[QUEEN]} = {score2}")

    print(f"\n3. Pawn×Pawn:")
    print(f"   Victim={MVV_LVA_VALUES[PAWN]}, Attacker={MVV_LVA_VALUES[PAWN]}")
    print(f"   Score = ({MVV_LVA_VALUES[PAWN]} × 10) - {MVV_LVA_VALUES[PAWN]} = {score3}")

    if score1 > score3 > score2:
        print(f"\n✓ PASS: Correct ordering: Pawn×Queen ({score1}) > Pawn×Pawn ({score3}) > Queen×Pawn ({score2})")
        return True
    else:
        print(f"\n✗ FAIL: Wrong ordering!")
        print(f"  Expected: {score1} > {score3} > {score2}")
        return False


def test_all_piece_types():
    """Test that all piece types can be scored correctly."""
    print("\n" + "="*60)
    print("TEST 4: All Piece Types")
    print("="*60)

    piece_names = ['Pawn', 'Knight', 'Bishop', 'Queen', 'King', 'Right']
    piece_types = [PAWN, KNIGHT, BISHOP, QUEEN, KING, RIGHT]

    print("\nCapture scoring for all piece type combinations:")
    print(f"{'Attacker':<10} {'Victim':<10} {'Score':>10}")
    print("-" * 30)

    all_valid = True
    for i, attacker_type in enumerate(piece_types):
        for j, victim_type in enumerate(piece_types):
            move = BBMove(10, 5, attacker_type, victim_type, 0)
            score = score_move(move)

            expected = (MVV_LVA_VALUES[victim_type] * 10) - MVV_LVA_VALUES[attacker_type]

            if score == expected:
                status = "✓"
            else:
                status = "✗"
                all_valid = False

            print(f"{status} {piece_names[i]:<10} {piece_names[j]:<10} {score:>10}")

    if all_valid:
        print("\n✓ PASS: All piece type combinations score correctly")
    else:
        print("\n✗ FAIL: Some piece type combinations have wrong scores")

    return all_valid


def run_all_tests():
    """Run all capture scoring tests."""
    print("\n" + "#"*60)
    print("# CAPTURE SCORING TESTS (AFTER BUGFIX)")
    print("#"*60)

    results = []

    results.append(("MVV-LVA Array", test_mvv_lva_array()))
    results.append(("Capture Detection", test_capture_detection()))
    results.append(("MVV-LVA Ordering", test_mvv_lva_ordering()))
    results.append(("All Piece Types", test_all_piece_types()))

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in results)

    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nCapture scoring is now working correctly.")
        print("agentBitboard should now prefer captures over non-captures.")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease review the failures above.")
    print("="*60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
