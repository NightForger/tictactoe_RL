# utils.py

def canonical_state(board, current_player):
    """
    Return a tuple of length=len(board), where:
      current_player's cells = +1,
      opponent's cells = -1,
      empty cells = 0.
    """
    return tuple(current_player * cell for cell in board)
