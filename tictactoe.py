# tictactoe.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class TicTacToe:
    """
    Environment class for an N x N tic-tac-toe game where a player needs K in a row to win.
    Includes an early check to see if no one can possibly achieve K => immediate draw.
    """
    def __init__(self, N=3, K=3):
        """
        N: board size NxN
        K: number in a row needed to win
        """
        self.N = N
        self.K = K
        self.reset()
        self.done = False
        self.history = []
        self.winning_lines = self._compute_winning_lines()

    def reset(self):
        self.board = [0]*(self.N * self.N)
        self.current_player = 1  # X (1) goes first
        self.done = False
        self.winner = None
        self.history = []
        return self.board.copy()

    def step(self, action):
        """
        action: an integer in [0..N*N-1], where we place X or O.
        Returns: (next_state, reward, done).
        """
        if self.board[action] != 0:
            raise ValueError("Invalid move: cell is not empty.")

        self.board[action] = self.current_player

        # Check if current_player just won
        if self.check_winner(self.current_player):
            # The current player wins => done
            self.done = True
            self.winner = self.current_player
            reward = 1.0
        elif 0 not in self.board:
            # Board is full => draw
            self.done = True
            self.winner = 0
            reward = 0.0
        else:
            # Check if anyone can still form K in a row
            if not self.can_still_win(1) and not self.can_still_win(-1):
                self.done = True
                self.winner = 0
                reward = 0.0
            else:
                reward = 0.0
                # Switch player
                self.current_player *= -1

        self.history.append(self.board.copy())
        return self.board.copy(), reward, self.done

    def check_winner(self, player):
        """
        Return True if 'player' (1 or -1) has a line of length K.
        """
        for line in self.winning_lines:
            if all(self.board[idx] == player for idx in line):
                return True
        return False

    def can_still_win(self, player):
        """
        Returns True if there's at least one winning line
        that has no opponent symbol in it (so it can still be potentially filled to length K).
        """
        opponent = -player
        for line in self.winning_lines:
            count_opp = sum(self.board[idx] == opponent for idx in line)
            if count_opp == 0:
                # This line has no opponent's symbols => possible to still fill for 'player'
                return True
        return False

    def get_available_actions(self):
        return [i for i, cell in enumerate(self.board) if cell == 0]

    def _compute_winning_lines(self):
        """
        Precompute all lines of length K in the NxN board:
        rows, columns, and diagonals.
        """
        N = self.N
        K = self.K
        grid = np.arange(N*N).reshape(N, N)
        lines = []
        for r in range(N - K + 1):
            for c in range(N - K + 1):
                # rows
                if r == 0:
                    lines.extend([grid[i, c:c+K].tolist() for i in range(N)])
                # columns
                if c == 0:
                    lines.extend([grid[r:r+K, j].tolist() for j in range(N)])
                # diagonals
                diag1 = grid[r:r+K, c:c+K].diagonal().tolist()
                diag2 = np.fliplr(grid)[r:r+K, c:c+K].diagonal().tolist()
                lines.append(diag1)
                lines.append(diag2)
        return lines

    def render(self, interval=500):
        """
        Animate the history of the game using matplotlib.
        """
        plt.ioff()
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, self.N)
        ax.set_ylim(0, self.N)
        ax.set_aspect("equal")
        ax.set_frame_on(False)

        for i in range(1, self.N):
            ax.axhline(i, color='black', linewidth=2)
            ax.axvline(i, color='black', linewidth=2)

        symbols = {1: "X", -1: "O", 0: " "}
        text_elems = [[ax.text(c+0.5, self.N-r-0.5, '', fontsize=24,
                               ha='center', va='center')
                       for c in range(self.N)] for r in range(self.N)]

        def update(frame):
            board_state = self.history[frame]
            for rr in range(self.N):
                for cc in range(self.N):
                    idx = rr*self.N + cc
                    text_elems[rr][cc].set_text(symbols[board_state[idx]])
            return []

        ani = FuncAnimation(fig, update, frames=len(self.history),
                            interval=interval, repeat=False, blit=True)
        plt.ion()
        return ani
