import numpy as np
from typing import Tuple
from env.Environment import Environment


class TicTacToeEnvironment(Environment):
    """
    This class represents a Tic Tac Toe game environment that extends the base Environment class.
    It has a specific board size of 3x3 and implements a unique method for identifying the game winner.
    """

    def __init__(self, board_size: Tuple[int, int] = (3, 3)):
        """
        Initialize a new game of Tic Tac Toe.

        Args:
            board_size (tuple, optional): The size of the Tic Tac Toe board. Defaults to (3,3).
        """
        super().__init__(board_size)

    def get_winner(self) -> int:
        """
        Identify the winner of the Tic Tac Toe game.
        A player is considered the winner if they have three of their markers in a row,
        column, or either of the diagonals on the board.

        Returns:
            int:
                0: No winner,
                1: Player 1 is the winner,
               -1: Player 2 is the winner.
        """
        for player in self.players:
            # check the diagonals
            diag_1 = True
            diag_2 = True

            for index in range(3):
                # check each row and column for a win
                if np.all(self.board[index, :] == player) or np.all(self.board[:, index] == player):
                    return player  # Return the winning player

                # Check the diagonals
                if self.board[index, index] != player:
                    diag_1 = False
                if self.board[index, 2 - index] != player:
                    diag_2 = False

            # If either of the diagonals have all the same player's marker, declare them as the winner
            if diag_1 or diag_2:
                return player

        # If no player has three in a row, column, or diagonal, then there is no winner
        return 0
