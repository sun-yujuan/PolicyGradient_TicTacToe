from copy import deepcopy
import numpy as np
import torch
from typing import List, Tuple


class Environment(object):
    """
    This class represents the game environment where players make moves, win, lose or draw.
    The environment includes the board (a 2D NumPy array) and the current player.
    The board's cells contain 0 if empty, 1 if occupied by player 1, and -1 if occupied by player 2.
    """

    STATUS_VALID_MOVE = 'valid'  # The last move was valid
    STATUS_INVALID_MOVE = 'invalid'  # The last move was invalid (the cell was not empty)
    STATUS_WIN = 'win'  # The current player has won the game
    STATUS_DRAW = 'draw'  # The game is a draw
    STATUS_LOSE = 'lose'  # The current player has lost the game

    def __init__(self, board_size: Tuple[int, int]):
        """
        Initialize a new game environment.

        :param board_size: a tuple (rows, columns) representing the size of the game board
        """
        # The game board is a 2D array filled with zeros (empty cells)
        self.board = np.zeros(board_size, dtype=int)
        # Players are represented as 1 and -1
        self.players = [1, -1]
        # The first player to make a move is player 1
        self.current_player = self.players[0]
        self.board_size = board_size
        # Used for transformation between index and coordinates
        self.side_length = board_size[1]
        self.done = False  # Is the game terminal?

    def reset(self) -> np.ndarray:
        """
        Reset the game environment to its initial state.

        :return: the initial state of the board
        """
        self.board = np.zeros(self.board_size, dtype=int)
        self.current_player = self.players[0]
        self.done = False
        return self.board

    def transition(self, action: Tuple[int, int]) -> np.ndarray:
        """
        Update the game board by making a move (putting the current player's marker in the chosen cell).

        :param action: a tuple (row, column) representing the chosen cell
        :return: the new state of the board
        """
        new_state = deepcopy(self.board)
        new_state[action] = self.current_player
        self.board = new_state
        return new_state

    def legal_actions(self) -> List[Tuple[int, int]]:
        """
        Get the list of legal actions for the current player.

        :return: a list of tuples (row, column) representing empty cells on the board
        """
        legal_actions = [(row, col) for row in range(self.board_size[0]) for col in range(self.board_size[1]) if
                         self.board[row][col] == 0]
        return legal_actions

    def is_terminal(self) -> bool:
        """
        Check if the game has ended.

        :return: True if the game has ended, False otherwise
        """
        return self.done

    def get_reward(self, status: str) -> float:
        """
        Get the reward for the current player based on the game status.

        :param status: a string representing the game status
        :return: A float representing the reward for the current status.
        """
        return {
            self.STATUS_VALID_MOVE: 0,
            self.STATUS_INVALID_MOVE: -5,
            self.STATUS_WIN: 1,
            self.STATUS_DRAW: 0,
            self.STATUS_LOSE: -1
        }[status]

    def get_winner(self) -> int:
        """
        Identify the winner of the game. This method should be implemented in the corresponding game environment.

        :return: the marker of the winning player, or 0 if there is no winner yet
        """
        raise NotImplementedError()

    def is_draw(self) -> bool:
        """
        Check if the game is a draw. This method can only be used after checking if the winner exist.

        :return: True if the game is a draw, False otherwise
        """
        return np.all(self.board != 0)

    def step(self, action: Tuple[int, int]) -> Tuple[float, bool]:
        """
        Perform a game step.

        :param action: a tuple (row, column) representing the chosen cell
        :return: the reward for the current player and a boolean value indicating if the game has ended
        """
        # If the chosen cell is not empty, it's an invalid move
        if self.board[action] != 0:
            self.done = True
            return self.get_reward(self.STATUS_INVALID_MOVE), self.done

        # Update the board state
        self.board = self.transition(action)

        # Check if the game has a winner
        winner = self.get_winner()
        if winner != 0:  # If there is a winner
            self.done = True  # Mark the game as done
            if winner == self.current_player:  # If the winner is the current player
                return self.get_reward(self.STATUS_WIN), self.done
            else:  # If the winner is the other player
                return self.get_reward(self.STATUS_LOSE), self.done

        # Check if the game is a draw
        if self.is_draw():
            self.done = True  # Mark the game as done
            return self.get_reward(self.STATUS_DRAW), self.done

        # Switch the current player
        self.current_player = -self.current_player

        # If the game is not over, it's a valid move
        return self.get_reward(self.STATUS_VALID_MOVE), self.done

    def get_computer_move(self) -> Tuple[int, int]:
        """
        Choose a random legal action for the computer player.

        :return: a tuple (row, column) representing the chosen cell
        """
        legal_actions = self.legal_actions()
        return legal_actions[np.random.choice(len(legal_actions))]

    def get_state(self) -> torch.Tensor:
        """
        Get the current state of the game by applying the one-hot encoding on the game board

        :return: a PyTorch tensor representing the game state
        """
        state = self.board.flatten() + 1
        state = torch.from_numpy(state).long().unsqueeze(0)

        # One-hot encode each number in state, then reshape the tensor into a new shape
        total_num = self.board_size[0] * self.board_size[1]
        state = torch.zeros(3, total_num).scatter_(0, state, 1).view(1, 3 * total_num)
        return state

    def print_board(self) -> None:
        """
        Print the current state of the game board.
        """
        print('-' * (4 * self.board_size[1] + 1))
        for row in range(self.board_size[0]):
            print('|', end='')
            for col in range(self.board_size[1]):
                if self.board[row][col] == 0:
                    character = ' '
                elif self.board[row][col] == 1:
                    character = 'X'
                else:
                    character = 'O'
                print(' ' + character + ' |', end='')
            print('\n' + '-' * (4 * self.board_size[1] + 1))  # Separating line
