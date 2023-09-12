import torch
from agents_tictactoe.Agent import Agent
from agents_tictactoe.ReinforceAgent import ReinforceAgent
from env.Environment import Environment
from env.TicTacToeEnvironment import TicTacToeEnvironment


def play_game_human(env: Environment, agent: Agent, agent_role=1):
    """
    This function allows a human player to play a game against the trained agent.

    Args:
        env: An instance of the game environment.
        agent: The agent that is playing the game.
        agent_role: The role of the agent (default is 1),
                    where value 1 refers to player 1 and value -1 refers to player 2.
                    If it is 1, the agent will move first.

    Returns:
        winner: The winner of the game. It will be 1 if agent wins, -1 if human wins, and 0 if it's a draw.
    """

    # Reset the environment before the start of the game
    env.reset()

    # While the game is not over, continue playing
    while not env.done:
        # If it's the agent's turn, let the agent select an action
        if agent_role == env.current_player:
            row, col = agent.get_action(is_eval=True)
        else:
            # If it's the human's turn, take input from the human player
            row = int(input(f"Enter the row (0-{env.board_size[0] - 1}): "))
            col = int(input(f"Enter the column (0-{env.board_size[1] - 1}): "))

            # If the chosen cell is not empty, indicate that the move is invalid and continue the loop
            if env.board[row][col] != 0:
                print("Invalid move. Try again.")
                continue

        # Execute the move and print the current state of the board
        env.step((row, col))
        env.print_board()

    # Check if there is a winner or a draw
    winner = env.get_winner()
    if winner == 1:
        print(f"Player {1} wins!")
    elif winner == -1:
        print(f"Player {2} wins!")
    elif env.is_draw():
        print("It's a draw!")
        winner = 0  # Indicate a draw by a 0

    return winner


def main():
    env = TicTacToeEnvironment()
    # Load the model which are stored in the models file
    net = torch.load('models/tictactoe/reinforce/agents_20k_0.001_8.pth')
    # Change the agent according to the algorithm used
    agent = ReinforceAgent(env, net)
    # Agent_role 1 for player 1 and -1 for player 2
    play_game_human(env, agent, agent_role=1)


if __name__ == '__main__':
    main()
