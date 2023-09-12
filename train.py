import random
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from agents_tictactoe.Agent import Agent
from env.Environment import Environment


def run_episode(env: Environment, agent_a: Agent, agent_b: Agent, draw_board: bool, is_eval: bool) -> None:
    """
    Runs a single episode in the game environment with two competing agents_tictactoe.

    Args:
        env: The game environment where the agents_tictactoe play.
        agent_a: The first agent.
        agent_b: The second agent.
        draw_board: if the game board will be drawn (set as true to see how the agent play the game)
        is_eval: Boolean flag indicating if the agents_tictactoe are in evaluation mode or training mode.
    """
    # Reset the environment before running the episode
    _ = env.reset()
    done = False

    while not done:
        # If it's agent A's turn, let it make a move
        if env.current_player == 1:
            done = agent_a.make_move(is_eval)
            if draw_board:
                env.print_board()
            # If agent A wins and agent B has a neural network, adjust agent B's last reward
            if done and env.get_winner() == 1 and agent_b.policy_net is not None:
                agent_b.saved_rewards[-1] = env.get_reward(Environment.STATUS_LOSE)
        # If it's agent B's turn and the game is not yet finished, let it make a move
        if not done and env.current_player == -1:
            done = agent_b.make_move(is_eval)
            if draw_board:
                env.print_board()
            # If agent B wins and agent A has a neural network, adjust agent A's last reward
            if done and env.get_winner() == -1 and agent_a.policy_net is not None:
                agent_a.saved_rewards[-1] = env.get_reward(Environment.STATUS_LOSE)


def train(env: Environment,
          agent_a: Agent,
          agent_b: Agent,
          episodes: int = 50000,
          gamma: float = 1.0,
          log_interval: int = 1000,
          writer: SummaryWriter = None) -> None:
    """
    Trains the agents_tictactoe over a specified number of episodes in the game environment.

    Args:
        env: The game environment where the agents_tictactoe play.
        agent_a: The first agent.
        agent_b: The second agent.
        episodes: The number of episodes to train over.
        gamma: Discount factor for future rewards.
        log_interval: The interval at which training progress is logged.
        writer: A TensorBoard SummaryWriter object for logging training progress.
    """
    # Track invalid move counts and win counts for both agents_tictactoe
    invalid_move_counts = [0, 0]
    win_count = [0, 0]

    run_loss = [0, 0]

    # Loop over each episode
    for episode in range(episodes):
        # Reset both agents_tictactoe
        agent_a.reset()
        agent_b.reset()

        # Randomly decide the order of the agents_tictactoe
        order = (0, 1)
        if random.random() < 0.5:
            run_episode(env, agent_a=agent_a, agent_b=agent_b, draw_board=False, is_eval=False)
        else:
            run_episode(env, agent_a=agent_b, agent_b=agent_a, draw_board=False, is_eval=False)
            order = (1, 0)

        # Record wins for the corresponding agent
        if env.get_winner() == 1:
            win_count[order[0]] += 1
        elif env.get_winner() == -1:
            win_count[order[1]] += 1

        # Record invalid moves
        invalid_move_counts[0] += agent_a.invalid_move_count
        invalid_move_counts[1] += agent_b.invalid_move_count

        # Update the networks of the agents_tictactoe if they exist
        if agent_a.policy_net is not None:
            run_loss[0] += agent_a.update_network(gamma)
        if agent_b.policy_net is not None:
            run_loss[1] += agent_b.update_network(gamma)

        # Log progress at set intervals
        if (episode + 1) % log_interval == 0:
            print(f"Episode: {episode + 1}:")
            print(f"Win Count for A: {win_count[0]}\tInvalid Moves for A: {invalid_move_counts[0]}")
            print(f"Win Count for B: {win_count[1]}\tInvalid Moves for B: {invalid_move_counts[1]}")
            if writer is not None:
                writer.add_scalar('Number of invalid Move for A', invalid_move_counts[0], episode + 1)
                writer.add_scalar('Number of invalid Move for B', invalid_move_counts[1], episode + 1)
                writer.add_scalar('Number of Win for A', win_count[0], episode + 1)
                writer.add_scalar('Number of Win for B', win_count[1], episode + 1)
                writer.add_scalar('Loss for A', run_loss[0], episode + 1)
                writer.add_scalar('Loss for B', run_loss[1], episode + 1)

            invalid_move_counts = [0, 0]
            win_count = [0, 0]
            run_loss = [0, 0]

    if writer is not None:
        # Flush the TensorBoard writer
        writer.flush()
        writer.close()


def test(env: Environment, agent_a: Agent, agent_b: Agent, draw_board: bool = False, episodes: int = 10000) -> None:
    """
    Tests the agents_tictactoe over a specified number of episodes in the game environment.

    Args:
        env: The game environment where the agents_tictactoe play.
        agent_a: The first agent.
        agent_b: The second agent.
        episodes: The number of episodes to test over.
    """
    invalid_move_counts = [0, 0]
    win_count = [0, 0]
    loss_count = [0, 0]

    # Loop over each episode
    for episode in range(episodes):
        agent_a.reset()
        agent_b.reset()
        order = (0, 1)

        if random.random() < 0.5:
            run_episode(env, agent_a=agent_a, agent_b=agent_b, draw_board=draw_board, is_eval=True)
        else:
            run_episode(env, agent_a=agent_b, agent_b=agent_a, draw_board=draw_board, is_eval=True)
            order = (1, 0)

        # Record wins for the corresponding agent
        if env.get_winner() == 1:
            win_count[order[0]] += 1
            loss_count[order[1]] += 1
        elif env.get_winner() == -1:
            win_count[order[1]] += 1
            loss_count[order[0]] += 1

        invalid_move_counts[0] += agent_a.invalid_move_count
        invalid_move_counts[1] += agent_b.invalid_move_count

    print(f"TEST RESULT for {episode + 1} episodes:")
    print(f"Win Count for A: {win_count[0]}\tInvalid Moves for A: {invalid_move_counts[0]}")
    print(f"Win Count for B: {win_count[1]}\tInvalid Moves for B: {invalid_move_counts[1]}")

    print(f"Loss Count for A: {loss_count[0]}")
    print(f"Loss Count for B: {loss_count[1]}")
    print(f"Draw Count: {episodes-win_count[0]-loss_count[0]}")


def analyse_model(net: nn.Module, state: np.array) -> None:
    """
    Analyzes the agent's neural network by printing its output for a given state.

    Args:
        net: The neural network of the agent.
        state: The game state to feed into the neural network.
    """
    print(net(state))
