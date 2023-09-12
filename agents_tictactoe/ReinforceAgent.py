from typing import Tuple
import torch
from torch import nn
from torch.distributions import Categorical
from env.Environment import Environment
from agents_tictactoe.Agent import Agent, compute_returns, index_to_coord


class ReinforceAgent(Agent):

    def __init__(self, env: Environment, policy_net: nn.Module = None, lr=0.002, weight_decay=0.01) -> None:
        Agent.__init__(self, env, policy_net, lr, weight_decay)

    def get_action(self, is_eval: bool) -> Tuple[int, int]:
        """
        Gets an action from the agent using the policy network.

        Args:
            is_eval: Boolean indicating if the agent is in evaluation mode or training mode.

        Returns:
            Tuple[int, int]: The action coordinates chosen by the agent.
        """
        state = self.env.get_state().to(self.device)

        if not is_eval:
            self.policy_net.train()  # Training mode
            action_probs = self.policy_net(state)  # Compute action probabilities
            m = Categorical(action_probs)  # Create distribution
            action = m.sample()  # Sample action
            if index_to_coord(action.item(), self.env.side_length) not in self.env.legal_actions():
                self.invalid_move_count += 1
            log_prob = m.log_prob(action)  # Compute log probability
            self.saved_log_probs.append(log_prob)  # Save for later update

        else:
            self.policy_net.eval()  # Evaluation mode
            with torch.no_grad():
                action_probs = self.policy_net(state)  # Compute action probabilities
            action = action_probs.argmax(dim=-1)  # Select action with highest probability
            while index_to_coord(action.item(), self.env.side_length) not in self.env.legal_actions():
                self.invalid_move_count += 1
                action_probs[0, action] = 0
                action = action_probs.argmax(dim=-1)

        return index_to_coord(action.item(), self.env.side_length)

    def update_network(self, gamma: float) -> float:
        """
        Updates the policy network using the saved rewards and log probabilities.

        Args:
            gamma: Discount factor for future rewards.

        Returns:
            float: Total loss in the episode.
        """
        total_loss = 0
        returns = torch.Tensor(compute_returns(self.saved_rewards, gamma)).to(self.device)

        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss = -log_prob * R  # Compute policy loss
            total_loss += policy_loss
            policy_loss.backward()  # Compute gradients

        self.policy_optimizer.step()  # Update weights
        self.policy_optimizer.zero_grad()  # Clear gradients
        return total_loss
