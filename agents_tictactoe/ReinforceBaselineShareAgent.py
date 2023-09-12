from typing import Tuple
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from env.Environment import Environment
from agents_tictactoe.Agent import Agent, compute_returns, index_to_coord


class ReinforceBaselineShareAgent(Agent):
    def __init__(self, env: Environment,
                 policy_net: nn.Module = None,
                 lr=0.002,
                 weight_decay=0.01) -> None:
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
            # Training mode
            self.policy_net.train()
            action_probs, state_value = self.policy_net(state)
            self.saved_state_values.append(state_value.detach())

            m = Categorical(action_probs)
            action = m.sample()
            if index_to_coord(action.item(), self.env.side_length) not in self.env.legal_actions():
                self.invalid_move_count += 1
                if self.invalid_move_count > 0 and self.invalid_move_count % 1000 == 0:
                    print(self.invalid_move_count)  # Print invalid move count at intervals
            log_prob = m.log_prob(action)
            self.saved_log_probs.append(log_prob)

        else:
            # Evaluation mode
            self.policy_net.eval()
            with torch.no_grad():
                action_probs, _ = self.policy_net(state)
            action = action_probs.argmax(dim=-1)
            while index_to_coord(action.item(), self.env.side_length) not in self.env.legal_actions():
                self.invalid_move_count += 1
                action_probs[0, action] = 0
                action = action_probs.argmax(dim=-1)

        return index_to_coord(action.item(), self.env.side_length)

    def update_network(self, gamma: float) -> float:
        """
        Updates the agent's neural network using the saved rewards, log probabilities, and state values.

        Args:
            gamma: Discount factor for future rewards.

        Returns:
            float: Total loss in the episode.
        """
        total_loss = 0
        returns = torch.Tensor(compute_returns(self.saved_rewards, gamma)).to(self.device)

        for log_prob, value, R in zip(self.saved_log_probs, self.saved_state_values, returns):
            delta = R - value
            policy_loss = -delta * log_prob  # Negative sign for gradient ascent
            value_loss = F.smooth_l1_loss(value, torch.Tensor([R]).view(1, 1).to(self.device))
            loss = policy_loss + value_loss  # Combined loss
            total_loss += loss
            loss.backward()

        self.policy_optimizer.step()  # Update the weights
        self.policy_optimizer.zero_grad()  # Clear the gradients
        return total_loss
