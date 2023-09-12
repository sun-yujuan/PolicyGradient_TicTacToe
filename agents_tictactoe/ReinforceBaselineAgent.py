from typing import Tuple
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from env.Environment import Environment
from agents_tictactoe.Agent import Agent, compute_returns, index_to_coord


class ReinforceBaselineAgent(Agent):
    def __init__(self, env: Environment, policy_net: nn.Module = None, value_net: nn.Module = None, lr=0.002,
                 weight_decay=0.01) -> None:
        Agent.__init__(self, env, policy_net, lr, weight_decay)
        if value_net is not None:
            self.value_net = value_net.to(self.device)
            self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr, weight_decay=weight_decay)

    def get_action(self, is_eval: bool) -> Tuple[int, int]:
        """
        Gets an action from the agent using policy and value networks.

        Args:
            is_eval: Boolean indicating if the agent is in evaluation mode or training mode.

        Returns:
            Tuple[int, int]: The action coordinates chosen by the agent.
        """
        state = self.env.get_state().to(self.device)

        if not is_eval:
            # Training mode
            self.policy_net.train()
            self.value_net.train()
            action_probs = self.policy_net(state)
            state_value = self.value_net(state)
            self.saved_state_values.append(state_value.detach())

            m = Categorical(action_probs)
            action = m.sample()
            if index_to_coord(action.item(), self.env.side_length) not in self.env.legal_actions():
                self.invalid_move_count += 1
            log_prob = m.log_prob(action)
            self.saved_log_probs.append(log_prob)

        else:
            # Evaluation mode
            self.policy_net.eval()
            self.value_net.eval()
            with torch.no_grad():
                action_probs = self.policy_net(state)
            action = action_probs.argmax(dim=-1)
            while index_to_coord(action.item(), self.env.side_length) not in self.env.legal_actions():
                self.invalid_move_count += 1
                action_probs[0, action] = 0
                action = action_probs.argmax(dim=-1)

        return index_to_coord(action.item(), self.env.side_length)

    def update_network(self, gamma: float) -> float:
        """
        Updates the agent's policy and value networks using the saved rewards, log probabilities, and state values.

        Args:
            gamma: Discount factor for future rewards.

        Returns:
            float: Total loss in the episode.
        """
        returns = torch.Tensor(compute_returns(self.saved_rewards, gamma)).to(self.device)
        total_loss = 0

        # Update policy network
        for log_prob, value, R in zip(self.saved_log_probs, self.saved_state_values, returns):
            policy_loss = -log_prob * (R - value.item())
            policy_loss.backward(retain_graph=True)
            total_loss += policy_loss
        self.policy_optimizer.step()
        self.policy_optimizer.zero_grad()

        # Update value network
        for value, R in zip(self.saved_state_values, returns):
            value_loss = F.smooth_l1_loss(value, torch.Tensor([R]).view(1, 1).to(self.device)).requires_grad_()
            value_loss.backward(retain_graph=True)
            total_loss += value_loss
        self.value_optimizer.step()
        self.value_optimizer.zero_grad()

        return total_loss
