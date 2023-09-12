import torch
from torch import nn
from torch.nn import functional as F
from env.Environment import Environment
from agents_tictactoe.ReinforceBaselineShareAgent import ReinforceBaselineShareAgent


class ActorCriticShareOneStepAgent(ReinforceBaselineShareAgent):
    def __init__(self, env: Environment,
                 policy_net: nn.Module = None,
                 lr=0.002,
                 weight_decay=0.01) -> None:
        ReinforceBaselineShareAgent.__init__(self, env, policy_net, lr, weight_decay)

    def update_network(self, gamma: float) -> float:
        """
        Updates the agent's neural network using one-step return.

        Args:
            gamma: Discount factor for future rewards.

        Returns:
            float: Total loss in the episode.
        """
        total_loss = 0
        # Extend state values list with zero for last state value
        next_state_values = self.saved_state_values[1:] + [torch.tensor(0.)]

        for reward, value, next_value, log_prob in zip(self.saved_rewards,
                                                       self.saved_state_values,
                                                       next_state_values,
                                                       self.saved_log_probs):
            Gt = reward + gamma * next_value.item()  # Compute one-step return

            delta = Gt - value.item()
            policy_loss = -delta * log_prob  # Negative sign for gradient ascent
            value_loss = F.smooth_l1_loss(value, torch.tensor([Gt]).view(1, 1).to(self.device))
            loss = policy_loss + value_loss  # Combined loss
            total_loss += loss
            loss.backward()

        self.policy_optimizer.step()  # Update the weights
        self.policy_optimizer.zero_grad()  # Clear the gradients
        return total_loss
