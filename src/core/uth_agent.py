# src/core/uth_agent.py
"""
Deep CFR agent for Ultimate Texas Hold'em.

Reuses the project's PokerNetwork / PrioritizedMemory / encode_state stack.
The action space mirrors the NLHE 3-action layout:
    0 = Fold        (legal only at the river decision)
    1 = Check       (preflop / flop)
    2 = Play        (raise; multiplier determined by stage: 3x/4x, 2x, 1x)

For the preflop Play the network's sizing head (output range 0.1..3.0) is
discretized: sizing >= PREFLOP_4X_THRESHOLD -> 4x, otherwise 3x.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.core.model import PokerNetwork, encode_state, set_verbose  # noqa: F401
from src.core import model as _model
from src.core.deep_cfr import PrioritizedMemory, _resolve_model_save_path
from src.core.uth_env import (
    UTHAction,
    UTHActionEnum,
    UTHStage,
    UTHStatus,
)
from src.utils.settings import STRICT_CHECKING

# Number of extra features appended to the base encoding.
UTH_EXTRA_FEATURES = 10

# Sizing head output is in [0.1, 3.0]; values at or above the midpoint
# select the 4x preflop play, below it the 3x play.
PREFLOP_4X_THRESHOLD = 1.55


def encode_uth_extras(state):
    """
    UTH-specific features appended to the base encode_state vector:
      [0:3]  one-hot decision point (preflop / flop / river)
      [3]    trips bet placed flag
      [4]    play bet placed flag
      [5]    folded flag
      [6:10] normalized amounts: ante, blind, trips, play
    """
    extras = np.zeros(UTH_EXTRA_FEATURES)

    if state.stage == UTHStage.Preflop:
        extras[0] = 1.0
    elif state.stage == UTHStage.Flop:
        extras[1] = 1.0
    elif state.stage == UTHStage.River:
        extras[2] = 1.0

    extras[3] = 1.0 if state.trips > 0 else 0.0
    extras[4] = 1.0 if state.play > 0 else 0.0
    extras[5] = 1.0 if state.folded else 0.0

    initial_stake = state.players_state[0].stake + state.players_state[0].bet_chips
    if initial_stake <= 0:
        initial_stake = 1.0
    extras[6] = state.ante / initial_stake
    extras[7] = state.blind / initial_stake
    extras[8] = state.trips / initial_stake
    extras[9] = state.play / initial_stake

    return extras


def encode_uth_state(state, player_id=0):
    """Base pokers-style encoding (duck-typed) + UTH extras."""
    return np.concatenate([encode_state(state, player_id), encode_uth_extras(state)])


class UTHDeepCFRAgent:
    """Deep CFR agent for the single-decision-maker UTH game tree."""

    def __init__(self, player_id=0, memory_size=300000, device='cpu'):
        self.player_id = player_id
        self.num_players = 2  # player + dealer (dealer never acts)
        self.device = device

        self.num_actions = 3  # 0=Fold, 1=Check, 2=Play

        # Base encoding size for num_players=2 (see DeepCFRAgent.__init__):
        # 52+52+5+1+P+P+P*4+1+4+5 with P=2 -> 132, plus UTH extras.
        base_input = 52 + 52 + 5 + 1 + 2 + 2 + 2 * 4 + 1 + 4 + 5
        self.input_size = base_input + UTH_EXTRA_FEATURES

        self.advantage_net = PokerNetwork(
            input_size=self.input_size, hidden_size=256, num_actions=self.num_actions
        ).to(device)
        self.optimizer = optim.Adam(
            self.advantage_net.parameters(), lr=1e-6, weight_decay=1e-5
        )
        self.advantage_memory = PrioritizedMemory(memory_size)

        self.strategy_net = PokerNetwork(
            input_size=self.input_size, hidden_size=256, num_actions=self.num_actions
        ).to(device)
        self.strategy_optimizer = optim.Adam(
            self.strategy_net.parameters(), lr=0.00005, weight_decay=1e-5
        )
        self.strategy_memory = deque(maxlen=memory_size)

        self.iteration_count = 0
        self.max_regret_seen = 1.0

        # Sizing head bounds (shared with PokerNetwork's output range).
        self.min_bet_size = 0.1
        self.max_bet_size = 3.0

    # -- action mapping ------------------------------------------------------

    def action_type_to_uth_action(self, action_type, state, bet_size_multiplier=None):
        """
        Map (action_type, sizing) to a legal UTHAction with NLHE-style
        fallbacks (Play -> Check -> Fold).
        """
        legal = state.legal_actions

        if action_type == 0:  # Fold
            if UTHActionEnum.Fold in legal:
                return UTHAction(UTHActionEnum.Fold)
            if UTHActionEnum.Check in legal:
                return UTHAction(UTHActionEnum.Check)
            return UTHAction(UTHActionEnum.Fold)  # last resort

        if action_type == 1:  # Check
            if UTHActionEnum.Check in legal:
                return UTHAction(UTHActionEnum.Check)
            if UTHActionEnum.Fold in legal:
                return UTHAction(UTHActionEnum.Fold)
            return UTHAction(UTHActionEnum.Check)  # last resort

        if action_type == 2:  # Play (Raise)
            if UTHActionEnum.Raise not in legal:
                if UTHActionEnum.Check in legal:
                    return UTHAction(UTHActionEnum.Check)
                return UTHAction(UTHActionEnum.Fold)
            multipliers = state.legal_play_multipliers()
            if len(multipliers) == 1:
                return UTHAction(UTHActionEnum.Raise, multipliers[0])
            # Preflop: discretize the sizing head into 3x vs 4x.
            sizing = 1.0 if bet_size_multiplier is None else float(bet_size_multiplier)
            sizing = max(self.min_bet_size, min(self.max_bet_size, sizing))
            chosen = 4.0 if sizing >= PREFLOP_4X_THRESHOLD else 3.0
            return UTHAction(UTHActionEnum.Raise, chosen)

        # Unknown action type: safe fallback.
        if UTHActionEnum.Check in legal:
            return UTHAction(UTHActionEnum.Check)
        return UTHAction(UTHActionEnum.Fold)

    def get_legal_action_types(self, state):
        legal_action_types = []
        if UTHActionEnum.Fold in state.legal_actions:
            legal_action_types.append(0)
        if UTHActionEnum.Check in state.legal_actions:
            legal_action_types.append(1)
        if UTHActionEnum.Raise in state.legal_actions:
            legal_action_types.append(2)
        return legal_action_types

    # -- CFR traversal ---------------------------------------------------------

    def cfr_traverse(self, state, iteration, depth=0):
        """
        External-sampling CFR traversal. UTH has a single decision maker
        (chance auto-advances inside apply_action), so every non-terminal
        node belongs to the trained agent.
        """
        max_depth = 50
        if depth > max_depth:
            if _model.VERBOSE:
                print(f"WARNING: UTH max recursion depth reached ({max_depth}).")
            return 0

        if state.final_state:
            return state.players_state[self.player_id].reward

        legal_action_types = self.get_legal_action_types(state)
        if not legal_action_types:
            if _model.VERBOSE:
                print(f"WARNING: No legal UTH actions at depth {depth}")
            return 0

        state_encoding = encode_uth_state(state, self.player_id)
        state_tensor = torch.FloatTensor(state_encoding).to(self.device)

        with torch.no_grad():
            advantages, bet_size_pred = self.advantage_net(state_tensor.unsqueeze(0))
            advantages = advantages[0].cpu().numpy()
            bet_size_multiplier = bet_size_pred[0][0].item()

        advantages_masked = np.zeros(self.num_actions)
        for a in legal_action_types:
            advantages_masked[a] = max(advantages[a], 0)

        if sum(advantages_masked) > 0:
            strategy = advantages_masked / sum(advantages_masked)
        else:
            strategy = np.zeros(self.num_actions)
            for a in legal_action_types:
                strategy[a] = 1.0 / len(legal_action_types)

        action_values = np.zeros(self.num_actions)
        for action_type in legal_action_types:
            try:
                if action_type == 2:
                    uth_action = self.action_type_to_uth_action(
                        action_type, state, bet_size_multiplier
                    )
                else:
                    uth_action = self.action_type_to_uth_action(action_type, state)

                new_state = state.apply_action(uth_action)

                if new_state.status != UTHStatus.Ok:
                    if STRICT_CHECKING:
                        raise ValueError(
                            f"UTH state status not OK ({new_state.status}) "
                            f"during CFR traversal."
                        )
                    if _model.VERBOSE:
                        print(f"WARNING: Invalid UTH action {action_type} at depth {depth}.")
                    continue

                action_values[action_type] = self.cfr_traverse(
                    new_state, iteration, depth + 1
                )
            except Exception:
                if STRICT_CHECKING:
                    raise
                action_values[action_type] = 0

        ev = sum(strategy[a] * action_values[a] for a in legal_action_types)
        max_abs_val = max(abs(max(action_values)), abs(min(action_values)), 1.0)

        for action_type in legal_action_types:
            regret = action_values[action_type] - ev
            normalized_regret = regret / max_abs_val
            clipped_regret = np.clip(normalized_regret, -10.0, 10.0)
            scale_factor = np.sqrt(iteration) if iteration > 1 else 1.0  # Linear CFR
            weighted_regret = clipped_regret * scale_factor
            priority = abs(weighted_regret) + 0.01

            self.advantage_memory.add(
                (state_encoding,
                 np.zeros(20),  # placeholder for opponent features
                 action_type,
                 bet_size_multiplier if action_type == 2 else 0.0,
                 weighted_regret),
                priority,
            )

        strategy_full = np.zeros(self.num_actions)
        for a in legal_action_types:
            strategy_full[a] = strategy[a]

        self.strategy_memory.append((
            state_encoding,
            np.zeros(20),  # placeholder for opponent features
            strategy_full,
            bet_size_multiplier if 2 in legal_action_types else 0.0,
            iteration,
        ))

        return ev

    # -- network training ------------------------------------------------------

    def train_advantage_network(self, batch_size=128, epochs=3,
                                beta_start=0.4, beta_end=1.0):
        """Prioritized experience replay training (mirrors DeepCFRAgent)."""
        if len(self.advantage_memory) < batch_size:
            return 0

        self.advantage_net.train()
        total_loss = 0

        progress = min(1.0, self.iteration_count / 10000)
        beta = beta_start + progress * (beta_end - beta_start)

        for _ in range(epochs):
            batch, indices, weights = self.advantage_memory.sample(batch_size, beta=beta)
            states, opponent_features, action_types, bet_sizes, regrets = zip(*batch)

            state_tensors = torch.FloatTensor(np.array(states)).to(self.device)
            opponent_feature_tensors = torch.FloatTensor(np.array(opponent_features)).to(self.device)
            action_type_tensors = torch.LongTensor(np.array(action_types)).to(self.device)
            bet_size_tensors = torch.FloatTensor(np.array(bet_sizes)).unsqueeze(1).to(self.device)
            regret_tensors = torch.FloatTensor(np.array(regrets)).to(self.device)
            weight_tensors = torch.FloatTensor(weights).to(self.device)

            action_advantages, bet_size_preds = self.advantage_net(
                state_tensors, opponent_feature_tensors
            )
            predicted_regrets = action_advantages.gather(
                1, action_type_tensors.unsqueeze(1)
            ).squeeze(1)

            action_loss = F.smooth_l1_loss(predicted_regrets, regret_tensors, reduction='none')
            weighted_action_loss = (action_loss * weight_tensors).mean()

            raise_mask = (action_type_tensors == 2)
            if torch.any(raise_mask):
                all_bet_losses = F.smooth_l1_loss(bet_size_preds, bet_size_tensors, reduction='none')
                masked_bet_losses = all_bet_losses * raise_mask.float().unsqueeze(1)
                raise_count = raise_mask.sum().item()
                if raise_count > 0:
                    weighted_bet_size_loss = (
                        masked_bet_losses.squeeze(1) * weight_tensors
                    ).sum() / raise_count
                    combined_loss = weighted_action_loss + 0.5 * weighted_bet_size_loss
                else:
                    combined_loss = weighted_action_loss
            else:
                combined_loss = weighted_action_loss

            self.optimizer.zero_grad()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.advantage_net.parameters(), max_norm=0.5)
            self.optimizer.step()

            with torch.no_grad():
                new_errors = F.smooth_l1_loss(predicted_regrets, regret_tensors, reduction='none')
                new_errors_np = new_errors.cpu().numpy()
                for i, idx in enumerate(indices):
                    self.advantage_memory.update_priority(idx, new_errors_np[i] + 0.01)

            total_loss += combined_loss.item()

        return total_loss / epochs

    def train_strategy_network(self, batch_size=128, epochs=3):
        """Train the strategy network (mirrors DeepCFRAgent)."""
        if len(self.strategy_memory) < batch_size:
            return 0

        self.strategy_net.train()
        total_loss = 0

        for _ in range(epochs):
            batch = random.sample(self.strategy_memory, batch_size)
            states, opponent_features, strategies, bet_sizes, iterations = zip(*batch)

            state_tensors = torch.FloatTensor(np.array(states)).to(self.device)
            strategy_tensors = torch.FloatTensor(np.array(strategies)).to(self.device)
            bet_size_tensors = torch.FloatTensor(np.array(bet_sizes)).unsqueeze(1).to(self.device)
            iteration_tensors = torch.FloatTensor(iterations).to(self.device).unsqueeze(1)

            weights = iteration_tensors / torch.sum(iteration_tensors)

            action_logits, bet_size_preds = self.strategy_net(state_tensors)
            predicted_strategies = F.softmax(action_logits, dim=1)

            action_loss = -torch.sum(
                weights * torch.sum(
                    strategy_tensors * torch.log(predicted_strategies + 1e-8), dim=1
                )
            )

            raise_mask = (strategy_tensors[:, 2] > 0)
            if raise_mask.sum() > 0:
                raise_indices = torch.nonzero(raise_mask).squeeze(1)
                raise_bet_preds = bet_size_preds[raise_indices]
                raise_bet_targets = bet_size_tensors[raise_indices]
                raise_weights = weights[raise_indices]
                bet_size_loss = F.smooth_l1_loss(
                    raise_bet_preds, raise_bet_targets, reduction='none'
                )
                weighted_bet_size_loss = torch.sum(raise_weights * bet_size_loss.squeeze(1))
                combined_loss = action_loss + 0.5 * weighted_bet_size_loss
            else:
                combined_loss = action_loss

            self.strategy_optimizer.zero_grad()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.strategy_net.parameters(), max_norm=0.5)
            self.strategy_optimizer.step()

            total_loss += combined_loss.item()

        return total_loss / epochs

    # -- play / persistence ------------------------------------------------------

    def choose_action(self, state):
        """Choose a UTHAction for actual play using the strategy network."""
        legal_action_types = self.get_legal_action_types(state)
        if not legal_action_types:
            if UTHActionEnum.Check in state.legal_actions:
                return UTHAction(UTHActionEnum.Check)
            return UTHAction(UTHActionEnum.Fold)

        state_tensor = torch.FloatTensor(
            encode_uth_state(state, self.player_id)
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, bet_size_pred = self.strategy_net(state_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            bet_size_multiplier = bet_size_pred[0][0].item()

        legal_probs = np.array([probs[a] for a in legal_action_types])
        if np.sum(legal_probs) > 0:
            legal_probs = legal_probs / np.sum(legal_probs)
        else:
            legal_probs = np.ones(len(legal_action_types)) / len(legal_action_types)

        action_idx = np.random.choice(len(legal_action_types), p=legal_probs)
        action_type = legal_action_types[action_idx]

        if action_type == 2:
            return self.action_type_to_uth_action(action_type, state, bet_size_multiplier)
        return self.action_type_to_uth_action(action_type, state)

    def save_model(self, path_prefix):
        """Save the model with the same checkpoint layout as DeepCFRAgent."""
        model_path = _resolve_model_save_path(path_prefix, self.iteration_count)
        torch.save({
            'iteration': self.iteration_count,
            'advantage_net': self.advantage_net.state_dict(),
            'strategy_net': self.strategy_net.state_dict(),
            'min_bet_size': self.min_bet_size,
            'max_bet_size': self.max_bet_size,
        }, model_path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.iteration_count = checkpoint['iteration']
        self.advantage_net.load_state_dict(checkpoint['advantage_net'])
        self.strategy_net.load_state_dict(checkpoint['strategy_net'])
        if 'min_bet_size' in checkpoint:
            self.min_bet_size = checkpoint['min_bet_size']
        if 'max_bet_size' in checkpoint:
            self.max_bet_size = checkpoint['max_bet_size']
