# src/core/uth_deep_cfr.py
"""
Deep CFR agent for Ultimate Texas Hold'em.

UTH is a single-player game against the house (all "opponent" behaviour is
chance + fixed dealer rules), so the CFR traversal enumerates the player's
own actions exactly at every decision node of a sampled deal and uses
regret matching over advantage-network outputs, mirroring the conventions
of `src/core/deep_cfr.py` (linear CFR weighting, advantage/strategy
memories, defensive fallbacks on illegal actions).

The discrete action space is the 6-way `UTHActionEnum`:
    0=Fold, 1=Check, 2=Bet1x, 3=Bet2x, 4=Bet3x, 5=Bet4x
with per-stage legality masks supplied by the environment.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.core.model import PokerNetwork
from src.envs.uth_env import (
    NUM_UTH_ACTIONS,
    UTHActionEnum,
    UTHStage,
    UTHState,
    UTHStatus,
)

# Encoded state size:
#   52 (hole cards) + 52 (board) + 4 (stage one-hot)
#   + 1 (play bet / ante, normalized by max 4x) + 1 (trips bet flag)
#   + 6 (legal action mask)
UTH_INPUT_SIZE = 52 + 52 + 4 + 1 + 1 + NUM_UTH_ACTIONS


def encode_uth_state(state):
    """Convert a UTHState into a fixed-size numpy feature vector."""
    encoded = []

    hand_enc = np.zeros(52)
    for card in state.player_hand:
        hand_enc[card.index()] = 1
    encoded.append(hand_enc)

    board_enc = np.zeros(52)
    for card in state.public_cards:
        board_enc[card.index()] = 1
    encoded.append(board_enc)

    stage_enc = np.zeros(4)
    stage_enc[int(state.stage)] = 1
    encoded.append(stage_enc)

    encoded.append([state.play_bet / (4.0 * state.ante) if state.ante > 0 else 0.0])
    encoded.append([1.0 if state.trips_bet > 0 else 0.0])

    legal_enc = np.zeros(NUM_UTH_ACTIONS)
    for action in state.legal_actions:
        legal_enc[int(action)] = 1
    encoded.append(legal_enc)

    return np.concatenate(encoded)


def legal_action_indices(state):
    return [int(a) for a in state.legal_actions]


class UTHDeepCFRAgent:
    """Deep CFR agent over the 6-way UTH action space."""

    def __init__(self, paytable="UTH-01", memory_size=200000, device="cpu"):
        self.paytable = paytable
        self.device = device
        self.num_actions = NUM_UTH_ACTIONS

        self.advantage_net = PokerNetwork(
            input_size=UTH_INPUT_SIZE, hidden_size=128, num_actions=self.num_actions
        ).to(device)
        self.strategy_net = PokerNetwork(
            input_size=UTH_INPUT_SIZE, hidden_size=128, num_actions=self.num_actions
        ).to(device)

        self.optimizer = optim.Adam(
            self.advantage_net.parameters(), lr=1e-4, weight_decay=1e-5
        )
        self.strategy_optimizer = optim.Adam(
            self.strategy_net.parameters(), lr=1e-4, weight_decay=1e-5
        )

        # (encoded_state, legal_mask, regrets, iteration)
        self.advantage_memory = deque(maxlen=memory_size)
        # (encoded_state, legal_mask, strategy, iteration)
        self.strategy_memory = deque(maxlen=memory_size)

        self.iteration_count = 0

    # -- strategy computation ------------------------------------------------

    def _regret_matching(self, state):
        """Regret matching over advantage net outputs, masked to legal actions."""
        legal = legal_action_indices(state)
        encoded = encode_uth_state(state)
        state_tensor = torch.tensor(
            encoded, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            advantages, _ = self.advantage_net(state_tensor)
        advantages = advantages[0].cpu().numpy()

        strategy = np.zeros(self.num_actions)
        positive = np.array([max(advantages[a], 0.0) for a in legal])
        if positive.sum() > 1e-12:
            for i, a in enumerate(legal):
                strategy[a] = positive[i] / positive.sum()
        else:
            for a in legal:
                strategy[a] = 1.0 / len(legal)
        return encoded, strategy, legal

    # -- CFR traversal -------------------------------------------------------

    def cfr_traverse(self, state, iteration):
        """
        Traverse a sampled deal, enumerating all of the player's actions.

        Returns the expected value of `state` under the current strategy.
        """
        if state.final_state:
            return state.reward

        encoded, strategy, legal = self._regret_matching(state)

        action_values = np.zeros(self.num_actions)
        for a in legal:
            next_state = state.apply_action(UTHActionEnum(a))
            if next_state.status != UTHStatus.Ok:
                # Defensive: treat engine-rejected actions as immediate fold-like loss
                action_values[a] = -(state.ante + state.blind + state.trips_bet)
                continue
            action_values[a] = self.cfr_traverse(next_state, iteration)

        ev = sum(strategy[a] * action_values[a] for a in legal)

        regrets = np.zeros(self.num_actions)
        legal_mask = np.zeros(self.num_actions)
        for a in legal:
            regrets[a] = action_values[a] - ev
            legal_mask[a] = 1.0

        # Linear CFR weighting, as in deep_cfr.py
        self.advantage_memory.append((encoded, legal_mask, regrets, iteration))
        self.strategy_memory.append((encoded, legal_mask, strategy.copy(), iteration))
        return ev

    # -- training ------------------------------------------------------------

    def _train_network(self, net, optimizer, memory, batch_size, target_index):
        if len(memory) < 2:
            return 0.0
        batch_size = min(batch_size, len(memory))
        batch = random.sample(list(memory), batch_size)

        states = torch.tensor(
            np.array([b[0] for b in batch]), dtype=torch.float32, device=self.device
        )
        masks = torch.tensor(
            np.array([b[1] for b in batch]), dtype=torch.float32, device=self.device
        )
        targets = torch.tensor(
            np.array([b[target_index] for b in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        weights = torch.tensor(
            np.array([b[3] for b in batch]), dtype=torch.float32, device=self.device
        )
        weights = weights / weights.max().clamp(min=1.0)

        outputs, _ = net(states)
        # Only legal actions contribute to the loss
        per_action = ((outputs - targets) ** 2) * masks
        per_sample = per_action.sum(dim=1) / masks.sum(dim=1).clamp(min=1.0)
        loss = (weights * per_sample).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()

    def train_advantage_network(self, batch_size=128, epochs=2):
        total = 0.0
        for _ in range(epochs):
            total += self._train_network(
                self.advantage_net, self.optimizer, self.advantage_memory,
                batch_size, target_index=2,
            )
        return total / max(epochs, 1)

    def train_strategy_network(self, batch_size=128, epochs=2):
        total = 0.0
        for _ in range(epochs):
            total += self._train_network(
                self.strategy_net, self.strategy_optimizer, self.strategy_memory,
                batch_size, target_index=2,
            )
        return total / max(epochs, 1)

    # -- acting --------------------------------------------------------------

    def choose_action(self, state, use_strategy_net=True, greedy=True):
        """Choose a legal UTH action; defensive fallback to a uniform legal pick."""
        legal = legal_action_indices(state)
        if not legal:
            return UTHActionEnum.Fold

        net = self.strategy_net if use_strategy_net else self.advantage_net
        encoded = encode_uth_state(state)
        state_tensor = torch.tensor(
            encoded, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            logits, _ = net(state_tensor)
        logits = logits[0].cpu().numpy()

        masked = np.full(self.num_actions, -np.inf)
        for a in legal:
            masked[a] = logits[a]

        if greedy:
            choice = int(np.argmax(masked))
        else:
            exp = np.exp(masked - masked[legal].max())
            exp[np.isinf(masked)] = 0.0
            probs = exp / exp.sum()
            choice = int(np.random.choice(self.num_actions, p=probs))

        action = UTHActionEnum(choice)
        # Defensive check, mirroring action_type_to_pokers_action fallbacks
        if action not in state.legal_actions:
            action = UTHActionEnum(random.choice(legal))
        return action

    # -- persistence ---------------------------------------------------------

    def save_model(self, path_prefix):
        torch.save(
            {
                "iteration": self.iteration_count,
                "advantage_net": self.advantage_net.state_dict(),
                "strategy_net": self.strategy_net.state_dict(),
                "paytable": self.paytable,
            },
            f"{path_prefix}.pt",
        )

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.advantage_net.load_state_dict(checkpoint["advantage_net"])
        self.strategy_net.load_state_dict(checkpoint["strategy_net"])
        self.iteration_count = checkpoint.get("iteration", 0)
        self.paytable = checkpoint.get("paytable", self.paytable)
