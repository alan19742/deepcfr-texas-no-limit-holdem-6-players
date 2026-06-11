# src/training/train_uth.py
"""
Deep CFR training for Ultimate Texas Hold'em.

Mirrors src/training/train.py (argparse / TensorBoard / checkpoints) but for
the single-player-vs-dealer UTH environment. The dealer is a fixed-rule,
non-learnable participant (see src/agents/uth_dealer_agent.py).

Quick smoke run:
    python -m src.training.train_uth --iterations 50 --traversals 20 --paytable UTH-01
"""

import argparse
import os
import time

import numpy as np
import torch

from src.core.model import set_verbose
from src.core.uth_agent import UTHDeepCFRAgent
from src.core.uth_env import DEFAULT_PAYTABLE, PAYTABLES, UTHState, UTHStatus
from src.utils.settings import STRICT_CHECKING, set_strict_checking


def evaluate_uth(agent, num_games=500, ante=10.0, bonus_bet=0.0,
                 stake=1000.0, paytable=DEFAULT_PAYTABLE, seed_offset=0):
    """
    Play `num_games` hands with the agent's current strategy network.

    Returns the average profit per hand expressed in ante units (a well
    trained agent should approach the theoretical house edge of roughly
    -2% of the ante for the base game).
    """
    total_profit = 0.0
    completed = 0

    for game in range(num_games):
        state = UTHState.from_seed(
            ante=ante,
            bonus_bet=bonus_bet,
            stake=stake,
            seed=game + seed_offset,
            paytable=paytable,
        )

        guard = 0
        while not state.final_state and guard < 10:
            action = agent.choose_action(state)
            state = state.apply_action(action)
            guard += 1
            if state.status != UTHStatus.Ok:
                if STRICT_CHECKING:
                    raise ValueError(
                        f"UTH evaluation produced status {state.status}"
                    )
                break

        if state.final_state and state.status == UTHStatus.Ok:
            total_profit += state.players_state[agent.player_id].reward
            completed += 1

    if completed == 0:
        return 0.0
    return (total_profit / completed) / ante


def train_deep_cfr_uth(num_iterations=1000, traversals_per_iteration=200,
                       ante=10.0, bonus_bet=0.0, stake=1000.0,
                       paytable=DEFAULT_PAYTABLE,
                       save_dir="models", log_dir="logs/deepcfr_uth",
                       checkpoint_path=None, num_eval_games=500,
                       verbose=False):
    """Train a Deep CFR agent for UTH. Returns (agent, losses, profits)."""
    from torch.utils.tensorboard import SummaryWriter

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    agent = UTHDeepCFRAgent(player_id=0, device=device)

    start_iteration = 1
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        agent.load_model(checkpoint_path)
        start_iteration = agent.iteration_count + 1

    losses = []
    profits = []

    initial_profit = evaluate_uth(
        agent, num_games=num_eval_games, ante=ante, bonus_bet=bonus_bet,
        stake=stake, paytable=paytable,
    )
    profits.append(initial_profit)
    writer.add_scalar('Performance/ProfitPerAnte', initial_profit, start_iteration - 1)
    print(f"Initial profit per ante: {initial_profit:.4f}")

    for iteration in range(start_iteration, start_iteration + num_iterations):
        agent.iteration_count = iteration
        start_time = time.time()

        traversal_start = time.time()
        for t in range(traversals_per_iteration):
            state = UTHState.from_seed(
                ante=ante,
                bonus_bet=bonus_bet,
                stake=stake,
                seed=iteration * traversals_per_iteration + t,
                paytable=paytable,
            )
            agent.cfr_traverse(state, iteration)
        traversal_time = time.time() - traversal_start
        writer.add_scalar('Time/Traversal', traversal_time, iteration)

        adv_loss = agent.train_advantage_network()
        losses.append(adv_loss)
        writer.add_scalar('Loss/Advantage', adv_loss, iteration)
        writer.add_scalar('Memory/Advantage', len(agent.advantage_memory), iteration)

        if iteration % 5 == 0 or iteration == start_iteration + num_iterations - 1:
            strat_loss = agent.train_strategy_network()
            writer.add_scalar('Loss/Strategy', strat_loss, iteration)

        if iteration % 10 == 0 or iteration == start_iteration + num_iterations - 1:
            avg_profit = evaluate_uth(
                agent, num_games=num_eval_games, ante=ante, bonus_bet=bonus_bet,
                stake=stake, paytable=paytable, seed_offset=10_000_000,
            )
            profits.append(avg_profit)
            writer.add_scalar('Performance/ProfitPerAnte', avg_profit, iteration)
            print(f"Iteration {iteration}: profit per ante = {avg_profit:.4f}")

        if iteration % 50 == 0 or iteration == start_iteration + num_iterations - 1:
            agent.save_model(os.path.join(save_dir, "uth_model"))

        elapsed = time.time() - start_time
        writer.add_scalar('Time/Iteration', elapsed, iteration)
        writer.add_scalar('Memory/Strategy', len(agent.strategy_memory), iteration)

        if verbose:
            print(f"Iteration {iteration} finished in {elapsed:.2f}s "
                  f"(advantage loss {adv_loss:.6f})")

    final_profit = evaluate_uth(
        agent, num_games=num_eval_games, ante=ante, bonus_bet=bonus_bet,
        stake=stake, paytable=paytable, seed_offset=20_000_000,
    )
    profits.append(final_profit)
    writer.add_scalar('Performance/FinalProfitPerAnte', final_profit, 0)
    print(f"Final profit per ante: {final_profit:.4f}")

    writer.flush()
    writer.close()
    return agent, losses, profits


def main():
    parser = argparse.ArgumentParser(
        description="Deep CFR training for Ultimate Texas Hold'em"
    )
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--iterations', type=int, default=1000, help='Number of CFR iterations')
    parser.add_argument('--traversals', type=int, default=200, help='Traversals per iteration')
    parser.add_argument('--ante', type=float, default=10.0, help='Ante (Blind is always equal)')
    parser.add_argument('--bonus-bet', type=float, default=0.0, help='Trips side bet (0 = no Trips)')
    parser.add_argument('--stake', type=float, default=1000.0, help='Player bankroll per hand')
    parser.add_argument('--paytable', type=str, default=DEFAULT_PAYTABLE,
                        choices=sorted(PAYTABLES.keys()), help='Blind/Trips paytable')
    parser.add_argument('--eval-games', type=int, default=500, help='Hands per evaluation')
    parser.add_argument('--save-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='logs/deepcfr_uth',
                        help='Directory for tensorboard logs')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to continue training from')
    parser.add_argument('--strict', action='store_true',
                        help='Enable strict error checking for invalid game states')
    args = parser.parse_args()

    set_verbose(args.verbose)
    set_strict_checking(args.strict)

    train_deep_cfr_uth(
        num_iterations=args.iterations,
        traversals_per_iteration=args.traversals,
        ante=args.ante,
        bonus_bet=args.bonus_bet,
        stake=args.stake,
        paytable=args.paytable,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        checkpoint_path=args.checkpoint,
        num_eval_games=args.eval_games,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
