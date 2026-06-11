# src/training/train_uth.py
"""
Training script for the Ultimate Texas Hold'em Deep CFR agent.

Mirrors the structure of src/training/train.py (iteration loop, TensorBoard
logging, periodic checkpoints and evaluation) but runs entirely on the
pure-Python UTH environment in src/envs/uth_env.py, so both training
traversals AND evaluation use the same UTH game - no State/BonusState
inconsistency.

Usage:
    python -m src.training.train_uth --iterations 200 --traversals-per-iteration 100
"""

import argparse
import os
import random
import time

import numpy as np
import torch

from src.core.uth_deep_cfr import UTHDeepCFRAgent
from src.envs.uth_env import DEFAULT_PAYTABLE, UTHActionEnum, UTHState
from src.envs.uth_paytables import TRIPS_PAYTABLES


def evaluate_agent(agent, num_games=500, ante=10.0, trips_bet=0.0,
                   stake=1000.0, seed_offset=1_000_000):
    """
    Evaluate the agent's strategy network over `num_games` fresh deals.

    Returns average profit per hand (in chips). Uses seeds disjoint from
    training seeds via `seed_offset`.
    """
    total_profit = 0.0
    completed = 0
    for game in range(num_games):
        state = UTHState.from_seed(
            ante=ante, trips_bet=trips_bet, stake=stake,
            seed=seed_offset + game, paytable=agent.paytable,
        )
        steps = 0
        while not state.final_state and steps < 10:
            action = agent.choose_action(state, use_strategy_net=True, greedy=True)
            next_state = state.apply_action(action)
            if next_state.status.value != 0:
                # Defensive fallback: pick the first legal action
                next_state = state.apply_action(state.legal_actions[0])
            state = next_state
            steps += 1
        total_profit += state.reward
        completed += 1
    return total_profit / max(completed, 1)


def evaluate_baseline_random(num_games=500, ante=10.0, trips_bet=0.0,
                             stake=1000.0, seed_offset=1_000_000):
    """Average profit per hand for a uniform-random policy (baseline)."""
    rng = random.Random(0)
    total_profit = 0.0
    for game in range(num_games):
        state = UTHState.from_seed(
            ante=ante, trips_bet=trips_bet, stake=stake, seed=seed_offset + game,
        )
        steps = 0
        while not state.final_state and steps < 10:
            action = rng.choice(state.legal_actions)
            state = state.apply_action(action)
            steps += 1
        total_profit += state.reward
    return total_profit / max(num_games, 1)


def train_uth_deep_cfr(
    iterations=100,
    traversals_per_iteration=200,
    ante=10.0,
    trips_bet=0.0,
    stake=1000.0,
    paytable=DEFAULT_PAYTABLE,
    save_dir="models_uth",
    log_dir=None,
    eval_every=10,
    eval_games=500,
    seed=42,
    verbose=False,
    writer=None,
):
    """
    Main UTH Deep CFR training loop.

    Returns (agent, profits) where profits is the list of evaluation results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(save_dir, exist_ok=True)

    if writer is None:
        from torch.utils.tensorboard import SummaryWriter
        if log_dir is None:
            log_dir = os.path.join("logs", f"uth_deepcfr_{int(time.time())}")
        writer = SummaryWriter(log_dir=log_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = UTHDeepCFRAgent(paytable=paytable, device=device)

    profits = []
    for iteration in range(1, iterations + 1):
        agent.iteration_count = iteration
        if verbose:
            print(f"Iteration {iteration}/{iterations}")

        # Collect data: every traversal is a fresh seeded deal
        for _ in range(traversals_per_iteration):
            state = UTHState.from_seed(
                ante=ante, trips_bet=trips_bet, stake=stake,
                seed=random.randint(0, 10_000_000), paytable=paytable,
            )
            agent.cfr_traverse(state, iteration)

        adv_loss = agent.train_advantage_network()
        strat_loss = agent.train_strategy_network()

        writer.add_scalar("Loss/Advantage", adv_loss, iteration)
        writer.add_scalar("Loss/Strategy", strat_loss, iteration)
        writer.add_scalar("Memory/Advantage", len(agent.advantage_memory), iteration)
        writer.add_scalar("Memory/Strategy", len(agent.strategy_memory), iteration)

        if verbose:
            print(f"  adv_loss={adv_loss:.4f} strat_loss={strat_loss:.4f}")

        if iteration % eval_every == 0 or iteration == iterations:
            avg_profit = evaluate_agent(
                agent, num_games=eval_games, ante=ante,
                trips_bet=trips_bet, stake=stake,
            )
            profits.append(avg_profit)
            writer.add_scalar("Performance/ProfitPerHand", avg_profit, iteration)
            if verbose:
                print(f"  eval profit/hand: {avg_profit:.3f}")

            checkpoint_path = os.path.join(save_dir, f"uth_checkpoint_iter_{iteration}")
            agent.save_model(checkpoint_path)

    final_path = os.path.join(save_dir, "uth_final")
    agent.save_model(final_path)
    writer.close()
    return agent, profits


def main():
    parser = argparse.ArgumentParser(
        description="Train a Deep CFR agent for Ultimate Texas Hold'em"
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--traversals-per-iteration", type=int, default=200)
    parser.add_argument("--ante", type=float, default=10.0)
    parser.add_argument("--trips-bet", type=float, default=0.0,
                        help="Optional Trips side bet size (0 = no Trips bet)")
    parser.add_argument("--stake", type=float, default=1000.0)
    parser.add_argument("--paytable", type=str, default=DEFAULT_PAYTABLE,
                        choices=sorted(TRIPS_PAYTABLES))
    parser.add_argument("--save-dir", type=str, default="models_uth")
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Training UTH Deep CFR: {args.iterations} iterations, "
          f"{args.traversals_per_iteration} traversals/iter, "
          f"ante={args.ante}, trips={args.trips_bet}, paytable={args.paytable}")

    baseline = evaluate_baseline_random(
        num_games=args.eval_games, ante=args.ante, trips_bet=args.trips_bet,
        stake=args.stake,
    )
    print(f"Random baseline profit/hand: {baseline:.3f}")

    agent, profits = train_uth_deep_cfr(
        iterations=args.iterations,
        traversals_per_iteration=args.traversals_per_iteration,
        ante=args.ante,
        trips_bet=args.trips_bet,
        stake=args.stake,
        paytable=args.paytable,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        seed=args.seed,
        verbose=args.verbose,
    )

    if profits:
        print(f"Final eval profit/hand: {profits[-1]:.3f} "
              f"(random baseline: {baseline:.3f})")
    print(f"Models saved to {args.save_dir}/")


if __name__ == "__main__":
    main()
