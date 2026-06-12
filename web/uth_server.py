# web/uth_server.py
"""
Interactive Ultimate Texas Hold'em web game.

Run:
    .venv/bin/python -m uvicorn web.uth_server:app --host 0.0.0.0 --port 8000

Serves a single-page casino interface backed by the pure-Python UTH engine
in src/envs/uth_env.py. Game sessions are kept in memory, keyed by a token
returned to the browser.
"""

import random
import secrets
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.envs.uth_env import (
    UTHActionEnum,
    UTHStage,
    UTHState,
)
from src.envs.uth_paytables import (
    BLIND_PAYTABLE,
    DEFAULT_PAYTABLE,
    HAND_CATEGORY_NAMES,
    get_trips_paytable,
)

app = FastAPI(title="Ultimate Texas Hold'em")

STATIC_DIR = Path(__file__).parent / "static"

# token -> {"state": UTHState, "bankroll": float, "hands": int}
SESSIONS = {}
MAX_SESSIONS = 200

ACTION_NAMES = {a.name: a for a in UTHActionEnum}

SUIT_SYMBOLS = {"c": "\u2663", "d": "\u2666", "h": "\u2665", "s": "\u2660"}


def card_to_dict(card):
    rank_str = "23456789TJQKA"[card.rank - 2]
    suit = "cdhs"[card.suit]
    return {
        "rank": "10" if rank_str == "T" else rank_str,
        "suit": SUIT_SYMBOLS[suit],
        "red": suit in ("d", "h"),
    }


def state_to_dict(token, session):
    state = session["state"]
    payload = {
        "token": token,
        "bankroll": round(session["bankroll"], 2),
        "hands": session["hands"],
        "ante": state.ante,
        "blind": state.blind,
        "trips_bet": state.trips_bet,
        "play_bet": state.play_bet,
        "stage": state.stage.name,
        "done": state.final_state,
        "player_hand": [card_to_dict(c) for c in state.player_hand],
        "public_cards": [card_to_dict(c) for c in state.public_cards],
        "legal_actions": [a.name for a in state.legal_actions],
    }
    if state.dealer_revealed:
        payload["dealer_hand"] = [card_to_dict(c) for c in state.dealer_hand]
    else:
        payload["dealer_hand"] = None

    if state.final_state and state.info:
        info = state.info
        payload["result"] = {
            "player_hand_name": HAND_CATEGORY_NAMES[info["player_eval"][0]],
            "dealer_hand_name": HAND_CATEGORY_NAMES[info["dealer_eval"][0]],
            "dealer_qualifies": info["dealer_qualifies"],
            "folded": info["folded"],
            "ante_net": info["ante_net"],
            "blind_net": info["blind_net"],
            "play_net": info["play_net"],
            "trips_net": info["trips_net"],
            "total": round(state.reward, 2),
        }
    else:
        payload["result"] = None
    return payload


class NewHandRequest(BaseModel):
    token: str | None = None
    ante: float = 10.0
    trips_bet: float = 0.0


class ActionRequest(BaseModel):
    token: str
    action: str


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/paytables")
def paytables():
    return {
        "blind": {HAND_CATEGORY_NAMES[k]: v for k, v in sorted(BLIND_PAYTABLE.items(), reverse=True)},
        "trips": {
            HAND_CATEGORY_NAMES[k]: v
            for k, v in sorted(get_trips_paytable(DEFAULT_PAYTABLE).items(), reverse=True)
        },
        "paytable": DEFAULT_PAYTABLE,
    }


@app.post("/api/new")
def new_hand(req: NewHandRequest):
    if req.ante <= 0 or req.ante > 10000:
        raise HTTPException(400, "Ante must be between 0 and 10000.")
    if req.trips_bet < 0 or req.trips_bet > 10000:
        raise HTTPException(400, "Trips bet must be between 0 and 10000.")

    if req.token and req.token in SESSIONS:
        token = req.token
        session = SESSIONS[token]
    else:
        if len(SESSIONS) >= MAX_SESSIONS:
            SESSIONS.pop(next(iter(SESSIONS)))
        token = secrets.token_urlsafe(16)
        session = {"state": None, "bankroll": 1000.0, "hands": 0}
        SESSIONS[token] = session

    cost = req.ante * 2 + req.trips_bet  # worst-case immediate exposure: ante+blind+trips
    if session["bankroll"] < cost + req.ante * 4:
        # Need room for the maximum 4x play bet too
        if session["bankroll"] < cost:
            raise HTTPException(400, "Insufficient bankroll for ante, blind and trips.")

    session["state"] = UTHState.from_seed(
        ante=req.ante,
        trips_bet=req.trips_bet,
        stake=session["bankroll"],
        seed=random.SystemRandom().randrange(2**63),
    )
    session["hands"] += 1
    return state_to_dict(token, session)


@app.post("/api/action")
def take_action(req: ActionRequest):
    session = SESSIONS.get(req.token)
    if session is None or session["state"] is None:
        raise HTTPException(404, "Session not found; start a new hand.")
    state = session["state"]
    if state.final_state:
        raise HTTPException(400, "Hand already finished; start a new hand.")
    action = ACTION_NAMES.get(req.action)
    if action is None or action not in state.legal_actions:
        raise HTTPException(
            400,
            f"Illegal action '{req.action}' at stage {state.stage.name}. "
            f"Legal: {[a.name for a in state.legal_actions]}",
        )
    new_state = state.apply_action(action)
    session["state"] = new_state
    if new_state.final_state:
        session["bankroll"] = round(session["bankroll"] + new_state.reward, 2)
    return state_to_dict(req.token, session)
