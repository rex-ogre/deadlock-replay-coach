"""A small synthetic match, hand-built so every assertion has a known answer.

Six heroes, two teams, one 25-minute game. Ticks are 60/s, so tick 36000 is
10:00 — the fixtures use round minute marks to keep the tests readable.
"""

from __future__ import annotations

import polars as pl
import pytest

from deadlock_coach import gamedata
from deadlock_coach.match import SCHEMAS, MatchData
from deadlock_coach.names import Names

@pytest.fixture(autouse=True)
def _no_network(monkeypatch, tmp_path):
    """Keep the suite off the Deadlock API.

    Movement constants and accuracy stats are fetched live in normal use. Tests
    must not depend on a network, on rate limits, or on whichever patch shipped
    this week, so every test sees the bundled snapshot and no match record.

    The cache directory is redirected too: an offline load still prefers a
    stale on-disk cache over the bundled snapshot, and a developer's own cache
    would otherwise decide what the population baseline looks like.
    """
    monkeypatch.setenv("DEADLOCK_COACH_CACHE", str(tmp_path / "cache"))
    gamedata.reset_cache()
    real = gamedata.load_constants
    monkeypatch.setattr(
        gamedata, "load_constants", lambda **kwargs: real(offline=True)
    )
    for module in ("tactics", "opportunities", "render", "cli"):
        target = f"deadlock_coach.{module}.load_constants"
        try:
            monkeypatch.setattr(target, lambda **kwargs: real(offline=True))
        except AttributeError:
            pass
    yield
    gamedata.reset_cache()


TICK_RATE = 60
MINUTE = 60 * TICK_RATE

TEAM_A = 2  # Hidden King
TEAM_B = 3  # Archmother
A_HEROES = [1, 2, 3]
B_HEROES = [11, 12, 13]


def frame(name: str, rows: list[dict]) -> pl.DataFrame:
    """Build a frame with the declared schema, filling absent columns with null."""
    schema = SCHEMAS[name]
    if not rows:
        return pl.DataFrame(schema=schema)
    normalized = [{col: row.get(col) for col in schema} for row in rows]
    return pl.DataFrame(normalized, schema=schema)


@pytest.fixture
def names() -> Names:
    return Names(
        heroes={1: "Infernus", 2: "Seven", 3: "Vindicta", 11: "Abrams", 12: "Lash", 13: "Haze"},
        teams={1: "Spectator", TEAM_A: "Hidden King", TEAM_B: "Archmother"},
    )


@pytest.fixture
def players() -> pl.DataFrame:
    rows = []
    for i, hero in enumerate(A_HEROES):
        rows.append(
            {
                "player_name": f"a{i}",
                "steam_id": 100 + i,
                "hero_id": hero,
                "team_num": TEAM_A,
                "start_lane": [1, 3, 4][i],
                "rank": 0,
            }
        )
    for i, hero in enumerate(B_HEROES):
        rows.append(
            {
                "player_name": f"b{i}",
                "steam_id": 200 + i,
                "hero_id": hero,
                "team_num": TEAM_B,
                "start_lane": [1, 3, 4][i],
                "rank": 0,
            }
        )
    return frame("players", rows)


@pytest.fixture
def kills() -> pl.DataFrame:
    """Nine kills: an early solo pickoff, a laning trade, two clustered fights."""
    return frame(
        "kills",
        [
            # 05:00 — team A picks off a lone B hero 3v1.
            {"tick": 5 * MINUTE, "victim_hero_id": 11, "attacker_hero_id": 1,
             "assister_hero_ids": [2, 3]},
            # 07:00 — an even 1v1 trade in lane.
            {"tick": 7 * MINUTE, "victim_hero_id": 2, "attacker_hero_id": 12,
             "assister_hero_ids": []},
            # 09:30 — hero 13 rotates in from the blue lane onto hero 2 in green.
            # This is the first cross-lane kill, so it ends the laning phase.
            {"tick": 9 * MINUTE + 30 * TICK_RATE, "victim_hero_id": 2,
             "attacker_hero_id": 13, "assister_hero_ids": [12]},
            # 12:00 fight — A wins 2-1.
            {"tick": 12 * MINUTE, "victim_hero_id": 11, "attacker_hero_id": 1,
             "assister_hero_ids": [2]},
            {"tick": 12 * MINUTE + 120, "victim_hero_id": 3, "attacker_hero_id": 12,
             "assister_hero_ids": []},
            {"tick": 12 * MINUTE + 300, "victim_hero_id": 12, "attacker_hero_id": 1,
             "assister_hero_ids": [2]},
            # 20:00 fight — B wins 2-0.
            {"tick": 20 * MINUTE, "victim_hero_id": 1, "attacker_hero_id": 11,
             "assister_hero_ids": [12, 13]},
            {"tick": 20 * MINUTE + 180, "victim_hero_id": 2, "attacker_hero_id": 11,
             "assister_hero_ids": []},
            # 22:00 — an environment death, no killer credited.
            {"tick": 22 * MINUTE, "victim_hero_id": 3, "attacker_hero_id": 3,
             "assister_hero_ids": []},
        ],
    )


@pytest.fixture
def objectives() -> pl.DataFrame:
    """Team B loses a Walker at 15:00 and a Barracks at 21:00.

    Structure names and health values follow build 10854 (verified against match
    12345678): Walker 12000 hp is the lane tower, Barracks 4000 hp sits behind
    it, and there is no Guardian at all.

    The Walker reports two zero-health rows, so the "first zero only" rule in
    `objective_events` has something to actually collapse.
    """
    return frame(
        "objectives",
        [
            {"tick": 1 * MINUTE, "objective_type": "walker", "team_num": TEAM_B, "lane": 1,
             "health": 12000, "max_health": 12000, "phase": 0, "entity_id": 500},
            {"tick": 15 * MINUTE, "objective_type": "walker", "team_num": TEAM_B, "lane": 1,
             "health": 0, "max_health": 12000, "phase": 0, "entity_id": 500},
            {"tick": 15 * MINUTE + 60, "objective_type": "walker", "team_num": TEAM_B,
             "lane": 1, "health": 0, "max_health": 12000, "phase": 0, "entity_id": 500},
            {"tick": 21 * MINUTE, "objective_type": "barracks", "team_num": TEAM_B, "lane": 1,
             "health": 0, "max_health": 4000, "phase": 0, "entity_id": 501},
            # Patron of team B flips to its final phase near the end.
            {"tick": 24 * MINUTE, "objective_type": "patron", "team_num": TEAM_B, "lane": 0,
             "health": 6000, "max_health": 6000, "phase": 0, "entity_id": 502},
            {"tick": 24 * MINUTE + 600, "objective_type": "patron", "team_num": TEAM_B,
             "lane": 0, "health": 3000, "max_health": 6000, "phase": 1, "entity_id": 502},
        ],
    )


@pytest.fixture
def teamfights() -> pl.DataFrame:
    return frame(
        "teamfights",
        [
            {"fight_id": 1, "start_tick": 12 * MINUTE - 60, "end_tick": 12 * MINUTE + 360,
             "start_seconds": 719.0, "end_seconds": 726.0, "duration_seconds": 7.0,
             "center_x": 0.0, "center_y": 0.0, "participants": [1, 2, 3, 11, 12],
             "num_participants": 5, "hero_damage": 4200, "kills": 3},
            {"fight_id": 2, "start_tick": 20 * MINUTE - 60, "end_tick": 20 * MINUTE + 240,
             "start_seconds": 1199.0, "end_seconds": 1204.0, "duration_seconds": 5.0,
             "center_x": 100.0, "center_y": 100.0, "participants": [1, 2, 11, 12, 13],
             "num_participants": 5, "hero_damage": 3800, "kills": 2},
        ],
    )


def _sample_rows(tick: int, positions: dict[int, tuple[float, float]], souls: int) -> list[dict]:
    rows = []
    for hero, (x, y) in positions.items():
        rows.append(
            {
                "tick": tick,
                "hero_id": hero,
                "x": x,
                "y": y,
                "z": 0.0,
                "is_alive": True,
                "health": 800,
                "max_health": 800,
                "souls": souls,
                "spent_souls": souls * 2,
                "level": 10,
                "kills": 0,
                "deaths": 0,
                "assists": 0,
                "last_hits": 100,
                "denies": 10,
                "hero_damage": 5000,
                "objective_damage": 2000,
            }
        )
    return rows


@pytest.fixture
def player_samples() -> pl.DataFrame:
    """Positions at each kill tick, arranged so the classifications are known.

    At 05:00 hero 11 is alone (allies 8000 units away) and dies to three
    attackers -> isolated and outnumbered. At 07:00 hero 2 dies next to an ally
    with a single attacker -> neither.
    """
    far = 8000.0
    rows: list[dict] = []

    # 05:00 — victim 11 alone; its allies parked far away.
    rows += _sample_rows(
        5 * MINUTE,
        {1: (0, 0), 2: (10, 0), 3: (20, 0), 11: (0, 0), 12: (far, far), 13: (far, -far)},
        3000,
    )
    # 07:00 — victim 2 stands next to ally 1; single attacker.
    rows += _sample_rows(
        7 * MINUTE,
        {1: (0, 0), 2: (100, 0), 3: (far, far), 11: (far, 0), 12: (200, 0), 13: (far, far)},
        4000,
    )
    # 20:00 — victim 1 alone against three.
    rows += _sample_rows(
        20 * MINUTE,
        {1: (0, 0), 2: (far, far), 3: (far, -far), 11: (50, 0), 12: (60, 0), 13: (70, 0)},
        9000,
    )
    return frame("player_samples", rows)


@pytest.fixture
def match(players, kills, objectives, teamfights, player_samples) -> MatchData:
    return MatchData(
        map_name="dl_midtown",
        match_id=28309863,
        tick_rate=TICK_RATE,
        total_ticks=25 * MINUTE,
        game_over_tick=25 * MINUTE,
        winning_team_num=TEAM_A,
        players=players,
        kills=kills,
        objectives=objectives,
        teamfights=teamfights,
        player_samples=player_samples,
    )


@pytest.fixture
def empty_match() -> MatchData:
    """A demo that parsed but carried nothing useful — the degraded path."""
    return MatchData(tick_rate=TICK_RATE, total_ticks=1000)
