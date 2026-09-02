"""The advantage ledger, on a match built so every figure has a known answer.

The fixture here is deliberately separate from ``conftest.match``: this module
needs base positions, neutral camps and map-prize datasets that the shared
fixture does not carry, and the shared fixture's counts are locked by other
tests.

Layout: team A's base sits at y=-8000 and team B's at y=+8000, so a hero's half
of the map is just the sign of y. Samples land on every whole minute, which
makes respawn gaps and window boundaries exact rather than approximate.
"""

from __future__ import annotations

import json

import pytest

from deadlock_coach import analyze_advantage
from deadlock_coach.advantage import (
    CAMP_CREDIT_RANGE,
    camp_clears,
    death_downtime,
    neutral_prizes,
    territory,
    urn_runs,
)
from deadlock_coach.match import MatchData
from deadlock_coach.names import Names
from deadlock_coach.render import render_json, render_report

from .conftest import MINUTE, TEAM_A, TEAM_B, TICK_RATE, frame

MATCH_MINUTES = 20
END = MATCH_MINUTES * MINUTE

A_HEROES = [1, 2, 3]
B_HEROES = [11, 12, 13]

A_HOME = (0.0, -3000.0)
B_HOME = (0.0, 3000.0)
A_CAMP = (0.0, -4000.0)  # sits in team A's half
B_CAMP = (0.0, 4000.0)  # sits in team B's half

# Deaths, chosen so that one respawn is measurable, one is denied mid-urn-run,
# and one lands on the final tick with no sample after it.
HERO_11_DEATH = 10 * MINUTE
HERO_1_DEATH = 16 * MINUTE + 120
HERO_12_DEATH = END

# Heroes emit no rows while dead, which is how downtime is read.
MISSING = {11: {11 * MINUTE}, 1: {17 * MINUTE}}


@pytest.fixture
def ledger_names() -> Names:
    return Names(
        heroes={1: "Infernus", 2: "Seven", 3: "Vindicta", 11: "Abrams", 12: "Lash", 13: "Haze"},
        teams={1: "Spectator", TEAM_A: "Hidden King", TEAM_B: "Archmother"},
    )


def _players():
    rows = [
        {"player_name": f"a{i}", "steam_id": 100 + i, "hero_id": hero,
         "team_num": TEAM_A, "start_lane": 1, "rank": 0}
        for i, hero in enumerate(A_HEROES)
    ]
    rows += [
        {"player_name": f"b{i}", "steam_id": 200 + i, "hero_id": hero,
         "team_num": TEAM_B, "start_lane": 1, "rank": 0}
        for i, hero in enumerate(B_HEROES)
    ]
    return frame("players", rows)


def _position(hero: int, minute: int) -> tuple[float, float]:
    """Home ground, except for the two documented invasions."""
    if hero == 11 and minute == 6:
        return A_CAMP  # B raids A's jungle
    if hero == 1 and minute == 8:
        return B_CAMP  # A raids B's jungle
    return A_HOME if hero in A_HEROES else B_HOME


def _player_samples():
    rows = []
    for minute in range(MATCH_MINUTES + 1):
        tick = minute * MINUTE
        for hero in A_HEROES + B_HEROES:
            if tick in MISSING.get(hero, set()):
                continue
            x, y = _position(hero, minute)
            # A earns 1000/min per hero, B earns 600/min: a known, steady gap.
            souls = minute * (1000 if hero in A_HEROES else 600)
            rows.append(
                {
                    "tick": tick, "hero_id": hero, "x": x, "y": y, "z": 0.0,
                    "is_alive": True, "health": 800, "max_health": 800,
                    "souls": souls, "spent_souls": 0, "level": 1 + minute // 4,
                    "kills": 0, "deaths": 0, "assists": 0,
                    "last_hits": minute * (5 if hero in A_HEROES else 3),
                    "denies": minute // 4,
                    "hero_damage": minute * 300, "objective_damage": minute * 100,
                }
            )
    return frame("player_samples", rows)


def _kills():
    return frame(
        "kills",
        [
            {"tick": HERO_11_DEATH, "victim_hero_id": 11, "attacker_hero_id": 1,
             "assister_hero_ids": []},
            {"tick": HERO_1_DEATH, "victim_hero_id": 1, "attacker_hero_id": 11,
             "assister_hero_ids": []},
            {"tick": HERO_12_DEATH, "victim_hero_id": 12, "attacker_hero_id": 2,
             "assister_hero_ids": []},
        ],
    )


def _objectives():
    return frame(
        "objectives",
        [
            {"tick": 0, "objective_type": "patron", "team_num": TEAM_A, "lane": 0,
             "health": 17000, "max_health": 17000, "phase": 0,
             "x": 0.0, "y": -8000.0, "z": 0.0, "entity_id": 400},
            {"tick": 0, "objective_type": "patron", "team_num": TEAM_B, "lane": 0,
             "health": 17000, "max_health": 17000, "phase": 0,
             "x": 0.0, "y": 8000.0, "z": 0.0, "entity_id": 401},
            {"tick": 18 * MINUTE, "objective_type": "walker", "team_num": TEAM_B, "lane": 1,
             "health": 0, "max_health": 12000, "phase": 0,
             "x": 0.0, "y": 5000.0, "z": 0.0, "entity_id": 402},
        ],
    )


def _neutrals():
    """Two camps. Each is seen full, then on low health, then full again minutes
    later — which is what a cleared camp looks like in this build."""
    rows = []
    for entity, (x, y), clear_minute, respawn_minute, maximum in (
        (900, A_CAMP, 6, 12, 100),
        (901, B_CAMP, 8, 14, 200),
    ):
        rows += [
            {"tick": (clear_minute - 1) * MINUTE, "team_num": 4, "health": maximum,
             "max_health": maximum, "x": x, "y": y, "z": 0.0, "entity_id": entity},
            {"tick": clear_minute * MINUTE, "team_num": 4, "health": maximum // 10,
             "max_health": maximum, "x": x, "y": y, "z": 0.0, "entity_id": entity},
            {"tick": respawn_minute * MINUTE, "team_num": 4, "health": maximum,
             "max_health": maximum, "x": x, "y": y, "z": 0.0, "entity_id": entity},
        ]
    return frame("neutrals", rows)


def _mid_boss():
    return frame(
        "mid_boss",
        [
            # The kill row carries the boss's own neutral team; the `used` rows
            # after it are what name the taker.
            {"tick": 13 * MINUTE, "team_num": 4, "event": "killed"},
            {"tick": 13 * MINUTE + 600, "team_num": TEAM_A, "event": "used"},
        ],
    )


def _rift():
    return frame(
        "rift",
        [
            {"rift_num": 1, "announce_tick": 15 * MINUTE - 600, "active_tick": 15 * MINUTE,
             "capture_tick": 15 * MINUTE + 600, "expire_tick": None,
             "winning_team": TEAM_B, "lane": 1, "x": 0.0, "y": 0.0, "z": 0.0},
            {"rift_num": 2, "announce_tick": 19 * MINUTE - 600, "active_tick": 19 * MINUTE,
             "capture_tick": None, "expire_tick": 19 * MINUTE + 600,
             "winning_team": None, "lane": 1, "x": 0.0, "y": 0.0, "z": 0.0},
        ],
    )


def _urn():
    return frame(
        "urn",
        [
            {"tick": 16 * MINUTE, "event": "picked_up", "hero_id": 1, "team_num": 0,
             "x": 0.0, "y": 0.0, "z": 0.0},
            {"tick": HERO_1_DEATH, "event": "dropped", "hero_id": 1, "team_num": 0,
             "x": 0.0, "y": 0.0, "z": 0.0},
            # The urn resetting home re-emits the last carrier one tick later.
            # That is not somebody reaching for it.
            {"tick": 17 * MINUTE, "event": "returned", "hero_id": 1, "team_num": 0,
             "x": 0.0, "y": 0.0, "z": 0.0},
            {"tick": 17 * MINUTE + 1, "event": "picked_up", "hero_id": 1, "team_num": 0,
             "x": 0.0, "y": 0.0, "z": 0.0},
        ],
    )


@pytest.fixture
def ledger_match() -> MatchData:
    return MatchData(
        map_name="dl_midtown",
        match_id=1,
        tick_rate=TICK_RATE,
        total_ticks=END,
        game_over_tick=END,
        winning_team_num=TEAM_A,
        players=_players(),
        kills=_kills(),
        objectives=_objectives(),
        player_samples=_player_samples(),
        neutrals=_neutrals(),
        mid_boss=_mid_boss(),
        rift=_rift(),
        urn=_urn(),
    )


class TestStreams:
    def test_every_stream_reports_both_teams(self, ledger_match, ledger_names):
        ledger = analyze_advantage(ledger_match, ledger_names)
        assert ledger.streams
        for stream in ledger.streams:
            assert set(stream.by_team) == {TEAM_A, TEAM_B}, stream.key

    def test_net_worth_and_income_follow_the_fixture(self, ledger_match, ledger_names):
        ledger = analyze_advantage(ledger_match, ledger_names)
        streams = {s.key: s for s in ledger.streams}
        # Three heroes each, 1000 vs 600 souls per hero-minute, twenty minutes.
        assert streams["net_worth"].by_team == {TEAM_A: 60_000.0, TEAM_B: 36_000.0}
        assert streams["souls_per_min"].by_team == {TEAM_A: 3000.0, TEAM_B: 1800.0}
        assert streams["net_worth"].leader == TEAM_A
        assert streams["net_worth"].margin == 24_000.0

    def test_a_lower_is_better_stream_names_the_right_leader(self, ledger_match, ledger_names):
        ledger = analyze_advantage(ledger_match, ledger_names)
        streams = {s.key: s for s in ledger.streams}
        dead = streams["dead_time"]
        # B lost 120s to a respawn; A lost 118s. Fewer seconds dead leads.
        assert dead.higher_is_better is False
        assert dead.leader == TEAM_A

    def test_structures_are_credited_to_the_destroyer_not_the_owner(
        self, ledger_match, ledger_names
    ):
        ledger = analyze_advantage(ledger_match, ledger_names)
        streams = {s.key: s for s in ledger.streams}
        assert streams["structures"].by_team == {TEAM_A: 1.0, TEAM_B: 0.0}


class TestWindows:
    def test_windows_report_souls_gained_not_running_totals(self, ledger_match, ledger_names):
        ledger = analyze_advantage(ledger_match, ledger_names)
        assert [w.label for w in ledger.windows] == [
            "00:00–05:00", "05:00–10:00", "10:00–15:00", "15:00–20:00",
        ]
        for window in ledger.windows:
            assert window.gained_by_team == {TEAM_A: 15_000, TEAM_B: 9_000}
            assert window.leader == TEAM_A
            assert window.margin == 6_000

    def test_camp_clears_land_in_the_window_they_happened_in(self, ledger_match, ledger_names):
        ledger = analyze_advantage(ledger_match, ledger_names)
        by_label = {w.label: w for w in ledger.windows}
        assert by_label["05:00–10:00"].camps_by_team == {TEAM_A: 1, TEAM_B: 1}
        # Both were raids: each team cleared a camp in the other's half.
        assert by_label["05:00–10:00"].raids_by_team == {TEAM_A: 1, TEAM_B: 1}
        assert by_label["00:00–05:00"].camps_by_team == {TEAM_A: 0, TEAM_B: 0}


class TestCampClears:
    def test_a_camp_going_quiet_on_low_health_counts_as_cleared(self, ledger_match):
        clears = camp_clears(ledger_match)
        assert [c.tick for c in clears] == [6 * MINUTE, 8 * MINUTE]

    def test_credit_goes_to_the_nearest_living_hero(self, ledger_match):
        clears = {c.tick: c for c in camp_clears(ledger_match)}
        assert clears[6 * MINUTE].team_num == TEAM_B
        assert clears[8 * MINUTE].team_num == TEAM_A
        assert all(c.distance is not None and c.distance <= CAMP_CREDIT_RANGE for c in clears.values())

    def test_a_camp_in_the_other_half_is_a_raid(self, ledger_match):
        clears = {c.tick: c for c in camp_clears(ledger_match)}
        assert clears[6 * MINUTE].half_of == TEAM_A
        assert clears[6 * MINUTE].is_raid is True
        assert clears[8 * MINUTE].half_of == TEAM_B
        assert clears[8 * MINUTE].is_raid is True

    def test_a_camp_that_never_drops_low_is_not_a_clear(self, ledger_match):
        untouched = frame(
            "neutrals",
            [
                {"tick": 2 * MINUTE, "team_num": 4, "health": 100, "max_health": 100,
                 "x": 0.0, "y": -4000.0, "z": 0.0, "entity_id": 900},
                # Health rises because camps scale with match time, not because
                # anything respawned.
                {"tick": 15 * MINUTE, "team_num": 4, "health": 140, "max_health": 140,
                 "x": 0.0, "y": -4000.0, "z": 0.0, "entity_id": 900},
            ],
        )
        assert camp_clears(ledger_match.with_frames(neutrals=untouched)) == []


class TestTerritory:
    def test_time_in_the_enemy_half_is_counted_from_samples(self, ledger_match):
        rows = {t.team_num: t for t in territory(ledger_match, (TEAM_A, TEAM_B))}
        # One sampled minute each, spent on the other team's jungle camp.
        assert rows[TEAM_A].enemy_half_samples == 1
        assert rows[TEAM_B].enemy_half_samples == 1
        assert rows[TEAM_A].share == pytest.approx(1 / rows[TEAM_A].samples)

    def test_kills_are_located_by_where_the_victim_stood(self, ledger_match):
        rows = {t.team_num: t for t in territory(ledger_match, (TEAM_A, TEAM_B))}
        # Two B heroes died on their own ground; one A hero died on his.
        assert rows[TEAM_A].kills_in_enemy_half == 2
        assert rows[TEAM_A].deaths_in_own_half == 1
        assert rows[TEAM_B].kills_in_enemy_half == 1
        assert rows[TEAM_B].deaths_in_own_half == 2

    def test_no_base_positions_means_no_territory_claim(self, ledger_match):
        blind = ledger_match.with_frames(objectives=frame("objectives", []))
        assert territory(blind, (TEAM_A, TEAM_B)) == []


class TestDowntime:
    def test_respawn_is_read_from_the_gap_in_samples(self, ledger_match):
        rows = {d.hero_id: d for d in death_downtime(ledger_match)}
        # Died at 10:00, absent at 11:00, back at 12:00.
        assert rows[11].seconds == pytest.approx(120.0)
        assert rows[11].measured_deaths == 1

    def test_a_death_with_no_reappearance_is_not_charged(self, ledger_match):
        rows = {d.hero_id: d for d in death_downtime(ledger_match)}
        assert rows[12].deaths == 1
        assert rows[12].measured_deaths == 0
        assert rows[12].seconds == 0.0

    def test_souls_forgone_is_priced_at_the_players_own_rate(self, ledger_match):
        rows = {d.hero_id: d for d in death_downtime(ledger_match)}
        hero = rows[11]
        assert hero.souls_per_min is not None
        # Net worth over the time actually spent alive, times the time dead.
        assert hero.souls_forgone == pytest.approx(
            round(hero.souls_per_min * hero.seconds / 60.0), abs=1
        )

    def test_a_player_who_never_died_forgoes_nothing(self, ledger_match):
        rows = {d.hero_id: d for d in death_downtime(ledger_match)}
        assert rows[13].deaths == 0
        assert rows[13].souls_forgone == 0


class TestNeutralPrizes:
    def test_mid_boss_is_credited_to_the_team_that_used_the_buff(self, ledger_match):
        prizes = [p for p in neutral_prizes(ledger_match) if p.kind == "mid_boss"]
        assert len(prizes) == 1
        assert prizes[0].team_num == TEAM_A

    def test_a_captured_rift_names_a_winner_and_an_expired_one_does_not(self, ledger_match):
        rifts = [p for p in neutral_prizes(ledger_match) if p.kind == "rift"]
        assert [p.team_num for p in rifts] == [TEAM_B, None]


class TestUrn:
    def test_the_reset_artifact_is_not_counted_as_a_run(self, ledger_match):
        runs = urn_runs(ledger_match)
        assert len(runs) == 1
        assert runs[0].pickup_tick == 16 * MINUTE

    def test_a_carrier_killed_at_the_drop_is_a_denial(self, ledger_match):
        runs = urn_runs(ledger_match)
        assert runs[0].denied_by == TEAM_B
        assert runs[0].ended_by == "dropped"


class TestLevers:
    def test_levers_are_written_for_the_losing_team_by_default(self, ledger_match, ledger_names):
        ledger = analyze_advantage(ledger_match, ledger_names)
        assert ledger.focus_team == TEAM_B
        assert ledger.focus_reason == "the losing team"
        assert all(lever.team_num == TEAM_B for lever in ledger.levers)

    def test_a_requested_player_reorients_the_levers(self, ledger_match, ledger_names):
        ledger = analyze_advantage(ledger_match, ledger_names, focus_hero_ids=(1,))
        assert ledger.focus_team == TEAM_A
        assert ledger.focus_reason == "the requested player's team"

    def test_levers_are_ranked_with_the_priced_ones_first(self, ledger_match, ledger_names):
        ledger = analyze_advantage(ledger_match, ledger_names)
        assert ledger.levers
        stakes = [lever.souls_at_stake for lever in ledger.levers]
        priced = [s for s in stakes if s is not None]
        assert priced == sorted(priced, reverse=True)
        assert stakes[: len(priced)] == priced

    def test_a_lever_without_a_soul_value_still_ships(self, ledger_match, ledger_names):
        """A resource whose soul value the demo does not carry must not be
        dropped just because it cannot be priced — it ships with counts, below
        everything that could be priced."""
        greedy = frame(
            "mid_boss",
            [
                {"tick": 13 * MINUTE, "team_num": 4, "event": "killed"},
                {"tick": 13 * MINUTE + 600, "team_num": TEAM_A, "event": "used"},
                {"tick": 17 * MINUTE, "team_num": 4, "event": "killed"},
                {"tick": 17 * MINUTE + 600, "team_num": TEAM_A, "event": "used"},
            ],
        )
        ledger = analyze_advantage(ledger_match.with_frames(mid_boss=greedy), ledger_names)
        prizes = [lever for lever in ledger.levers if lever.key == "map_prizes"]
        assert prizes and prizes[0].souls_at_stake is None
        assert ledger.levers[-1].key == "map_prizes"

    def test_every_lever_carries_evidence_and_an_action(self, ledger_match, ledger_names):
        ledger = analyze_advantage(ledger_match, ledger_names)
        for lever in ledger.levers:
            assert lever.evidence
            assert lever.action
            assert lever.confidence in ("high", "medium", "low")


class TestDegradedInput:
    def test_a_match_without_two_teams_says_so_instead_of_guessing(self, empty_match):
        ledger = analyze_advantage(empty_match, Names(heroes={}, teams={}))
        assert ledger.streams == ()
        assert ledger.levers == ()
        assert any("two playing teams" in caveat for caveat in ledger.caveats)

    def test_the_shared_fixture_still_produces_a_ledger(self, match, names):
        ledger = analyze_advantage(match, names)
        assert ledger.teams == (TEAM_A, TEAM_B)
        # No base positions and no neutral data there — the ledger degrades to
        # the streams it can measure rather than failing.
        assert ledger.territory == ()
        assert ledger.camp_clears == ()
        assert {s.key for s in ledger.streams} >= {"net_worth", "kills"}

    def test_missing_neutral_frame_is_unknown_not_two_zeroes(self, match, names):
        ledger = analyze_advantage(match, names)
        keys = {stream.key for stream in ledger.streams}
        assert keys.isdisjoint({"camps", "raids", "camps_conceded"})
        assert any("unknown, not zero" in caveat for caveat in ledger.caveats)


class TestRendering:
    def test_markdown_contains_the_complete_advantage_ledger(self, ledger_match, ledger_names):
        report = render_report(ledger_match, ledger_names)
        section = report.split("## Win conditions — the advantage ledger", 1)[1]
        section = section.split("## Teamfights", 1)[0]

        for heading in (
            "### Resource streams, whole match",
            "### Where the gap opened, five minutes at a time",
            "### Map control — whose ground the match was played on",
            "### Death downtime — the resource nobody counts",
            "### Levers — ranked by the souls at stake",
        ):
            assert heading in section
        assert "| Net worth (final) | 60,000 | 36,000 |" in section
        assert "Levers below are written for **Archmother** (the losing team)." in section

    def test_player_focus_reorients_the_rendered_levers(self, ledger_match, ledger_names):
        report = render_report(ledger_match, ledger_names, focus_hero_ids=(1,))
        assert (
            "Levers below are written for **Hidden King** "
            "(the requested player's team)."
        ) in report

    def test_json_carries_comparisons_and_derived_properties(self, ledger_match, ledger_names):
        advantage = json.loads(render_json(ledger_match, ledger_names))["advantage"]
        assert set(advantage) == {
            "focus_team",
            "focus_reason",
            "streams",
            "windows",
            "territory",
            "downtime",
            "camp_clears",
            "neutral_prizes",
            "urn_runs",
            "levers",
            "caveats",
        }
        net_worth = next(stream for stream in advantage["streams"] if stream["key"] == "net_worth")
        assert net_worth["leader"] == TEAM_A
        assert net_worth["margin"] == 24_000.0
        assert all("share" in row for row in advantage["territory"])
