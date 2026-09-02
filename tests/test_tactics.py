from __future__ import annotations

import polars as pl

from dataclasses import replace

from deadlock_coach.match import MatchData, empty
from deadlock_coach.tactics import (
    CONVERSION_WINDOW_SECONDS,
    UNSUPPORTABLE_SECONDS,
    analyze_fights,
    biggest_swing,
    economy_curve,
    kill_contexts,
    phase_at,
    phase_stats,
    phases,
    player_reports,
)

from .conftest import MINUTE, TEAM_A, TEAM_B, TICK_RATE, frame


class TestPhases:
    def test_laning_ends_at_the_first_rotation(self, match):
        """Heroes 1 and 2 start in different lanes, so the 09:30 kill on hero 2
        by hero 12 (a different lane again) is the first cross-lane kill."""
        laning, mid, late = phases(match)
        assert laning.name == "laning"
        assert laning.end_tick == 9 * MINUTE + 30 * TICK_RATE
        assert "cross-lane" in laning.trigger
        assert mid.end_tick == 15 * MINUTE  # first Walker
        assert late.end_tick == match.end_tick

    def test_same_lane_kills_do_not_end_laning(self):
        """A 1v1 inside one lane is still laning."""
        md = MatchData(
            tick_rate=TICK_RATE,
            total_ticks=30 * MINUTE,
            players=frame("players", [
                {"hero_id": 1, "team_num": TEAM_A, "player_name": "a", "start_lane": 1},
                {"hero_id": 11, "team_num": TEAM_B, "player_name": "b", "start_lane": 1},
            ]),
            kills=frame("kills", [
                {"tick": 2 * MINUTE, "victim_hero_id": 11, "attacker_hero_id": 1,
                 "assister_hero_ids": []},
            ]),
        )
        laning = phases(md)[0]
        assert laning.end_tick == 10 * 60 * TICK_RATE
        assert "no rotation detected" in laning.trigger

    def test_falls_back_to_the_clock_without_any_data(self, empty_match):
        laning = phases(empty_match)[0]
        assert "no rotation detected" in laning.trigger

    def test_walker_boundary_ignores_neutral_objectives(self, match):
        """A Mid Boss dying is not a Walker falling."""
        md = match.with_frames(objectives=frame("objectives", [
            {"tick": 12 * MINUTE, "objective_type": "mid_boss", "team_num": 4,
             "lane": 16777215, "health": 0, "max_health": 19435, "entity_id": 9},
        ]))
        mid = phases(md)[1]
        assert "no Walker data" in mid.trigger

    def test_phase_at_maps_ticks_to_names(self, match):
        p = phases(match)
        assert phase_at(p, 1 * MINUTE) == "laning"
        assert phase_at(p, 12 * MINUTE) == "mid game"
        assert phase_at(p, 23 * MINUTE) == "late game"

    def test_tick_past_the_end_still_resolves(self, match):
        assert phase_at(phases(match), 99 * MINUTE) == "late game"

    def test_empty_match_produces_a_phase_list(self, empty_match):
        assert [p.name for p in phases(empty_match)] == ["laning"]


class TestFights:
    def test_scores_kills_inside_the_window(self, match, names):
        first = analyze_fights(match, names)[0]
        assert first.kills_by_team == {TEAM_A: 2, TEAM_B: 1}
        assert first.winner == TEAM_A
        assert first.verdict == "won 2-1"

    def test_reports_engagement_size_per_team(self, match, names):
        first = analyze_fights(match, names)[0]
        assert first.participants_by_team[TEAM_A] == [1, 2, 3]
        assert first.participants_by_team[TEAM_B] == [11, 12]
        assert first.engagement == "3v2"

    def test_second_fight_flips_the_winner(self, match, names):
        second = analyze_fights(match, names)[1]
        assert second.winner == TEAM_B
        assert second.kills_by_team[TEAM_B] == 2

    def test_conversion_credits_an_objective_taken_soon_after(self, names):
        """A Walker falling 20s after the fight is a conversion."""
        md = _fight_and_objective(offset_seconds=20)
        fight = analyze_fights(md, names)[0]
        assert fight.converted_into == "Walker"
        assert fight.conversion_assessment.status == "converted"

    def test_conversion_ignores_objectives_outside_the_window(self, names):
        md = _fight_and_objective(offset_seconds=CONVERSION_WINDOW_SECONDS + 10)
        assert analyze_fights(md, names)[0].converted_into is None

    def test_conversion_ignores_the_winners_own_objective_falling(self, names):
        """If the objective that dies belongs to the fight's *winner*, they did
        not convert — they got counter-pushed."""
        md = _fight_and_objective(offset_seconds=10, objective_owner=TEAM_A)
        assert analyze_fights(md, names)[0].converted_into is None

    def test_fight_with_no_kills_is_a_standoff(self, names):
        md = _fight_and_objective(offset_seconds=10, kills=[])
        fight = analyze_fights(md, names)[0]
        assert fight.winner is None
        assert "no kills" in fight.verdict

    def test_even_trade_has_no_winner(self, names):
        md = _fight_and_objective(
            offset_seconds=10,
            kills=[
                {"tick": 1000, "victim_hero_id": 11, "attacker_hero_id": 1,
                 "assister_hero_ids": []},
                {"tick": 1010, "victim_hero_id": 1, "attacker_hero_id": 11,
                 "assister_hero_ids": []},
            ],
        )
        fight = analyze_fights(md, names)[0]
        assert fight.winner is None
        assert "even trade" in fight.verdict

    def test_environment_death_credits_the_opposing_team(self, names):
        """No attacker, but the victim's opponents still gain the man advantage."""
        md = _fight_and_objective(
            offset_seconds=10,
            kills=[{"tick": 1000, "victim_hero_id": 11, "attacker_hero_id": 11,
                    "assister_hero_ids": []}],
        )
        assert analyze_fights(md, names)[0].winner == TEAM_A

    def test_no_teamfights_frame_yields_nothing(self, empty_match, names):
        assert analyze_fights(empty_match, names) == []

    def test_adjacent_fragments_with_shared_players_are_one_fight(self, match, names):
        rows = list(match.teamfights.iter_rows(named=True))
        rows.insert(
            1,
            {
                "fight_id": 99,
                "start_tick": rows[0]["end_tick"] + TICK_RATE,
                "end_tick": rows[0]["end_tick"] + 2 * TICK_RATE,
                "duration_seconds": 1.0,
                "participants": [1, 2, 11, 12],
                "num_participants": 4,
                "hero_damage": 500,
                "kills": 0,
            },
        )
        combined = match.with_frames(teamfights=frame("teamfights", rows))
        fights = analyze_fights(combined, names)
        assert len(fights) == 2
        assert fights[0].end_tick == rows[1]["end_tick"]
        assert fights[0].hero_damage == 4_700

    def test_simultaneous_fights_do_not_share_the_same_kills(self, match, names):
        rows = list(match.teamfights.iter_rows(named=True))
        original = rows[0]
        rows.append(
            {
                "fight_id": 99,
                "start_tick": original["start_tick"],
                "end_tick": original["end_tick"],
                "duration_seconds": original["duration_seconds"],
                "participants": [1, 13],
                "num_participants": 2,
                "hero_damage": 100,
                "kills": 0,
            }
        )
        split = match.with_frames(teamfights=frame("teamfights", rows))
        other_lane = next(fight for fight in analyze_fights(split, names) if fight.fight_id == 99)
        assert other_lane.winner is None
        assert other_lane.kills_by_team == {TEAM_A: 0, TEAM_B: 0}

    def test_wave_and_nearby_winner_make_a_structure_pushable(self, names):
        md = _conversion_match(
            allied_positions=[(900.0, 0.0), (1_000.0, 50.0), (1_100.0, -50.0)],
            enemy_positions=[(1_200.0, 0.0)],
            winner_position=(1_500.0, 0.0),
        )
        read = analyze_fights(md, names)[0].conversion_assessment
        assert read.status == "push_now"
        assert read.target == "yellow-lane Walker"
        assert read.allied_front_troopers == 3
        assert read.enemy_contesting_troopers == 1

    def test_distant_wave_is_not_scored_as_a_missed_conversion(self, names):
        md = _conversion_match(
            allied_positions=[(8_000.0, 0.0), (8_100.0, 0.0), (8_200.0, 0.0)],
            enemy_positions=[],
            winner_position=(1_000.0, 0.0),
        )
        read = analyze_fights(md, names)[0].conversion_assessment
        assert read.status == "no_structure_window"
        assert read.target is None
        assert "no allied lane wave" in read.reason

    def test_wave_that_needs_a_rotation_is_setup_not_push_now(self, names):
        md = _conversion_match(
            allied_positions=[(3_500.0, 0.0), (3_600.0, 0.0), (3_700.0, 0.0)],
            enemy_positions=[],
            winner_position=(6_000.0, 0.0),
        )
        read = analyze_fights(md, names)[0].conversion_assessment
        assert read.status == "setup_required"
        assert read.nearest_winner_distance == 6_000.0

    def test_enemy_wave_can_block_an_apparently_close_push(self, names):
        md = _conversion_match(
            allied_positions=[(1_000.0, 0.0), (1_050.0, 0.0), (1_100.0, 0.0)],
            enemy_positions=[
                (1_000.0, 100.0),
                (1_050.0, 100.0),
                (1_100.0, 100.0),
                (1_150.0, 100.0),
            ],
            winner_position=(1_500.0, 0.0),
        )
        assert analyze_fights(md, names)[0].conversion_assessment.status == "no_structure_window"

    def test_missing_troopers_stays_unknown(self, names):
        md = _conversion_match(
            allied_positions=[], enemy_positions=[], winner_position=(1_500.0, 0.0)
        ).with_frames(trooper_samples=frame("trooper_samples", []))
        read = analyze_fights(md, names)[0].conversion_assessment
        assert read.status == "unknown"
        assert "unavailable" in read.reason


def _fight_and_objective(
    *, offset_seconds: float, objective_owner: int = TEAM_B, kills: list[dict] | None = None
) -> MatchData:
    end_tick = 1200
    if kills is None:
        kills = [
            {"tick": 1000, "victim_hero_id": 11, "attacker_hero_id": 1, "assister_hero_ids": []}
        ]
    return MatchData(
        tick_rate=TICK_RATE,
        total_ticks=10 * MINUTE,
        players=frame("players", [
            {"hero_id": 1, "team_num": TEAM_A, "player_name": "a"},
            {"hero_id": 11, "team_num": TEAM_B, "player_name": "b"},
        ]),
        kills=frame("kills", kills),
        objectives=frame("objectives", [
            {"tick": end_tick + int(offset_seconds * TICK_RATE), "objective_type": "walker",
             "team_num": objective_owner, "lane": 1, "health": 0, "max_health": 100,
             "entity_id": 7},
        ]),
        teamfights=frame("teamfights", [
            {"fight_id": 1, "start_tick": 900, "end_tick": end_tick, "duration_seconds": 5.0,
             "participants": [1, 11], "num_participants": 2, "hero_damage": 1000, "kills": 1},
        ]),
    )


def _conversion_match(
    *,
    allied_positions: list[tuple[float, float]],
    enemy_positions: list[tuple[float, float]],
    winner_position: tuple[float, float],
) -> MatchData:
    end_tick = 1_200
    troopers = []
    entity = 1
    for team, positions in ((TEAM_A, allied_positions), (TEAM_B, enemy_positions)):
        for x, y in positions:
            troopers.append(
                {
                    "tick": end_tick,
                    "trooper_type": "trooper",
                    "team_num": team,
                    "lane": 1,
                    "health": 100,
                    "max_health": 100,
                    "x": x,
                    "y": y,
                    "z": 0.0,
                    "entity_id": entity,
                }
            )
            entity += 1
    return MatchData(
        tick_rate=TICK_RATE,
        total_ticks=10 * MINUTE,
        players=frame(
            "players",
            [
                {"hero_id": 1, "team_num": TEAM_A, "player_name": "a", "start_lane": 1},
                {"hero_id": 11, "team_num": TEAM_B, "player_name": "b", "start_lane": 1},
            ],
        ),
        kills=frame(
            "kills",
            [
                {
                    "tick": 1_000,
                    "victim_hero_id": 11,
                    "attacker_hero_id": 1,
                    "assister_hero_ids": [],
                }
            ],
        ),
        objectives=frame(
            "objectives",
            [
                {
                    "tick": 1,
                    "objective_type": "walker",
                    "team_num": TEAM_B,
                    "lane": 1,
                    "health": 12_000,
                    "max_health": 12_000,
                    "x": 0.0,
                    "y": 0.0,
                    "entity_id": 500,
                }
            ],
        ),
        teamfights=frame(
            "teamfights",
            [
                {
                    "fight_id": 1,
                    "start_tick": 900,
                    "end_tick": end_tick,
                    "duration_seconds": 5.0,
                    "participants": [1, 11],
                    "num_participants": 2,
                    "hero_damage": 1_000,
                    "kills": 1,
                }
            ],
        ),
        player_samples=frame(
            "player_samples",
            [
                {
                    "tick": end_tick,
                    "hero_id": 1,
                    "x": winner_position[0],
                    "y": winner_position[1],
                    "is_alive": True,
                    "health": 800,
                    "max_health": 800,
                }
            ],
        ),
        trooper_samples=frame("trooper_samples", troopers),
    )


class TestKillContexts:
    def test_detects_an_isolated_outnumbered_pickoff(self, match):
        first = [c for c in kill_contexts(match) if c.tick == 5 * MINUTE][0]
        assert first.attackers == 3
        assert first.defenders_nearby == 0
        assert first.isolated is True
        assert first.outnumbered is True

    def test_even_fight_next_to_an_ally_is_neither(self, match):
        even = [c for c in kill_contexts(match) if c.tick == 7 * MINUTE][0]
        assert even.attackers == 1
        assert even.defenders_nearby == 1
        assert even.isolated is False
        assert even.outnumbered is False

    def test_unknown_when_no_positional_sample_exists(self, match):
        """Ticks we never sampled must report None, never a fabricated zero."""
        unsampled = [c for c in kill_contexts(match) if c.tick == 12 * MINUTE][0]
        assert unsampled.defenders_nearby is None
        assert unsampled.isolated is None
        assert unsampled.outnumbered is None

    def test_tags_each_kill_with_its_phase(self, match):
        contexts = {c.tick: c.phase for c in kill_contexts(match)}
        assert contexts[5 * MINUTE] == "laning"
        assert contexts[20 * MINUTE] == "late game"


class TestPlayerReports:
    def test_kda_comes_from_the_kill_feed(self, match):
        hero_1 = [r for r in player_reports(match) if r.hero_id == 1][0]
        assert (hero_1.kills, hero_1.deaths, hero_1.assists) == (3, 1, 0)

    def test_self_kill_counts_as_a_death_not_a_kill(self, match):
        hero_3 = [r for r in player_reports(match) if r.hero_id == 3][0]
        assert hero_3.kills == 0
        assert hero_3.deaths == 2  # killed once in the 12:00 fight, once by the map

    def test_kill_participation_is_a_share_of_team_kills(self, match):
        # Team A is credited with 3 kills (the three deaths of team B heroes).
        reports = {r.hero_id: r for r in player_reports(match)}
        assert reports[2].kill_participation == 1.0  # assisted on all three
        assert reports[3].kill_participation == 1 / 3  # assisted on one

    def test_deaths_split_by_phase(self, match):
        hero_2 = [r for r in player_reports(match) if r.hero_id == 2][0]
        assert hero_2.deaths_by_phase == {"laning": 2, "late game": 1}

    def test_net_worth_includes_spent_souls(self, match):
        hero_1 = [r for r in player_reports(match) if r.hero_id == 1][0]
        assert hero_1.final_net_worth == 9000 + 18000  # souls + spent at the last sample

    def test_isolated_deaths_unknown_without_samples(self, match):
        md = match.with_frames(
            player_samples=pl.DataFrame(schema=match.player_samples.schema)
        )
        assert all(r.isolated_deaths is None for r in player_reports(md))

    def test_isolation_note_does_not_fire_on_a_single_sampled_death(self, match):
        hero_1 = [r for r in player_reports(match) if r.hero_id == 1][0]
        assert hero_1.deaths == 1
        assert not any("no living teammate" in n for n in hero_1.notes)

    def test_isolation_note_fires_on_repeated_isolated_deaths(self):
        """Four deaths, every one of them alone — the pattern a coach flags."""
        ticks = [600, 1200, 1800, 2400]
        far = 8000.0
        samples = []
        for tick in ticks:
            samples += [
                {"tick": tick, "hero_id": 1, "x": 0.0, "y": 0.0, "is_alive": True},
                {"tick": tick, "hero_id": 2, "x": far, "y": far, "is_alive": True},
                {"tick": tick, "hero_id": 11, "x": 10.0, "y": 0.0, "is_alive": True},
            ]
        md = MatchData(
            tick_rate=TICK_RATE,
            total_ticks=10 * MINUTE,
            players=frame("players", [
                {"hero_id": 1, "team_num": TEAM_A, "player_name": "a"},
                {"hero_id": 2, "team_num": TEAM_A, "player_name": "a2"},
                {"hero_id": 11, "team_num": TEAM_B, "player_name": "b"},
            ]),
            kills=frame("kills", [
                {"tick": t, "victim_hero_id": 1, "attacker_hero_id": 11,
                 "assister_hero_ids": []} for t in ticks
            ]),
            player_samples=frame("player_samples", samples),
        )
        hero_1 = [r for r in player_reports(md) if r.hero_id == 1][0]
        assert hero_1.isolated_deaths == 4
        assert any("no living teammate" in n for n in hero_1.notes)
        assert any("laning phase" in n for n in hero_1.notes)

    def test_low_participation_is_called_out(self, match):
        hero_3 = [r for r in player_reports(match) if r.hero_id == 3][0]
        assert any("Kill participation" in n for n in hero_3.notes)

    def test_empty_match_produces_no_reports(self, empty_match):
        assert player_reports(empty_match) == []


class TestEconomy:
    def test_sums_net_worth_per_team_per_sample(self, match):
        curve = economy_curve(match)
        assert [s.tick for s in curve] == [5 * MINUTE, 7 * MINUTE, 20 * MINUTE]
        first = curve[0]
        # Three heroes per team, each 3000 souls + 6000 spent.
        assert first.net_worth_by_team[TEAM_A] == 3 * 9000

    def test_reports_no_lead_when_teams_are_level(self, match):
        assert economy_curve(match)[0].lead == 0
        assert economy_curve(match)[0].lead_team is None

    def test_biggest_swing_finds_the_steepest_step(self):
        md = _economy_match(
            [(0, {1: 100, 11: 100}), (600, {1: 500, 11: 100}), (1200, {1: 520, 11: 120})]
        )
        swing = biggest_swing(economy_curve(md))
        assert swing is not None
        assert (swing[0].tick, swing[1].tick) == (0, 600)

    def test_a_lead_changing_hands_registers_as_one_swing(self):
        """Signed against a fixed team, so +400 -> -400 reads as 800, not 0."""
        md = _economy_match([(0, {1: 900, 11: 500}), (600, {1: 500, 11: 900})])
        swing = biggest_swing(economy_curve(md))
        assert swing is not None
        assert swing[1].lead_team == TEAM_B

    def test_dead_players_net_worth_carries_forward(self):
        """A dead player emits no row in `player_ticks`. Their souls must not
        vanish from the team total and reappear when they respawn — that showed
        up as 150k phantom swings on match 12345678."""
        md = MatchData(
            tick_rate=TICK_RATE,
            players=frame("players", [
                {"hero_id": 1, "team_num": TEAM_A, "player_name": "a"},
                {"hero_id": 2, "team_num": TEAM_A, "player_name": "a2"},
                {"hero_id": 11, "team_num": TEAM_B, "player_name": "b"},
            ]),
            player_samples=frame("player_samples", [
                {"tick": 0, "hero_id": 1, "souls": 1000, "spent_souls": 0},
                {"tick": 0, "hero_id": 2, "souls": 1000, "spent_souls": 0},
                {"tick": 0, "hero_id": 11, "souls": 500, "spent_souls": 0},
                # Hero 2 is dead at tick 600 and reports nothing at all.
                {"tick": 600, "hero_id": 1, "souls": 1200, "spent_souls": 0},
                {"tick": 600, "hero_id": 11, "souls": 700, "spent_souls": 0},
            ]),
        )
        curve = economy_curve(md)
        assert curve[0].net_worth_by_team[TEAM_A] == 2000
        # 1200 (alive, updated) + 1000 (dead, carried forward) — not 1200.
        assert curve[1].net_worth_by_team[TEAM_A] == 2200

    def test_net_worth_is_monotonic_when_inputs_are(self):
        """Sanity guard: with only increasing per-hero values, no team total may
        ever drop. This is the invariant the forward-fill exists to protect."""
        rows = []
        for i, tick in enumerate([0, 600, 1200, 1800]):
            rows.append({"tick": tick, "hero_id": 1, "souls": 1000 * (i + 1), "spent_souls": 0})
            # Hero 2 only reports on the first and last tick.
            if tick in (0, 1800):
                rows.append({"tick": tick, "hero_id": 2, "souls": 500 * (i + 1), "spent_souls": 0})
            rows.append({"tick": tick, "hero_id": 11, "souls": 100, "spent_souls": 0})
        md = MatchData(
            tick_rate=TICK_RATE,
            players=frame("players", [
                {"hero_id": 1, "team_num": TEAM_A, "player_name": "a"},
                {"hero_id": 2, "team_num": TEAM_A, "player_name": "a2"},
                {"hero_id": 11, "team_num": TEAM_B, "player_name": "b"},
            ]),
            player_samples=frame("player_samples", rows),
        )
        totals = [s.net_worth_by_team[TEAM_A] for s in economy_curve(md)]
        assert totals == sorted(totals), totals

    def test_no_samples_yields_no_curve(self, empty_match):
        assert economy_curve(empty_match) == []


def _economy_match(points: list[tuple[int, dict[int, int]]]) -> MatchData:
    rows = []
    for tick, souls_by_hero in points:
        for hero, souls in souls_by_hero.items():
            rows.append({"tick": tick, "hero_id": hero, "souls": souls, "spent_souls": 0})
    return MatchData(
        tick_rate=TICK_RATE,
        players=frame("players", [
            {"hero_id": 1, "team_num": TEAM_A, "player_name": "a"},
            {"hero_id": 11, "team_num": TEAM_B, "player_name": "b"},
        ]),
        player_samples=frame("player_samples", rows),
    )


class TestPhaseStats:
    def test_covers_every_player_in_every_phase(self, match):
        stats = phase_stats(match)
        assert len({s.phase for s in stats}) == 3
        assert len(stats) == 3 * 6

    def test_figures_are_deltas_not_running_totals(self):
        """`hero_damage` is cumulative in the demo. A phase must report what was
        added during it, otherwise every later phase silently includes the whole
        game up to that point."""
        md = _phase_stat_match([
            (0, 0), (5 * MINUTE, 1_000),          # laning ends 10:00 (fallback)
            (12 * MINUTE, 4_000),                  # mid game
            (26 * MINUTE, 10_000),                 # late game (mid ends 25:00)
        ])
        stats = {s.phase: s for s in phase_stats(md) if s.hero_id == 1}
        assert stats["laning"].hero_damage == 1_000     # 1,000 - 0
        assert stats["mid game"].hero_damage == 3_000   # 4,000 - 1,000
        assert stats["late game"].hero_damage == 6_000  # 10,000 - 4,000
        # The three deltas must reconstruct the match total, with no overlap.
        assert sum(s.hero_damage for s in stats.values()) == 10_000

    def test_kills_are_attributed_to_the_phase_they_happened_in(self, match):
        stats = {(s.hero_id, s.phase): s for s in phase_stats(match)}
        # Hero 2 dies at 07:00 and 09:30 (laning) and 20:00 (late game).
        assert stats[(2, "laning")].deaths == 2
        assert stats[(2, "late game")].deaths == 1
        assert stats[(2, "mid game")].deaths == 0

    def test_damage_share_is_relative_to_the_players_own_team(self):
        md = MatchData(
            tick_rate=TICK_RATE,
            total_ticks=20 * MINUTE,
            players=frame("players", [
                {"hero_id": 1, "team_num": TEAM_A, "player_name": "a"},
                {"hero_id": 2, "team_num": TEAM_A, "player_name": "a2"},
                {"hero_id": 11, "team_num": TEAM_B, "player_name": "b"},
            ]),
            player_samples=frame("player_samples", [
                {"tick": 0, "hero_id": 1, "hero_damage": 0},
                {"tick": 0, "hero_id": 2, "hero_damage": 0},
                {"tick": 0, "hero_id": 11, "hero_damage": 0},
                # Inside the laning phase, which ends at 10:00 here.
                {"tick": 9 * MINUTE, "hero_id": 1, "hero_damage": 300},
                {"tick": 9 * MINUTE, "hero_id": 2, "hero_damage": 100},
                {"tick": 9 * MINUTE, "hero_id": 11, "hero_damage": 9999},
            ]),
        )
        laning = {s.hero_id: s for s in phase_stats(md) if s.phase == "laning"}
        assert laning[1].damage_share == 0.75  # 300 of team A's 400
        assert laning[11].damage_share == 1.0

    def test_missing_samples_report_unknown_not_zero(self, empty_match):
        md = empty_match.with_frames(players=frame("players", [
            {"hero_id": 1, "team_num": TEAM_A, "player_name": "a"},
            {"hero_id": 11, "team_num": TEAM_B, "player_name": "b"},
        ]))
        stats = phase_stats(md)
        assert stats and all(s.hero_damage is None for s in stats)
        assert all(s.net_worth_gained is None for s in stats)


def _phase_stat_match(points: list[tuple[int, int]]) -> MatchData:
    """One hero on each team, with a cumulative hero_damage curve to difference."""
    rows = []
    for tick, dealt in points:
        rows.append({"tick": tick, "hero_id": 1, "hero_damage": dealt})
        rows.append({"tick": tick, "hero_id": 11, "hero_damage": 0})
    return MatchData(
        tick_rate=TICK_RATE,
        total_ticks=40 * MINUTE,
        game_over_tick=40 * MINUTE,
        players=frame("players", [
            {"hero_id": 1, "team_num": TEAM_A, "player_name": "a"},
            {"hero_id": 11, "team_num": TEAM_B, "player_name": "b"},
        ]),
        player_samples=frame("player_samples", rows),
    )


class TestSupportReach:
    """Distance to help, in the units that decide whether "alone" was a mistake."""

    def test_a_death_next_to_a_teammate_records_a_short_reach(self, match):
        contexts = {c.tick: c for c in kill_contexts(match)}
        with_reach = [c for c in contexts.values() if c.support_seconds is not None]
        assert with_reach, "positional samples exist, so reach must be computed"
        assert all(c.nearest_ally_m is not None for c in with_reach)

    def test_reach_is_reported_in_metres_not_hammer_units(self, match):
        # A Deadlock lane is tens of metres across, not thousands. This catches
        # the unit slipping back to raw replay coordinates.
        gaps = [c.nearest_ally_m for c in kill_contexts(match) if c.nearest_ally_m is not None]
        assert gaps and max(gaps) < 2_000

    def test_an_unreachable_teammate_is_called_out_as_such(self, match):
        stranded = [
            c
            for c in kill_contexts(match)
            if c.support_seconds is not None and c.support_seconds >= UNSUPPORTABLE_SECONDS
        ]
        for context in stranded:
            if context.defenders_nearby:
                continue
            assert "nobody could have helped" in (context.support_read or "")

    def test_no_samples_means_no_claim_about_support(self, match):
        blind = replace(match, player_samples=empty("player_samples"))
        for context in kill_contexts(blind):
            assert context.support_seconds is None
            assert context.support_read is None
