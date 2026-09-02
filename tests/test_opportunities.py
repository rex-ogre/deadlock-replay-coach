from __future__ import annotations

from dataclasses import replace

from deadlock_coach.match import MatchData
from deadlock_coach.opportunities import (
    MID_BOSS_AVAILABLE_SECONDS,
    MID_BOSS_UNGATED_FROM_MATCH_ID,
    analyze_opportunities,
    mid_boss_available_from,
    observation_frames,
)

from .conftest import MINUTE, TEAM_A, TEAM_B, TICK_RATE, frame


def _players():
    return frame(
        "players",
        [
            {"hero_id": 1, "team_num": TEAM_A, "start_lane": 1, "player_name": "observer"},
            {"hero_id": 2, "team_num": TEAM_A, "start_lane": 1, "player_name": "ally"},
            {"hero_id": 11, "team_num": TEAM_B, "start_lane": 1, "player_name": "target"},
            {"hero_id": 12, "team_num": TEAM_B, "start_lane": 4, "player_name": "hidden"},
        ],
    )


def _kill_window_match(hidden_x: float = 10_000.0, *, with_kill: bool = False) -> MatchData:
    ticks = [100, 160]
    rows = []
    for tick in ticks:
        rows.extend(
            [
                {"tick": tick, "hero_id": 1, "x": 0.0, "y": 0.0,
                 "health": 900, "max_health": 1000, "is_alive": True},
                {"tick": tick, "hero_id": 2, "x": 100.0, "y": 0.0,
                 "health": 800, "max_health": 1000, "is_alive": True},
                {"tick": tick, "hero_id": 11, "x": 800.0, "y": 0.0,
                 "health": 200, "max_health": 1000, "is_alive": True},
                {"tick": tick, "hero_id": 12, "x": hidden_x, "y": hidden_x,
                 "health": 1000, "max_health": 1000, "is_alive": True},
            ]
        )
    kills = []
    if with_kill:
        kills.append(
            {"tick": 200, "victim_hero_id": 11, "attacker_hero_id": 1,
             "assister_hero_ids": []}
        )
    return MatchData(
        tick_rate=TICK_RATE,
        total_ticks=600,
        players=_players(),
        player_samples=frame("player_samples", rows),
        damage=frame(
            "damage",
            [{"tick": 95, "damage": 100, "attacker_hero_id": 1, "victim_hero_id": 11}],
        ),
        kills=frame("kills", kills),
    )


class TestInformationFirewall:
    def test_last_seen_state_freezes_instead_of_following_hidden_live_position(self):
        md = _kill_window_match()
        extra = [
            {"tick": 400, "hero_id": 1, "x": 0.0, "y": 0.0,
             "health": 900, "max_health": 1000, "is_alive": True},
            {"tick": 400, "hero_id": 2, "x": 0.0, "y": 0.0,
             "health": 800, "max_health": 1000, "is_alive": True},
            # The replay knows the target moved. The observer timeline must not.
            {"tick": 400, "hero_id": 11, "x": 9000.0, "y": 9000.0,
             "health": 999, "max_health": 1000, "is_alive": True},
            {"tick": 400, "hero_id": 12, "x": 10000.0, "y": 10000.0,
             "health": 1000, "max_health": 1000, "is_alive": True},
        ]
        samples = frame(
            "player_samples",
            list(md.player_samples.iter_rows(named=True)) + extra,
        )
        frames = observation_frames(md.with_frames(player_samples=samples), 1)
        at_400 = next(f for f in frames if f.tick == 400)
        target = next(e for e in at_400.enemies if e.hero_id == 11)
        assert (target.x, target.y, target.health) == (800.0, 0.0, 200)
        assert target.source.startswith("last seen")

    def test_moving_a_still_hidden_enemy_does_not_change_kill_recommendations(self):
        left = analyze_opportunities(_kill_window_match(hidden_x=10_000.0)).kill_windows
        right = analyze_opportunities(_kill_window_match(hidden_x=-10_000.0)).kill_windows
        assert left == right
        mine = [window for window in left if window.observer_hero_id == 1]
        assert len(mine) == 1
        assert mine[0].target_hero_id == 11

    def test_pressure_that_converted_to_a_real_kill_is_not_called_additional(self):
        analysis = analyze_opportunities(_kill_window_match(with_kill=True))
        assert analysis.kill_windows == ()


def _jungle_match(camp_health: int) -> MatchData:
    return MatchData(
        tick_rate=TICK_RATE,
        total_ticks=600,
        players=_players(),
        objectives=frame(
            "objectives",
            [
                {"tick": 0, "objective_type": "patron", "team_num": TEAM_A,
                 "x": -10_000.0, "y": 0.0, "entity_id": 1, "health": 1000},
                {"tick": 0, "objective_type": "patron", "team_num": TEAM_B,
                 "x": 10_000.0, "y": 0.0, "entity_id": 2, "health": 1000},
            ],
        ),
        neutrals=frame(
            "neutrals",
            [{"tick": 1, "entity_id": 100, "x": 7_000.0, "y": 0.0,
              "health": camp_health, "max_health": 500, "team_num": 4}],
        ),
        player_samples=frame(
            "player_samples",
            [
                {"tick": 100, "hero_id": 1, "x": 2_500.0, "y": 0.0,
                 "health": 1000, "max_health": 1000, "is_alive": True},
                # Ally vision supplies two last-known enemy positions far from the camp.
                {"tick": 100, "hero_id": 2, "x": -7_000.0, "y": 0.0,
                 "health": 1000, "max_health": 1000, "is_alive": True},
                {"tick": 100, "hero_id": 11, "x": -7_100.0, "y": 0.0,
                 "health": 1000, "max_health": 1000, "is_alive": True},
                {"tick": 100, "hero_id": 12, "x": -6_900.0, "y": 0.0,
                 "health": 1000, "max_health": 1000, "is_alive": True},
            ],
        ),
    )


class TestUncommittedLaningWindows:
    """A laning target standing in the open at low health is a real miss even
    when the observer never opened fire — but it can never rank as high."""

    def _match(self, distance: float, target_health: int) -> MatchData:
        rows = []
        for tick in (100, 160):
            rows.extend(
                [
                    {"tick": tick, "hero_id": 1, "x": 0.0, "y": 0.0,
                     "health": 900, "max_health": 1000, "is_alive": True},
                    {"tick": tick, "hero_id": 2, "x": 100.0, "y": 0.0,
                     "health": 800, "max_health": 1000, "is_alive": True},
                    {"tick": tick, "hero_id": 11, "x": distance, "y": 0.0,
                     "health": target_health, "max_health": 1000, "is_alive": True},
                    {"tick": tick, "hero_id": 12, "x": 10_000.0, "y": 0.0,
                     "health": 1000, "max_health": 1000, "is_alive": True},
                ]
            )
        return MatchData(
            tick_rate=TICK_RATE,
            total_ticks=600,
            players=_players(),
            player_samples=frame("player_samples", rows),
        )

    def test_low_enemy_within_reach_in_lane_is_a_window_without_a_trade(self):
        windows = analyze_opportunities(self._match(1_200.0, 200)).kill_windows
        mine = [w for w in windows if w.observer_hero_id == 1]
        assert len(mine) == 1
        assert mine[0].recent_damage_by_observer == 0
        assert "no committed trade" in " ".join(mine[0].evidence)

    def test_an_uncommitted_read_never_reaches_high_confidence(self):
        windows = analyze_opportunities(self._match(1_200.0, 200)).kill_windows
        assert all(w.confidence != "high" for w in windows)

    def test_out_of_reach_targets_are_not_claimed_without_a_trade(self):
        # 2400 units is inside the kill range but past the positional-read
        # cutoff, so with no damage behind it there is nothing to assert.
        assert analyze_opportunities(self._match(2_400.0, 200)).kill_windows == ()


EXTRA_DEFENDERS = [12, 14, 15, 16, 17]


def _rotation_match(
    *,
    observer_x: float = -5_000.0,
    ally_count: int = 1,
    defenders: int = 2,
    observer_follows: bool = False,
    first_death_tick: int = 800,
    observer_team_wins: bool = False,
) -> MatchData:
    """Observer idle in the yellow lane; a teammate is trading in purple.

    Lanes are located from structures, so the fixture gives each lane a Walker
    at either end and parks the heroes on top of them. The enemy roster is sized
    to ``defenders`` rather than parking spares off-screen: an unseen hero counts
    as an unknown position, which moves the score and would make these fixtures
    test the unknown-count bonus instead of the numbers rule they are about.
    """
    extras = EXTRA_DEFENDERS[: defenders - 1]
    players = frame(
        "players",
        [
            {"hero_id": 1, "team_num": TEAM_A, "start_lane": 1, "player_name": "observer"},
            {"hero_id": 2, "team_num": TEAM_A, "start_lane": 6, "player_name": "ally"},
            {"hero_id": 3, "team_num": TEAM_A, "start_lane": 6, "player_name": "ally2"},
            {"hero_id": 11, "team_num": TEAM_B, "start_lane": 6, "player_name": "target"},
            {"hero_id": 13, "team_num": TEAM_B, "start_lane": 1, "player_name": "hidden"},
        ]
        + [
            {"hero_id": h, "team_num": TEAM_B, "start_lane": 6, "player_name": f"defender{h}"}
            for h in extras
        ],
    )
    objectives = frame(
        "objectives",
        [
            {"tick": 0, "objective_type": "walker", "team_num": TEAM_A, "lane": 1,
             "x": -8_000.0, "y": 0.0, "entity_id": 1, "health": 12_000, "max_health": 12_000},
            {"tick": 0, "objective_type": "walker", "team_num": TEAM_B, "lane": 1,
             "x": -8_000.0, "y": 4_000.0, "entity_id": 2, "health": 12_000, "max_health": 12_000},
            {"tick": 0, "objective_type": "walker", "team_num": TEAM_A, "lane": 6,
             "x": 0.0, "y": 0.0, "entity_id": 3, "health": 12_000, "max_health": 12_000},
            {"tick": 0, "objective_type": "walker", "team_num": TEAM_B, "lane": 6,
             "x": 0.0, "y": 4_000.0, "entity_id": 4, "health": 12_000, "max_health": 12_000},
        ],
    )
    rows = []
    for tick in (100, 160, 220, 600, 900):
        # The observer walks to the fight only in the "follows" variant.
        here = 0.0 if observer_follows and tick >= 600 else observer_x
        rows.extend(
            [
                {"tick": tick, "hero_id": 1, "x": here, "y": 0.0,
                 "health": 1000, "max_health": 1000, "is_alive": True},
                {"tick": tick, "hero_id": 2, "x": 0.0, "y": 0.0,
                 "health": 700, "max_health": 1000, "is_alive": True},
                {"tick": tick, "hero_id": 11, "x": 200.0, "y": 0.0,
                 "health": 250, "max_health": 1000, "is_alive": True},
                {"tick": tick, "hero_id": 13, "x": -8_100.0, "y": 9_000.0,
                 "health": 1000, "max_health": 1000, "is_alive": True},
            ]
        )
        # Defenders stand beside the target, inside the engage radius.
        rows.extend(
            {"tick": tick, "hero_id": hero, "x": 400.0 + 100.0 * i, "y": 0.0,
             "health": 900, "max_health": 1000, "is_alive": True}
            for i, hero in enumerate(extras)
        )
        if ally_count > 1:
            rows.append(
                {"tick": tick, "hero_id": 3, "x": 100.0, "y": 0.0,
                 "health": 1000, "max_health": 1000, "is_alive": True}
            )
    return MatchData(
        tick_rate=TICK_RATE,
        total_ticks=1200,
        players=players,
        objectives=objectives,
        player_samples=frame("player_samples", rows),
        damage=frame(
            "damage",
            [{"tick": 95, "damage": 200, "attacker_hero_id": 2, "victim_hero_id": 11}],
        ),
        kills=frame(
            "kills",
            [
                {
                    "tick": first_death_tick,
                    "victim_hero_id": 11 if observer_team_wins else 2,
                    "attacker_hero_id": 2 if observer_team_wins else 11,
                    "assister_hero_ids": [],
                }
            ],
        ),
        teamfights=frame(
            "teamfights",
            [
                {
                    "fight_id": 1,
                    "start_tick": 90,
                    "end_tick": 840,
                    "duration_seconds": 12.5,
                    "participants": [2, 11, *extras] + ([3] if ally_count > 1 else []),
                    "num_participants": ally_count + defenders,
                    "hero_damage": 2000,
                    "kills": 1,
                }
            ],
        ),
    )


class TestRotationOpportunities:
    def test_a_free_player_owes_the_even_fight_in_another_lane(self):
        windows = analyze_opportunities(_rotation_match()).rotation_windows
        mine = [w for w in windows if w.observer_hero_id == 1]
        assert len(mine) == 1
        assert mine[0].target_hero_id == 11
        assert (mine[0].from_lane, mine[0].to_lane) == (1, 6)
        assert mine[0].allies_engaged == 1
        assert mine[0].known_enemies_there == 2
        assert mine[0].estimated_travel_seconds < mine[0].seconds_before_first_ally_death
        assert mine[0].actual_kill_margin == 1

    def test_a_player_who_actually_rotated_is_not_faulted_for_it(self):
        windows = analyze_opportunities(
            _rotation_match(observer_follows=True)
        ).rotation_windows
        assert [w for w in windows if w.observer_hero_id == 1] == []

    def test_no_window_for_a_fight_you_cannot_fix_by_showing_up(self):
        """Two allies against the whole enemy team is a lost fight, not a
        rotation someone owed. Faulting a player for skipping it turns the last
        stand of a losing match into a list of mistakes."""
        windows = analyze_opportunities(_rotation_match(defenders=5)).rotation_windows
        assert [w for w in windows if w.observer_hero_id == 1] == []

    def test_arrival_must_make_the_actual_fight_even(self):
        windows = analyze_opportunities(
            _rotation_match(ally_count=2, defenders=3)
        ).rotation_windows
        mine = [w for w in windows if w.observer_hero_id == 1]
        assert len(mine) == 1
        assert mine[0].known_enemies_there == 3

    def test_no_window_when_arrival_is_after_the_first_ally_dies(self):
        windows = analyze_opportunities(
            _rotation_match(first_death_tick=300)
        ).rotation_windows
        assert [w for w in windows if w.observer_hero_id == 1] == []

    def test_no_window_for_a_fight_the_team_already_won(self):
        windows = analyze_opportunities(
            _rotation_match(observer_team_wins=True)
        ).rotation_windows
        assert [w for w in windows if w.observer_hero_id == 1] == []

    def test_no_window_when_teammates_already_outnumber_them(self):
        windows = analyze_opportunities(
            _rotation_match(ally_count=2, defenders=1)
        ).rotation_windows
        assert [w for w in windows if w.observer_hero_id == 1] == []


def _macro_match(
    *,
    objective_damage: int = 0,
    dead_enemies: int = 2,
    mid_boss: bool = False,
    urn_x: float | None = None,
    down_minute: int = 11,
    mid_boss_x: float = -5_000.0,
) -> MatchData:
    """Team A is up two bodies for a full minute at 11:00 and spends it on nothing.

    The clock matters: the Mid Boss is only assumed contestable from 10:00, so a
    fixture set in the first minutes could never exercise that branch.
    """
    dead = [11, 12][:dead_enemies]
    down, up = down_minute * MINUTE, (down_minute + 1) * MINUTE
    rows = []
    for tick in range(0, 15 * MINUTE, 60):
        for hero in (1, 2, 3):
            rows.append(
                {"tick": tick, "hero_id": hero, "x": -6_000.0, "y": 0.0,
                 "health": 1000, "max_health": 1000, "is_alive": True,
                 # Counters are cumulative, so the damage has to land *inside*
                 # the window to register as a delta against its opening value.
                 "objective_damage": objective_damage if tick > down else 0}
            )
        for hero in (11, 12, 13):
            alive = hero not in dead or not down <= tick < up
            rows.append(
                {"tick": tick, "hero_id": hero, "x": 6_000.0, "y": 0.0,
                 "health": 1000, "max_health": 1000, "is_alive": alive,
                 "objective_damage": 0}
            )
    players = frame(
        "players",
        [
            {"hero_id": 1, "team_num": TEAM_A, "start_lane": 1},
            {"hero_id": 2, "team_num": TEAM_A, "start_lane": 6},
            {"hero_id": 3, "team_num": TEAM_A, "start_lane": 4},
            {"hero_id": 11, "team_num": TEAM_B, "start_lane": 1},
            {"hero_id": 12, "team_num": TEAM_B, "start_lane": 6},
            {"hero_id": 13, "team_num": TEAM_B, "start_lane": 4},
        ],
    )
    return MatchData(
        tick_rate=TICK_RATE,
        total_ticks=15 * MINUTE,
        game_over_tick=15 * MINUTE,
        players=players,
        kills=frame(
            "kills",
            # An early cross-lane pickoff closes the laning phase, so the man
            # advantage lands in the mid game where macro decisions actually
            # live. The samples never show hero 13 dead, so it is a one-sample
            # blip and never a man advantage of its own.
            [{"tick": 120, "victim_hero_id": 13, "attacker_hero_id": 2,
              "assister_hero_ids": []}]
            + [
                {"tick": down, "victim_hero_id": hero, "attacker_hero_id": 1,
                 "assister_hero_ids": []}
                for hero in dead
            ],
        ),
        objectives=frame(
            "objectives",
            [
                {"tick": 0, "objective_type": "walker", "team_num": TEAM_B, "lane": 1,
                 "x": 4_000.0, "y": 0.0, "entity_id": 1,
                 "health": 12_000, "max_health": 12_000},
                {"tick": 0, "objective_type": "walker", "team_num": TEAM_A, "lane": 1,
                 "x": -4_000.0, "y": 0.0, "entity_id": 2,
                 "health": 12_000, "max_health": 12_000},
            ]
            + (
                []
                if not mid_boss
                # The boss position lives in the objectives frame, not the
                # mid_boss event frame, and without it the boss cannot be
                # ranked on distance at all.
                else [
                    {"tick": 0, "objective_type": "mid_boss", "team_num": 4, "lane": None,
                     "x": mid_boss_x, "y": 0.0, "entity_id": 3,
                     "health": 13_000, "max_health": 13_000}
                ]
            ),
        ),
        mid_boss=frame(
            "mid_boss", [{"tick": 1, "team_num": 4, "event": "spawned"}] if mid_boss else []
        ),
        urn=frame(
            "urn",
            []
            if urn_x is None
            else [{"tick": 1, "event": "returned", "hero_id": 0, "team_num": 0,
                   "x": urn_x, "y": 0.0, "z": 0.0}],
        ),
        player_samples=frame("player_samples", rows),
    )


class TestMacroOpportunities:
    def test_an_unspent_man_advantage_becomes_a_siege_window(self):
        windows = analyze_opportunities(_macro_match()).macro_windows
        siege = [w for w in windows if w.kind == "siege"]
        assert siege
        assert {w.observer_hero_id for w in siege} == {1, 2, 3}
        assert siege[0].enemies_known_dead == 2
        assert siege[0].target_kind == "walker"
        assert siege[0].target_team_num == TEAM_B

    def test_a_team_that_hit_a_structure_gets_no_window(self):
        windows = analyze_opportunities(_macro_match(objective_damage=5_000)).macro_windows
        assert [w for w in windows if w.kind == "siege"] == []

    def test_one_dead_opponent_is_not_a_man_advantage(self):
        windows = analyze_opportunities(_macro_match(dead_enemies=1)).macro_windows
        assert [w for w in windows if w.kind == "siege"] == []

    def test_an_unclaimed_urn_wins_when_it_is_closer_than_the_structure(self):
        """The Patron always stands, so ordering the Urn behind "nearest
        standing structure" would make it unreachable in every real match."""
        # Team A sits at -6000; the Walker is at +4000 and the Urn at -5000.
        windows = analyze_opportunities(_macro_match(urn_x=-5_000.0)).macro_windows
        urn = [w for w in windows if w.kind == "urn"]
        assert urn
        assert urn[0].target_kind == "urn"
        assert urn[0].distance == 1_000.0

    def test_a_distant_urn_does_not_displace_the_siege_target(self):
        windows = analyze_opportunities(_macro_match(urn_x=9_000.0)).macro_windows
        assert {w.kind for w in windows if w.kind != "defend"} == {"siege"}

    def test_a_carried_urn_is_not_an_available_one(self):
        md = _macro_match(urn_x=-5_000.0)
        carried = list(md.urn.iter_rows(named=True)) + [
            {"tick": 2, "event": "picked_up", "hero_id": 11, "team_num": 0,
             "x": -5_000.0, "y": 0.0, "z": 0.0}
        ]
        windows = analyze_opportunities(
            md.with_frames(urn=frame("urn", carried))
        ).macro_windows
        assert [w for w in windows if w.kind == "urn"] == []

    def test_the_spawn_event_does_not_open_the_mid_boss(self):
        """The demo announces the Mid Boss entity spawning around 00:30, long
        before it can be attacked. Trusting that tick marked the whole match as
        "the boss was up" and buried every siege call behind it."""
        windows = analyze_opportunities(
            _macro_match(mid_boss=True, down_minute=3)
        ).macro_windows
        assert "mid_boss" not in {w.kind for w in windows}
        assert all("mid_boss" not in w.alternatives for w in windows)
        assert "siege" in {w.kind for w in windows}

    def test_the_closest_available_objective_is_the_call(self):
        """Whether the Mid Boss beats a siege needs judgement the demo cannot
        supply, so the ranking uses the one thing we measured: travel."""
        windows = analyze_opportunities(_macro_match(mid_boss=True)).macro_windows
        boss = [w for w in windows if w.kind == "mid_boss"]
        assert boss
        assert boss[0].distance == 1_000.0
        assert "walker" in boss[0].alternatives

    def test_a_distant_mid_boss_is_named_as_an_alternative_not_the_call(self):
        windows = analyze_opportunities(
            _macro_match(mid_boss=True, mid_boss_x=20_000.0)
        ).macro_windows
        siege = [w for w in windows if w.kind == "siege"]
        assert siege
        assert "mid_boss" in siege[0].alternatives


def _defend_match(distance: float) -> MatchData:
    rows = []
    for tick in range(0, 3_000, 60):
        rows.append(
            {"tick": tick, "hero_id": 1, "x": distance, "y": 0.0,
             "health": 1000, "max_health": 1000, "is_alive": True}
        )
        rows.append(
            {"tick": tick, "hero_id": 11, "x": 0.0, "y": 0.0,
             "health": 1000, "max_health": 1000, "is_alive": True}
        )
    return MatchData(
        tick_rate=TICK_RATE,
        total_ticks=3_000,
        game_over_tick=3_000,
        players=frame(
            "players",
            [
                {"hero_id": 1, "team_num": TEAM_A, "start_lane": 1},
                {"hero_id": 11, "team_num": TEAM_B, "start_lane": 1},
            ],
        ),
        objectives=frame(
            "objectives",
            [
                {"tick": 600, "objective_type": "walker", "team_num": TEAM_A, "lane": 1,
                 "x": 0.0, "y": 0.0, "entity_id": 1,
                 "health": 12_000, "max_health": 12_000},
                {"tick": 1_200, "objective_type": "walker", "team_num": TEAM_A, "lane": 1,
                 "x": 0.0, "y": 0.0, "entity_id": 1,
                 "health": 4_000, "max_health": 12_000},
            ],
        ),
        player_samples=frame("player_samples", rows),
    )


class TestDefenceWindows:
    def test_a_structure_bleeding_out_while_you_were_across_the_map(self):
        windows = analyze_opportunities(_defend_match(9_000.0)).macro_windows
        defend = [w for w in windows if w.kind == "defend"]
        assert len(defend) == 1
        assert defend[0].observer_hero_id == 1
        assert defend[0].target_kind == "walker"
        assert "67%" in " ".join(defend[0].evidence)

    def test_a_player_already_standing_on_it_is_not_faulted(self):
        windows = analyze_opportunities(_defend_match(500.0)).macro_windows
        assert [w for w in windows if w.kind == "defend"] == []


def _crowded_camp_match(loiter_x: float) -> MatchData:
    """Two opponents across the map, one loitering near the camp.

    The camp sits at (7000, 0). ``loiter_x`` moves the third opponent between
    "comfortably clear" and "close enough to contest".
    """
    return MatchData(
        tick_rate=TICK_RATE,
        total_ticks=600,
        players=frame(
            "players",
            [
                {"hero_id": 1, "team_num": TEAM_A, "start_lane": 1, "player_name": "observer"},
                {"hero_id": 2, "team_num": TEAM_A, "start_lane": 1, "player_name": "ally"},
                {"hero_id": 11, "team_num": TEAM_B, "start_lane": 1},
                {"hero_id": 12, "team_num": TEAM_B, "start_lane": 4},
                {"hero_id": 13, "team_num": TEAM_B, "start_lane": 6},
            ],
        ),
        objectives=frame(
            "objectives",
            [
                {"tick": 0, "objective_type": "patron", "team_num": TEAM_A,
                 "x": -10_000.0, "y": 0.0, "entity_id": 1, "health": 1000},
                {"tick": 0, "objective_type": "patron", "team_num": TEAM_B,
                 "x": 10_000.0, "y": 0.0, "entity_id": 2, "health": 1000},
            ],
        ),
        neutrals=frame(
            "neutrals",
            [{"tick": 1, "entity_id": 100, "x": 7_000.0, "y": 0.0,
              "health": 500, "max_health": 500, "team_num": 4}],
        ),
        player_samples=frame(
            "player_samples",
            [
                {"tick": 100, "hero_id": 1, "x": 6_500.0, "y": 0.0,
                 "health": 1000, "max_health": 1000, "is_alive": True},
                # The ally's vision is what places every opponent below.
                {"tick": 100, "hero_id": 2, "x": 3_000.0, "y": 0.0,
                 "health": 1000, "max_health": 1000, "is_alive": True},
                {"tick": 100, "hero_id": 11, "x": 1_000.0, "y": 0.0,
                 "health": 1000, "max_health": 1000, "is_alive": True},
                {"tick": 100, "hero_id": 12, "x": 1_200.0, "y": 0.0,
                 "health": 1000, "max_health": 1000, "is_alive": True},
                {"tick": 100, "hero_id": 13, "x": loiter_x, "y": 0.0,
                 "health": 1000, "max_health": 1000, "is_alive": True},
            ],
        ),
    )


class TestCampSafetyIsReportedWhole:
    """Counting only the opponents that were comfortably far away told the
    reassuring half of the story. On match 98811241 a camp was called clear
    with "3 opponents at least 4500 units away" while the other three sat at
    3979, 4147 and 4440 — barely further than the one that counted as away."""

    def test_the_nearest_known_opponent_is_measured_and_named(self):
        windows = analyze_opportunities(_crowded_camp_match(3_200.0)).jungle_windows
        mine = [w for w in windows if w.observer_hero_id == 1]
        assert mine
        assert mine[0].nearest_known_enemy == 3_800.0
        assert "the nearest was 3800 units from the camp" in " ".join(mine[0].evidence)

    def test_an_opponent_inside_the_away_range_makes_it_a_scout_call(self):
        windows = analyze_opportunities(_crowded_camp_match(3_200.0)).jungle_windows
        mine = [w for w in windows if w.observer_hero_id == 1]
        assert mine[0].confidence == "low"

    def test_a_genuinely_clear_camp_still_reads_as_actionable(self):
        windows = analyze_opportunities(_crowded_camp_match(1_400.0)).jungle_windows
        mine = [w for w in windows if w.observer_hero_id == 1]
        assert mine[0].nearest_known_enemy == 5_600.0
        assert mine[0].confidence == "medium"


class TestJungleOpportunities:
    def test_hidden_camp_health_does_not_change_a_scout_recommendation(self):
        alive = analyze_opportunities(_jungle_match(500)).jungle_windows
        dead = analyze_opportunities(_jungle_match(0)).jungle_windows
        mine_alive = tuple(w for w in alive if w.observer_hero_id == 1)
        mine_dead = tuple(w for w in dead if w.observer_hero_id == 1)
        assert mine_alive == mine_dead
        assert mine_alive
        assert mine_alive[0].camp_status.startswith("unknown")

    def test_hidden_camp_is_a_scout_window_not_a_claim_that_it_is_available(self):
        windows = analyze_opportunities(_jungle_match(500)).jungle_windows
        mine = [w for w in windows if w.observer_hero_id == 1]
        assert mine[0].confidence == "low"
        assert "hidden camp state was not used" in " ".join(mine[0].limitations)


class TestMidBossGate:
    """When the Mid Boss becomes contestable moved mid-2026, so the gate has to."""

    def test_an_old_replay_keeps_the_ten_minute_gate(self, match):
        old = replace(match, match_id=70_000_000)
        assert mid_boss_available_from(old) == MID_BOSS_AVAILABLE_SECONDS

    def test_a_replay_after_the_ungating_patch_has_no_gate(self, match):
        new = replace(match, match_id=MID_BOSS_UNGATED_FROM_MATCH_ID + 1)
        assert mid_boss_available_from(new) == 0.0

    def test_a_demo_without_a_match_id_stays_conservative(self, match):
        # Under-reporting macro windows is a smaller error than inventing half
        # an hour of them.
        assert mid_boss_available_from(replace(match, match_id=None)) == (
            MID_BOSS_AVAILABLE_SECONDS
        )

    def test_an_explicit_override_wins(self, match):
        assert mid_boss_available_from(match, override=120.0) == 120.0
