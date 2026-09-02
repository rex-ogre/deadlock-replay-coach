from __future__ import annotations

import polars as pl
import pytest

from deadlock_coach.match import SCHEMAS, Clock, MatchData, conform, empty

from .conftest import MINUTE, TEAM_A, TEAM_B, TICK_RATE


class TestClock:
    def test_linear_conversion(self):
        clock = Clock(60)
        assert clock.seconds(120) == 2.0
        assert clock.mmss(0) == "00:00"
        assert clock.mmss(75 * 60) == "01:15"

    def test_uses_injected_seconds_fn(self):
        """boon's pause-aware clock must win over the linear default."""
        clock = Clock(60, seconds_fn=lambda tick: tick / 60 - 30)
        assert clock.mmss(60 * 60) == "00:30"

    def test_none_tick_is_placeholder_not_zero(self):
        assert Clock(60).mmss(None) == "--:--"

    def test_negative_seconds_clamp_to_zero(self):
        clock = Clock(60, seconds_fn=lambda tick: -5.0)
        assert clock.mmss(100) == "00:00"

    def test_rejects_zero_tick_rate(self):
        with pytest.raises(ValueError):
            Clock(0)


class TestConform:
    def test_adds_missing_columns_as_null(self):
        partial = pl.DataFrame({"tick": [1], "victim_hero_id": [11]})
        out = conform("kills", partial)
        assert set(SCHEMAS["kills"]) <= set(out.columns)
        assert out["attacker_hero_id"][0] is None

    def test_preserves_extra_columns(self):
        partial = pl.DataFrame({"tick": [1], "custom": ["x"]})
        assert "custom" in conform("kills", partial).columns

    def test_none_becomes_empty_frame(self):
        assert conform("kills", None).is_empty()

    def test_empty_frame_has_declared_dtypes(self):
        assert empty("kills").schema["assister_hero_ids"] == pl.List(pl.Int64)


class TestRoster:
    def test_team_lookup(self, match):
        assert match.team_of(1) == TEAM_A
        assert match.team_of(11) == TEAM_B
        assert match.team_of(999) is None

    def test_teammates_exclude_self_and_enemies(self, match):
        assert match.teammates_of(1) == [2, 3]

    def test_spectator_team_is_not_a_playing_team(self, players):
        with_spectator = pl.concat(
            [
                players,
                pl.DataFrame(
                    [{"player_name": "obs", "steam_id": 9, "hero_id": 99,
                      "team_num": 1, "start_lane": 0, "rank": 0}],
                    schema=SCHEMAS["players"],
                ),
            ]
        )
        md = MatchData(players=with_spectator)
        assert md.team_nums == [TEAM_A, TEAM_B]

    def test_player_name_lookup(self, match):
        assert match.player_name(1) == "a0"
        assert match.player_name(999) is None

    def test_end_tick_prefers_game_over(self):
        md = MatchData(total_ticks=10_000, game_over_tick=8_000)
        assert md.end_tick == 8_000

    def test_end_tick_falls_back_to_total(self):
        assert MatchData(total_ticks=10_000).end_tick == 10_000


def test_match_fixture_is_internally_consistent(match):
    assert match.tick_rate == TICK_RATE
    assert match.end_tick == 25 * MINUTE
    assert len(match.hero_ids) == 6
