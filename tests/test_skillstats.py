"""Tests for the post-match mechanical stats.

The whole module is optional by design: Valve drops older matches and the API
is rate limited, so the failure path is exercised at least as hard as the happy
one. An unavailable stat must read as unknown, never as zero.
"""

from __future__ import annotations

import json
import urllib.error

import pytest

from deadlock_coach import gamedata, skillstats


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("DEADLOCK_COACH_CACHE", str(tmp_path))
    gamedata.reset_cache()
    yield
    gamedata.reset_cache()


@pytest.fixture
def constants():
    return gamedata.load_constants(offline=True)


def _metadata(**overrides):
    sample = {
        "shots_hit": 600,
        "shots_missed": 400,
        "hero_bullets_hit": 180,
        "hero_bullets_hit_crit": 18,
        "creep_kills": 80,
        "possible_creeps": 100,
        "time_stamp_s": 1800,
    }
    sample.update(overrides)
    return {
        "match_info": {
            "average_badge_team0": 62,
            "average_badge_team1": 64,
            "players": [
                {
                    "account_id": 492740207,
                    "hero_id": 1,
                    "stats": [dict(sample, shots_hit=100, shots_missed=100), sample],
                }
            ],
        }
    }


def _seed(tmp_path, match_id, payload):
    path = tmp_path / "matches"
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{match_id}.json").write_text(json.dumps(payload))


class TestAccountIds:
    def test_a_steam_id_becomes_an_account_id(self):
        assert skillstats.account_id_of(76561198453005935) == 492740207

    def test_an_account_id_passes_through_unchanged(self):
        assert skillstats.account_id_of(492740207) == 492740207

    def test_a_missing_id_is_none(self):
        assert skillstats.account_id_of(None) is None


class TestParsing:
    def test_the_last_sample_is_the_whole_match(self, tmp_path, constants):
        # The series is cumulative, so an early row must not be mistaken for
        # the total -- that would halve every accuracy in the report.
        _seed(tmp_path, 42, _metadata())
        stat = skillstats.fetch_match_stats(42, constants)[492740207]
        assert stat.shots_taken == 1000
        assert stat.accuracy == pytest.approx(0.6)

    def test_crit_rate_is_a_share_of_hero_hits_not_all_hits(self, tmp_path, constants):
        _seed(tmp_path, 42, _metadata())
        stat = skillstats.fetch_match_stats(42, constants)[492740207]
        assert stat.crit_rate == pytest.approx(0.1)
        assert stat.hero_bullet_share == pytest.approx(0.3)

    def test_creep_efficiency_uses_the_creeps_that_existed(self, tmp_path, constants):
        _seed(tmp_path, 42, _metadata())
        stat = skillstats.fetch_match_stats(42, constants)[492740207]
        assert stat.creep_efficiency == pytest.approx(0.8)

    def test_the_wider_line_is_read_from_wherever_valve_put_it(self, tmp_path, constants):
        """Some counters live on the player row, some in the series.

        ``last_hits`` is only ever on the player row, and the population table
        counts the same field, so reading it from the series alone would report
        a silent zero and make the farming comparison a lie.
        """
        payload = _metadata(
            player_damage=50_000,
            player_damage_taken=60_000,
            boss_damage=4_000,
            creep_damage=44_000,
            neutral_damage=9_000,
            net_worth=41_766,
            deaths=8,
            kills=9,
            assists=19,
            denies=0,
        )
        payload["match_info"]["players"][0]["last_hits"] = 199
        _seed(tmp_path, 42, payload)
        stat = skillstats.fetch_match_stats(42, constants)[492740207]
        assert stat.last_hits == 199
        assert stat.net_worth == 41_766
        assert stat.player_damage == 50_000
        assert stat.player_damage_taken == 60_000
        assert stat.boss_damage == 4_000
        assert stat.deaths == 8
        assert stat.duration_seconds == 1800

    def test_an_absent_counter_reads_zero_rather_than_crashing(self, tmp_path, constants):
        _seed(tmp_path, 42, _metadata())
        stat = skillstats.fetch_match_stats(42, constants)[492740207]
        assert stat.net_worth == 0
        assert stat.last_hits == 0

    def test_the_summary_names_the_baseline_it_compares_against(self, tmp_path, constants):
        _seed(tmp_path, 42, _metadata())
        summary = skillstats.fetch_match_stats(42, constants)[492740207].summary()
        assert "accuracy 60%" in summary
        assert "median for this hero at this rank" in summary


class TestMissingData:
    def test_no_match_id_means_no_stats(self, constants):
        assert skillstats.fetch_match_stats(None, constants) == {}

    def test_offline_means_no_stats(self, constants):
        assert skillstats.fetch_match_stats(42, constants, offline=True) == {}

    def test_a_player_with_no_series_is_skipped_not_zeroed(self, tmp_path, constants):
        payload = _metadata()
        payload["match_info"]["players"][0]["stats"] = []
        _seed(tmp_path, 42, payload)
        assert skillstats.fetch_match_stats(42, constants) == {}

    def test_a_player_who_never_fired_reports_unknown_not_zero(self, tmp_path, constants):
        _seed(tmp_path, 42, _metadata(shots_hit=0, shots_missed=0, hero_bullets_hit=0))
        stat = skillstats.fetch_match_stats(42, constants)[492740207]
        assert stat.accuracy is None
        assert stat.crit_rate is None
        assert stat.summary() is None

    def test_a_corrupt_cache_does_not_take_the_report_down(self, tmp_path, constants, monkeypatch):
        path = tmp_path / "matches"
        path.mkdir(parents=True, exist_ok=True)
        (path / "42.json").write_text("{not json")
        monkeypatch.setattr(
            skillstats.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError())
        )
        assert skillstats.fetch_match_stats(42, constants) == {}

    def test_expired_metadata_falls_back_to_the_match_archive(
        self, constants, monkeypatch
    ):
        archived = [
            {
                "average_badge_team0": 62,
                "average_badge_team1": 64,
                "players": [
                    {
                        "account_id": 492740207,
                        "hero_id": 1,
                        "last_hits": 199,
                        "final_stats": {
                            "shots_hit": 600,
                            "shots_missed": 400,
                            "hero_bullets_hit": 180,
                            "hero_bullets_hit_crit": 18,
                            "creep_kills": 80,
                            "possible_creeps": 100,
                        },
                    }
                ],
            }
        ]

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self):
                return json.dumps(archived).encode()

        def urlopen(request, **kwargs):
            if "/matches/42/metadata" in request.full_url:
                raise urllib.error.HTTPError(request.full_url, 503, "gone", {}, None)
            assert "/v1/matches/metadata?" in request.full_url
            return Response()

        monkeypatch.setattr(skillstats.urllib.request, "urlopen", urlopen)
        stat = skillstats.fetch_match_stats(42, constants)[492740207]
        assert stat.accuracy == pytest.approx(0.6)
        assert stat.crit_rate == pytest.approx(0.1)
        # The archive keeps last_hits on the player row, not in final_stats.
        assert stat.last_hits == 199
        cache_path = skillstats._metadata_path(42)
        assert cache_path.exists()
        assert json.loads(cache_path.read_text())["stats_source"] == "deadlock-api-archive"

    def test_cached_metadata_without_shot_counters_does_not_block_archive_fallback(
        self, tmp_path, constants, monkeypatch
    ):
        _seed(
            tmp_path,
            42,
            {"match_info": {"players": [{"account_id": 492740207, "hero_id": 1}]}},
        )
        archived = [
            {
                "average_badge_team0": 62,
                "average_badge_team1": 64,
                "players": [
                    {
                        "account_id": 492740207,
                        "hero_id": 1,
                        "final_stats": {
                            "shots_hit": 3,
                            "shots_missed": 1,
                            "hero_bullets_hit": 2,
                            "hero_bullets_hit_crit": 1,
                            "creep_kills": 1,
                            "possible_creeps": 2,
                        },
                    }
                ],
            }
        ]

        class Response:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def read(self):
                return json.dumps(self.payload).encode()

        def urlopen(request, **kwargs):
            if "/matches/42/metadata" in request.full_url:
                return Response({"match_info": {"players": []}})
            return Response(archived)

        monkeypatch.setattr(skillstats.urllib.request, "urlopen", urlopen)
        stat = skillstats.fetch_match_stats(42, constants)[492740207]
        assert stat.accuracy == pytest.approx(0.75)


class TestHeroKeying:
    def test_stats_are_rekeyed_from_accounts_onto_heroes(self, tmp_path, constants):
        _seed(tmp_path, 42, _metadata())
        stats = skillstats.fetch_match_stats(42, constants)
        by_hero = skillstats.by_hero(stats, {7: 76561198453005935})
        assert by_hero[7].account_id == 492740207

    def test_a_player_missing_from_the_record_is_simply_absent(self, tmp_path, constants):
        _seed(tmp_path, 42, _metadata())
        stats = skillstats.fetch_match_stats(42, constants)
        assert skillstats.by_hero(stats, {7: 76561198000000000}) == {}

    def test_an_anonymised_demo_can_match_stats_by_hero(self, tmp_path, constants):
        _seed(tmp_path, 42, _metadata())
        stats = skillstats.fetch_match_stats(42, constants)
        by_hero = skillstats.by_hero(stats, {1: None})
        assert by_hero[1].account_id == 492740207
