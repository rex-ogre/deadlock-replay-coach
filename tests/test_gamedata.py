"""Tests for the game-constants layer.

These numbers are fetched from a live API that tracks a game patched most
weeks, so what is worth pinning is not the values but the *degradation*: a run
must survive an unreachable API, a corrupt cache, and a hero the snapshot has
never heard of.
"""

from __future__ import annotations

import json

import pytest

from deadlock_coach import gamedata


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("DEADLOCK_COACH_CACHE", str(tmp_path))
    gamedata.reset_cache()
    yield
    gamedata.reset_cache()


class TestBundledSnapshot:
    def test_it_loads_without_a_network(self):
        constants = gamedata.load_constants(offline=True)
        assert constants.source == "bundled"
        assert len(constants.heroes) > 20

    def test_movement_numbers_are_in_metres_per_second(self):
        hero = gamedata.load_constants(offline=True).hero(1)
        # A hero walks at jogging pace, not at highway speed. This catches a
        # unit mix-up in the snapshot far more reliably than an exact value,
        # which a patch is free to change.
        assert 4.0 < hero.max_move_speed < 12.0
        assert hero.sprint_total_speed > hero.max_move_speed

    def test_dashing_is_faster_than_sprinting_for_every_hero(self):
        for hero in gamedata.load_constants(offline=True).heroes.values():
            assert hero.ground_dash_speed > hero.sprint_total_speed, hero.name

    def test_the_roster_has_more_than_one_dash_duration(self):
        # The 2026-07-28 patch split the cast into stamina buckets. If this
        # ever collapses to one value, the per-hero modelling is pointless and
        # someone should be told rather than the code quietly over-fitting.
        durations = {h.ground_dash_duration for h in gamedata.load_constants(offline=True).heroes.values()}
        assert len(durations) > 1

    def test_zip_lines_outrun_every_hero(self):
        constants = gamedata.load_constants(offline=True)
        fastest = max(h.sprint_total_speed for h in constants.heroes.values())
        assert constants.zipline.speed_inner > fastest
        assert constants.zipline.speed_outer > constants.zipline.speed_inner

    def test_a_dismount_carries_most_of_the_line_speed(self):
        zipline = gamedata.load_constants(offline=True).zipline
        assert zipline.dismount_carry_speed > zipline.speed_outer * 0.5
        assert zipline.dismount_carry_speed < zipline.speed_outer


class TestDegradation:
    def test_an_unknown_hero_returns_none_rather_than_guessing(self):
        assert gamedata.load_constants(offline=True).hero(999_999) is None

    def test_a_null_hero_id_is_tolerated(self):
        assert gamedata.load_constants(offline=True).hero(None) is None

    def test_a_corrupt_cache_is_ignored_not_fatal(self, tmp_path):
        (tmp_path / "game_constants.json").write_text("{not json")
        assert gamedata.load_constants(offline=True).heroes

    def test_a_stale_cache_still_beats_the_bundled_snapshot(self, tmp_path, monkeypatch):
        payload = json.loads(gamedata.BUNDLED_SNAPSHOT.read_text())
        payload["heroes"] = {"1": payload["heroes"]["1"]}
        payload["fetched_at"] = "2026-01-01T00:00:00+00:00"
        (tmp_path / "game_constants.json").write_text(json.dumps(payload))

        def explode():
            raise OSError("no network")

        monkeypatch.setattr(gamedata, "fetch_snapshot", lambda **kw: explode())
        constants = gamedata.load_constants(refresh=True)
        assert constants.source == "cache"
        assert constants.fetched_at == "2026-01-01T00:00:00+00:00"

    def test_offline_never_calls_out(self, monkeypatch):
        def explode(*args, **kwargs):
            raise AssertionError("offline mode must not touch the network")

        monkeypatch.setattr(gamedata, "_get_json", explode)
        assert gamedata.load_constants(offline=True).heroes


class TestAccuracyBaseline:
    def test_a_known_hero_has_a_population_median(self):
        baseline = gamedata.load_constants(offline=True).baseline_accuracy(1, None)
        assert baseline is not None
        share, matches = baseline
        assert 0.2 < share < 0.95
        assert matches > 0

    def test_thin_rank_rows_fall_back_to_the_all_rank_aggregate(self):
        constants = gamedata.load_constants(offline=True)
        # A badge nobody in the sample plays at must not produce a confident
        # number off a handful of matches.
        baseline = constants.baseline_accuracy(1, 999)
        assert baseline == constants.accuracy_baseline[1].get(0)

    def test_an_unknown_hero_has_no_baseline(self):
        assert gamedata.load_constants(offline=True).baseline_accuracy(999_999, 61) is None


class TestHeroBaseline:
    """The population rows behind "you versus the top rank"."""

    def test_the_top_row_is_a_higher_rank_than_the_players_own(self):
        constants = gamedata.load_constants(offline=True)
        peer = constants.peer_baseline(1, 61)
        top = constants.top_baseline(1)
        assert peer is not None and top is not None
        assert top.badge > peer.badge
        assert top.matches >= gamedata.MIN_BASELINE_MATCHES

    def test_thin_buckets_are_never_quoted_as_the_top(self):
        constants = gamedata.load_constants(offline=True)
        for hero_id in constants.hero_baseline:
            top = constants.top_baseline(hero_id)
            if top is not None:
                assert top.matches >= gamedata.MIN_BASELINE_MATCHES, hero_id

    def test_accuracy_rises_with_rank(self):
        """The one sanity check on the whole comparison being worth making."""
        constants = gamedata.load_constants(offline=True)
        low = constants.peer_baseline(1, 21)
        top = constants.top_baseline(1)
        assert low is not None and top is not None
        assert top.accuracy > low.accuracy

    def test_an_unknown_hero_has_no_rows(self):
        constants = gamedata.load_constants(offline=True)
        assert constants.peer_baseline(999_999, 61) is None
        assert constants.top_baseline(999_999) is None

    def test_accuracy_is_derived_from_the_same_rows(self):
        constants = gamedata.load_constants(offline=True)
        row = constants.peer_baseline(1, 61)
        share, matches = constants.baseline_accuracy(1, 61)
        assert share == pytest.approx(row.accuracy, abs=1e-4)
        assert matches == row.matches

    def test_an_older_cache_without_the_wide_table_still_loads(self, tmp_path, monkeypatch):
        """Caches written before the population counters existed stay usable."""
        payload = json.loads(gamedata.BUNDLED_SNAPSHOT.read_text())
        payload.pop("hero_baseline")
        payload["accuracy_baseline"] = {"1": {"0": [0.55, 90_000]}}
        (tmp_path / "game_constants.json").write_text(json.dumps(payload))
        monkeypatch.setattr(
            gamedata, "fetch_snapshot", lambda **kw: (_ for _ in ()).throw(OSError("no network"))
        )
        constants = gamedata.load_constants(refresh=True)
        assert constants.baseline_accuracy(1, None) == (0.55, 90_000)
        assert constants.top_baseline(1) is None

    def test_rank_badges_read_as_names(self):
        constants = gamedata.load_constants(offline=True)
        assert constants.rank_name(101).startswith("Ascendant")
        assert gamedata.badge_name(0) == "all ranks"
        assert gamedata.badge_name(None) == "unknown rank"
