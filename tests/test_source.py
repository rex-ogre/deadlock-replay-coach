"""Adapter tests that need no `.dem` on disk.

The point of `source.py` is to survive a demo that is missing datasets, so the
fake below is deliberately hostile: some properties raise, some return None.
"""

from __future__ import annotations

import polars as pl

from deadlock_coach.source import (
    _frame,
    _player_samples,
    _preload_event_datasets,
    _sample_frames,
    _sample_ticks,
    _teamfights,
    _trooper_samples,
)

from .conftest import frame


class FakeDemo:
    tick_rate = 60
    total_ticks = 600

    def __init__(self, **overrides):
        self._overrides = overrides
        self.snapshot_calls: list[list[int]] = []

    def __getattr__(self, name):
        if name in self._overrides:
            value = self._overrides[name]
            if isinstance(value, Exception):
                raise value
            return value
        raise AttributeError(name)

    def snapshots(self, dataset, *, ticks=None, **kwargs):
        self.snapshot_calls.append(list(ticks or []))
        return pl.DataFrame(
            {
                "tick": ticks,
                "hero_id": [1] * len(ticks),
                "x": [0.0] * len(ticks),
                "souls": [100] * len(ticks),
                "junk_column": [None] * len(ticks),
            }
        )


class TestFrame:
    def test_event_datasets_are_preloaded_in_one_shared_pass(self):
        class Loadable:
            def __init__(self):
                self.calls = []

            def load(self, *datasets):
                self.calls.append(datasets)

        demo = Loadable()
        _preload_event_datasets(demo)
        assert len(demo.calls) == 1
        assert {
            "kills", "damage", "objectives", "world_ticks", "item_purchases",
            "ability_upgrades", "ability_ticks", "abilities",
        } <= set(demo.calls[0])

    def test_failed_batch_preload_leaves_individual_fallback_available(self):
        class OldDemo:
            kills = frame("kills", [{"tick": 1}])

            def load(self, *datasets):
                raise RuntimeError("unsupported dataset")

        demo = OldDemo()
        _preload_event_datasets(demo)
        assert _frame(demo, "kills").height == 1

    def test_returns_the_dataset_when_present(self):
        demo = FakeDemo(kills=frame("kills", [{"tick": 1, "victim_hero_id": 11}]))
        assert _frame(demo, "kills").height == 1

    def test_missing_dataset_degrades_to_empty(self):
        assert _frame(FakeDemo(), "kills").is_empty()

    def test_raising_dataset_degrades_to_empty(self):
        demo = FakeDemo(urn=RuntimeError("no urn in this mode"))
        assert _frame(demo, "urn").is_empty()

    def test_none_dataset_degrades_to_empty(self):
        assert _frame(FakeDemo(rift=None), "rift").is_empty()

    def test_failed_teamfight_detection_is_not_fatal(self):
        demo = FakeDemo(teamfights=ValueError("tick rate is 0"))
        assert _teamfights(demo).is_empty()

    def test_teamfights_as_a_method_is_called_with_the_threshold(self):
        """boon exposes teamfight detection as a method, unlike the datasets,
        and min_players must reach it — the default of 3 counts lane poke."""
        want = frame("teamfights", [{"fight_id": 1, "start_tick": 1, "end_tick": 2}])
        seen: dict = {}

        def fake(*, min_players):
            seen["min_players"] = min_players
            return want

        assert _teamfights(FakeDemo(teamfights=fake), min_players=5).height == 1
        assert seen == {"min_players": 5}

    def test_teamfights_as_a_property_also_works(self):
        want = frame("teamfights", [{"fight_id": 1, "start_tick": 1, "end_tick": 2}])
        assert _teamfights(FakeDemo(teamfights=want)).height == 1

    def test_teamfights_returning_none_degrades_to_empty(self):
        assert _teamfights(FakeDemo(teamfights=lambda: None)).is_empty()


class TestSampleTicks:
    def test_strides_across_the_match(self):
        ticks = _sample_ticks(FakeDemo(), frame("kills", []), sample_seconds=5)
        assert ticks == [0, 300, 600]

    def test_kill_ticks_are_always_included_exactly(self):
        kills = frame("kills", [{"tick": 137, "victim_hero_id": 11}])
        ticks = _sample_ticks(FakeDemo(), kills, sample_seconds=5)
        assert 137 in ticks

    def test_deduplicates_and_sorts(self):
        kills = frame("kills", [{"tick": 300}, {"tick": 300}, {"tick": 10}])
        ticks = _sample_ticks(FakeDemo(), kills, sample_seconds=5)
        assert ticks == sorted(set(ticks))
        assert ticks.count(300) == 1

    def test_sub_second_interval_never_produces_a_zero_stride(self):
        ticks = _sample_ticks(FakeDemo(), frame("kills", []), sample_seconds=0.001)
        assert len(ticks) == 601


class TestPlayerSamples:
    def test_requests_one_pass_and_drops_unknown_columns(self):
        demo = FakeDemo()
        out = _player_samples(demo, frame("kills", [{"tick": 137}]), sample_seconds=5)
        assert len(demo.snapshot_calls) == 1
        assert "junk_column" not in out.columns
        assert set(out.columns) <= {"tick", "hero_id", "x", "souls"}

    def test_snapshot_failure_degrades_to_empty(self):
        class Broken(FakeDemo):
            def snapshots(self, *a, **k):
                raise RuntimeError("decode failed")

        assert _player_samples(Broken(), frame("kills", []), sample_seconds=5).is_empty()

    def test_no_ticks_means_no_snapshot_call(self):
        demo = FakeDemo(total_ticks=0)
        demo.total_ticks = 0
        out = _player_samples(demo, frame("kills", []), sample_seconds=5)
        assert out.is_empty()
        assert demo.snapshot_calls == []


class TestTrooperSamples:
    def test_samples_only_the_post_fight_decision_window(self):
        class TrooperDemo(FakeDemo):
            total_ticks = 3_000

            def __init__(self):
                super().__init__()
                self.requested: list[int] = []

            def snapshots(self, dataset, *, ticks=None, **kwargs):
                assert dataset == "troopers"
                self.requested = list(ticks or [])
                return frame(
                    "trooper_samples",
                    [
                        {
                            "tick": tick,
                            "trooper_type": "trooper",
                            "team_num": 2,
                            "lane": 1,
                            "health": 100,
                            "max_health": 100,
                            "x": 0.0,
                            "y": 0.0,
                            "z": 0.0,
                            "entity_id": tick,
                        }
                        for tick in ticks or []
                    ],
                )

        demo = TrooperDemo()
        fights = frame("teamfights", [{"fight_id": 1, "start_tick": 900, "end_tick": 1_200}])
        out = _trooper_samples(demo, fights)
        assert demo.requested == [1_200, 1_500, 1_800, 2_100, 2_400]
        assert out["tick"].to_list() == demo.requested
        assert out.height == 5  # not the multi-million-row raw dataset

    def test_empty_fights_do_not_request_troopers(self):
        demo = FakeDemo()
        assert _trooper_samples(demo, frame("teamfights", [])).is_empty()
        assert demo.snapshot_calls == []

    def test_snapshot_failure_degrades_to_empty(self):
        class Broken(FakeDemo):
            def snapshots(self, *args, **kwargs):
                raise RuntimeError("trooper decode failed")

        fights = frame("teamfights", [{"fight_id": 1, "end_tick": 1_200}])
        assert _trooper_samples(Broken(), fights).is_empty()


class TestCombinedSamples:
    def test_player_and_trooper_samples_share_one_snapshot_pass(self):
        class CombinedDemo(FakeDemo):
            total_ticks = 3_000

            def __init__(self):
                super().__init__()
                self.calls = []

            def snapshots(self, datasets, *, ticks=None, **kwargs):
                self.calls.append((datasets, list(ticks or [])))
                assert datasets == ["player_ticks", "troopers"]
                all_ticks = list(ticks or [])
                return {
                    "player_ticks": pl.DataFrame(
                        {
                            "tick": all_ticks,
                            "hero_id": [1] * len(all_ticks),
                            "x": [0.0] * len(all_ticks),
                        }
                    ),
                    "troopers": frame(
                        "trooper_samples",
                        [
                            {
                                "tick": tick,
                                "trooper_type": "trooper",
                                "team_num": 2,
                                "lane": 1,
                                "health": 100,
                                "max_health": 100,
                                "x": 0.0,
                                "y": 0.0,
                                "z": 0.0,
                                "entity_id": tick,
                            }
                            for tick in all_ticks
                        ],
                    ),
                }

        demo = CombinedDemo()
        kills = frame("kills", [{"tick": 137}])
        fights = frame("teamfights", [{"fight_id": 1, "end_tick": 1_200}])
        players, troopers = _sample_frames(demo, kills, fights, sample_seconds=5)

        assert len(demo.calls) == 1
        assert 137 in players["tick"].to_list()
        assert troopers["tick"].to_list() == [1_200, 1_500, 1_800, 2_100, 2_400]
        assert 137 not in troopers["tick"].to_list()

    def test_combined_failure_falls_back_without_losing_either_frame(self):
        class FallbackDemo(FakeDemo):
            total_ticks = 3_000

            def snapshots(self, datasets, *, ticks=None, **kwargs):
                if isinstance(datasets, list):
                    raise RuntimeError("old boon")
                if datasets == "player_ticks":
                    return pl.DataFrame(
                        {"tick": ticks, "hero_id": [1] * len(ticks), "x": [0.0] * len(ticks)}
                    )
                return frame(
                    "trooper_samples",
                    [
                        {
                            "tick": tick,
                            "trooper_type": "trooper",
                            "team_num": 2,
                            "lane": 1,
                            "health": 100,
                            "max_health": 100,
                            "x": 0.0,
                            "y": 0.0,
                            "z": 0.0,
                            "entity_id": tick,
                        }
                        for tick in ticks or []
                    ],
                )

        demo = FallbackDemo()
        fights = frame("teamfights", [{"fight_id": 1, "end_tick": 1_200}])
        players, troopers = _sample_frames(demo, frame("kills", []), fights, 5)
        assert not players.is_empty()
        assert not troopers.is_empty()
