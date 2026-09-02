from __future__ import annotations

from deadlock_coach.events import (
    Event,
    build_timeline,
    format_timeline,
    kill_events,
    objective_events,
    patron_phase_events,
    rift_events,
)
from deadlock_coach.match import MatchData

from .conftest import MINUTE, TEAM_A, TEAM_B, frame


class TestKillEvents:
    def test_names_hero_and_player(self, match, names):
        first = kill_events(match, names)[0]
        assert "Infernus (a0)" in first.text
        assert "Abrams (b0)" in first.text

    def test_lists_assisters(self, match, names):
        first = kill_events(match, names)[0]
        assert "assist: Seven, Vindicta" in first.text
        assert first.detail["attackers"] == 3

    def test_self_kill_reads_as_uncredited_death(self, match, names):
        """attacker == victim is an environment death, not a kill for that hero."""
        suicide = [e for e in kill_events(match, names) if e.tick == 22 * MINUTE][0]
        assert "no killer credited" in suicide.text
        assert suicide.actors == ()
        assert suicide.team is None

    def test_hero_id_zero_is_a_trooper_not_a_hero_called_base(self, names):
        """The name table renders hero 0 as "Base"; crediting it would invent a
        13th player. Verified against build 10854, where early lane deaths to
        troopers all report attacker_hero_id 0."""
        md = MatchData(
            players=frame("players", [
                {"hero_id": 1, "team_num": TEAM_A, "player_name": "a"},
                {"hero_id": 11, "team_num": TEAM_B, "player_name": "b"},
            ]),
            kills=frame("kills", [
                {"tick": 100, "victim_hero_id": 1, "attacker_hero_id": 0,
                 "assister_hero_ids": [11]},
            ]),
        )
        event = kill_events(md, names)[0]
        assert "Base" not in event.text
        assert "non-hero damage" in event.text
        # The phantom killer is dropped, but the assisting hero is a real
        # participant — this is a tower dive — so they keep the credit.
        assert event.actors == (11,)
        assert event.team == TEAM_B
        assert "enemy nearby" in event.text

    def test_trooper_kill_with_no_assist_credits_nobody(self, names):
        md = MatchData(
            players=frame("players", [
                {"hero_id": 1, "team_num": TEAM_A, "player_name": "a"},
                {"hero_id": 11, "team_num": TEAM_B, "player_name": "b"},
            ]),
            kills=frame("kills", [
                {"tick": 100, "victim_hero_id": 1, "attacker_hero_id": 0,
                 "assister_hero_ids": []},
            ]),
        )
        event = kill_events(md, names)[0]
        assert event.actors == ()
        assert event.team is None
        assert "enemy nearby" not in event.text

    def test_credits_the_attacker_team(self, match, names):
        assert kill_events(match, names)[0].team == TEAM_A

    def test_sorted_by_tick(self, match, names):
        ticks = [e.tick for e in kill_events(match, names)]
        assert ticks == sorted(ticks)


class TestObjectiveEvents:
    def test_one_event_per_objective_despite_repeated_zero_rows(self, match, names):
        """The Guardian reports health 0 twice; that is one destruction."""
        walkers = [e for e in objective_events(match, names) if "Walker" in e.text]
        assert len(walkers) == 1
        assert walkers[0].tick == 15 * MINUTE

    def test_destroyer_is_the_other_team(self, match, names):
        walker = [e for e in objective_events(match, names) if "Walker" in e.text][0]
        assert walker.team == TEAM_A
        assert walker.text.startswith("Hidden King destroyed Archmother's Walker")

    def test_objectives_still_standing_produce_no_event(self, match, names):
        # The Patron never reaches zero health in the fixture.
        assert not [e for e in objective_events(match, names) if "Patron" in e.text]

    def test_lane_is_named_by_colour(self, match, names):
        walker = [e for e in objective_events(match, names) if "Walker" in e.text][0]
        assert "(yellow lane)" in walker.text

    def test_neutral_objectives_are_not_team_structures(self, names):
        """Mid Boss is owned by team 4 in build 10854. It is not anybody's
        structure falling, so it must not appear as a destruction event."""
        md = MatchData(
            players=frame("players", [
                {"hero_id": 1, "team_num": TEAM_A, "player_name": "a"},
                {"hero_id": 11, "team_num": TEAM_B, "player_name": "b"},
            ]),
            objectives=frame("objectives", [
                {"tick": 100, "objective_type": "mid_boss", "team_num": 4,
                 "lane": 16777215, "health": 0, "max_health": 19435, "entity_id": 9},
                {"tick": 200, "objective_type": "walker", "team_num": TEAM_B,
                 "lane": 6, "health": 0, "max_health": 12000, "entity_id": 10},
            ]),
        )
        events = objective_events(md, names)
        assert len(events) == 1
        assert "Walker" in events[0].text
        assert "(purple lane)" in events[0].text

    def test_base_objectives_omit_the_lane(self, names):
        """Patron and Shrines report lane 0; naming a lane for them is noise."""
        md = MatchData(
            players=frame("players", [
                {"hero_id": 1, "team_num": TEAM_A, "player_name": "a"},
                {"hero_id": 11, "team_num": TEAM_B, "player_name": "b"},
            ]),
            objectives=frame("objectives", [
                {"tick": 100, "objective_type": "patron", "team_num": TEAM_B,
                 "lane": 0, "health": 0, "max_health": 17000, "entity_id": 1},
            ]),
        )
        assert "lane" not in objective_events(md, names)[0].text

    def test_unknown_objective_type_passes_through(self, names):
        md = MatchData(
            players=frame("players", [
                {"hero_id": 1, "team_num": TEAM_A, "player_name": "a"},
                {"hero_id": 11, "team_num": TEAM_B, "player_name": "b"},
            ]),
            objectives=frame("objectives", [
                {"tick": 100, "objective_type": "new_thing_2027", "team_num": TEAM_B,
                 "lane": 1, "health": 0, "max_health": 10, "entity_id": 1},
            ]),
        )
        assert "new_thing_2027" in objective_events(md, names)[0].text


class TestPatronPhase:
    def test_emits_only_on_change_away_from_normal(self, match, names):
        events = patron_phase_events(match, names)
        assert len(events) == 1
        assert events[0].tick == 24 * MINUTE + 600
        assert "final" in events[0].text


class TestRiftEvents:
    def test_capture_and_expiry_are_distinct(self, names):
        md = MatchData(
            players=frame("players", [
                {"hero_id": 1, "team_num": TEAM_A, "player_name": "a"},
                {"hero_id": 11, "team_num": TEAM_B, "player_name": "b"},
            ]),
            rift=frame("rift", [
                {"rift_num": 1, "active_tick": 100, "capture_tick": 200, "winning_team": TEAM_A,
                 "lane": 1},
                {"rift_num": 2, "active_tick": 300, "expire_tick": 400, "lane": 2},
            ]),
        )
        texts = [e.text for e in rift_events(md, names)]
        assert any("captured by Hidden King" in t for t in texts)
        assert any("expired uncaptured" in t for t in texts)
        # A captured rift must not also report as expired.
        assert sum("Rift #1" in t for t in texts) == 2  # active + capture


class TestTimeline:
    def test_is_sorted_and_stable(self, match, names):
        timeline = build_timeline(match, names)
        assert [e.sort_key() for e in timeline] == sorted(e.sort_key() for e in timeline)
        assert build_timeline(match, names) == timeline

    def test_objectives_precede_kills_on_the_same_tick(self, names):
        kill = Event(tick=100, kind="kill", text="k")
        objective = Event(tick=100, kind="objective", text="o")
        assert sorted([kill, objective], key=Event.sort_key) == [objective, kill]

    def test_contains_every_source(self, match, names):
        kinds = {e.kind for e in build_timeline(match, names)}
        assert {"kill", "objective", "patron_phase"} <= kinds

    def test_empty_match_yields_empty_timeline(self, empty_match, names):
        assert build_timeline(empty_match, names) == []


def test_format_timeline_stamps_clock_time(match, names):
    lines = format_timeline(match, build_timeline(match, names))
    assert lines[0].startswith("[05:00]")
