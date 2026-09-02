from __future__ import annotations

import json

from deadlock_coach.events import Event
from deadlock_coach.render import _trim, render_json, render_report

from .conftest import MINUTE


class TestReport:
    def test_has_every_section(self, match, names):
        report = render_report(match, names)
        for heading in (
            "## Match",
            "## Roster and final line",
            "## Phases",
            "## Soul economy",
            "## Teamfights",
            "## Player-perspective opportunities",
            "## Per-player review",
            "## Kill patterns",
            "## Event timeline",
        ):
            assert heading in report

    def test_uses_names_not_raw_ids(self, match, names):
        report = render_report(match, names)
        assert "Infernus" in report
        assert "Hidden King" in report
        assert "hero_id" not in report

    def test_states_unknowns_explicitly(self, match, names):
        """A model must be able to see the gap rather than infer past it."""
        stripped = match.with_frames(
            player_samples=match.player_samples.clear()
        )
        assert "unknown" in render_report(stripped, names)

    def test_survives_an_empty_match(self, empty_match, names):
        report = render_report(empty_match, names)
        assert "## Match" in report
        assert report.endswith("\n")

    def test_timeline_is_clock_stamped(self, match, names):
        assert "[05:00]" in render_report(match, names)

    def test_fight_conversion_rate_is_reported(self, match, names):
        report = render_report(match, names)
        assert "converted into an objective" in report
        assert "Available next move" in report
        assert "trooper samples unavailable" in report

    def test_standoffs_are_counted_but_not_tabled(self, match, names):
        """Most detected clusters trade no kills. Listing them all buries the
        match in lane poke, so they are summarised as a count instead."""
        from .conftest import frame
        noisy = list(match.teamfights.iter_rows(named=True))
        noisy.append({"fight_id": 99, "start_tick": 1, "end_tick": 2,
                      "duration_seconds": 1.0, "participants": [1, 11],
                      "num_participants": 2, "hero_damage": 50, "kills": 0})
        report = render_report(match.with_frames(teamfights=frame("teamfights", noisy)), names)
        assert "omitted from the table below" in report
        assert "| 99 |" not in report

    def test_every_phase_appears_for_every_player(self, match, names):
        """The whole match must be visible per player, not just the totals."""
        report = render_report(match, names)
        section = report.split("## Per-player review")[1].split("## Kill patterns")[0]
        for phase in ("laning", "mid game", "late game"):
            assert section.count(f"**{phase}**") == 6, phase

    def test_fight_record_table_covers_every_phase(self, match, names):
        report = render_report(match, names)
        phases_block = report.split("## Phases")[1].split("## Soul economy")[0]
        assert "Decisive fights won, by phase" in phases_block
        assert "laning (" in phases_block and "late game (" in phases_block

    def test_a_clean_sheet_is_stated_not_left_blank(self, match, names):
        """A player who never died must not render as missing data."""
        from .conftest import frame
        no_deaths = [r for r in match.kills.iter_rows(named=True)
                     if r["victim_hero_id"] != 3]
        report = render_report(match.with_frames(kills=frame("kills", no_deaths)), names)
        assert "Did not die once all match." in report


class TestOpportunitySection:
    def test_every_window_type_is_summarised_per_player(self, match, names):
        report = render_report(match, names)
        section = report.split("## Player-perspective opportunities")[1]
        section = section.split("## Per-player review")[0]
        for column in (
            "Kill-pressure signals",
            "Cross-lane rotations",
            "Macro windows",
            "Actionable invade / scout-only",
        ):
            assert column in section, column

    def test_every_window_is_listed_even_when_ranked(self, match, names):
        """Capping the tables silently deleted whole stretches of the match: a
        player whose clearest reads were in lane got a list that stopped at
        10:00 and said nothing about the mid game they asked about."""
        from deadlock_coach.opportunities import JungleOpportunity, KillOpportunity
        from deadlock_coach.render import _opportunity_rows

        def kill(tick: int) -> KillOpportunity:
            return KillOpportunity(
                observer_hero_id=1, target_hero_id=11, start_tick=tick, end_tick=tick,
                phase="laning", confidence="high", score=12, observation_basis="direct combat",
                target_health_pct=0.2, observer_health_pct=0.9, distance=800.0,
                recent_damage_by_observer=100, known_allies_nearby=1, known_enemies_nearby=1,
                unknown_enemy_count=0, evidence=(), limitations=(),
            )

        def camp(tick: int) -> JungleOpportunity:
            return JungleOpportunity(
                observer_hero_id=1, camp_id=1, lane=1, camp_x=0.0, camp_y=0.0,
                enemy_team_num=3, start_tick=tick, end_tick=tick, confidence="low",
                camp_status="visible and available", distance=1000.0,
                enemies_last_seen_away=3, nearest_known_enemy=9000.0,
                unknown_enemy_count=0, evidence=(), limitations=(),
            )

        kills = [kill(t * MINUTE) for t in range(1, 9)]
        rows = _opportunity_rows(match, names, kills, [], [], [camp(20 * MINUTE)])
        assert len(rows) == len(kills) + 1, "nothing may be dropped"
        assert any(r.start_tick == 20 * MINUTE for r in rows)
        assert rows == sorted(rows, key=lambda r: (-r.priority, r.start_tick, r.action))

    def test_higher_confidence_window_ranks_before_earlier_medium_window(self, match, names):
        from deadlock_coach.opportunities import JungleOpportunity, KillOpportunity
        from deadlock_coach.render import _opportunity_rows

        kill = KillOpportunity(
            observer_hero_id=1, target_hero_id=11, start_tick=9 * MINUTE,
            end_tick=9 * MINUTE, phase="laning", confidence="high", score=12,
            observation_basis="direct combat", target_health_pct=0.18,
            observer_health_pct=0.9, distance=800.0, recent_damage_by_observer=100,
            known_allies_nearby=1, known_enemies_nearby=1, unknown_enemy_count=0,
            evidence=(), limitations=(),
        )
        camp = JungleOpportunity(
            observer_hero_id=1, camp_id=7, lane=1, camp_x=0.0, camp_y=0.0,
            enemy_team_num=3, start_tick=2 * MINUTE, end_tick=2 * MINUTE,
            confidence="medium", camp_status="visible and available", distance=1500.0,
            enemies_last_seen_away=3, nearest_known_enemy=9000.0, unknown_enemy_count=0,
            evidence=(), limitations=(),
        )
        rows = _opportunity_rows(match, names, [kill], [], [], [camp])
        assert [r.start_tick for r in rows] == [9 * MINUTE, 2 * MINUTE]
        assert "finish Abrams (18% hp" in rows[0].action
        assert "take enemy camp #7" in rows[1].action

    def test_focus_keeps_only_the_selected_players_details(self, match, names):
        report = render_report(match, names, focus_hero_ids=[1])
        assert "## Report focus" in report
        assert "### Infernus" in report
        assert "### Seven" not in report
        payload = json.loads(render_json(match, names, focus_hero_ids=[1]))
        assert [player["hero_id"] for player in payload["players"]] == [1]
        for kind in ("kill_windows", "jungle_windows", "rotation_windows", "macro_windows"):
            assert all(
                window["observer_hero_id"] == 1
                for window in payload["opportunities"][kind]
            )

    def test_standing_caveats_are_stated_once_not_per_row(self, match, names):
        """They were repeated on every window until the rows were unreadable."""
        report = render_report(match, names)
        section = report.split("## Player-perspective opportunities")[1]
        section = section.split("## Per-player review")[0]
        boilerplate = "aim, opponent reactions, ammo, and full damage simulation are not modeled"
        assert section.count(boilerplate) <= 1

    def test_macro_targets_are_named_not_left_as_raw_types(self, match, names):
        """Structured target fields, not pre-baked strings — so the name tables
        stay the single source of truth for what a structure is called."""
        from deadlock_coach.opportunities import MacroOpportunity
        from deadlock_coach.render import _macro_target

        window = MacroOpportunity(
            observer_hero_id=1, kind="siege", action="push the objective",
            start_tick=0, end_tick=60, phase="mid game", confidence="medium",
            score=8, target_kind="walker", target_lane=1, target_team_num=3, alternatives=(),
            enemies_known_dead=2, window_seconds=20.0, team_objective_damage=0,
            observer_objective_damage=0, distance=4000.0, evidence=(), limitations=(),
        )
        assert _macro_target(names, window) == "yellow-lane Walker (Archmother)"

    def test_positionless_targets_render_by_name(self, match, names):
        from deadlock_coach.opportunities import MacroOpportunity
        from deadlock_coach.render import _macro_target

        window = MacroOpportunity(
            observer_hero_id=1, kind="mid_boss", action="take the Mid Boss",
            start_tick=0, end_tick=60, phase="mid game", confidence="high",
            score=10, target_kind="mid_boss", target_lane=None, target_team_num=None,
            alternatives=("urn", "walker"),
            enemies_known_dead=3, window_seconds=25.0, team_objective_damage=0,
            observer_objective_damage=0, distance=None, evidence=(), limitations=(),
        )
        assert _macro_target(names, window) == "Mid Boss — also on: Urn, Walker"


class TestTrim:
    def test_keeps_everything_under_the_limit(self):
        events = [Event(tick=i, kind="kill", text=str(i)) for i in range(10)]
        kept, dropped = _trim(events, 400)
        assert kept == events and dropped == 0

    def test_drops_low_value_kinds_first(self):
        events = [Event(tick=i, kind="urn", text="u") for i in range(50)]
        events += [Event(tick=i, kind="kill", text="k") for i in range(50, 60)]
        kept, dropped = _trim(events, 20)
        assert dropped == 50
        assert all(e.kind == "kill" for e in kept)

    def test_always_keeps_objectives_when_over_budget(self):
        events = [Event(tick=i, kind="kill", text="k") for i in range(100)]
        events += [Event(tick=i, kind="objective", text="o") for i in range(100, 110)]
        kept, _ = _trim(events, 20)
        assert sum(e.kind == "objective" for e in kept) == 10
        assert len(kept) == 20

    def test_trimming_samples_the_whole_match_not_just_the_end(self):
        """An earlier version kept the most recent N events, which deleted the
        entire early game from any bloody match. Coverage must stay even."""
        events = [Event(tick=i, kind="kill", text=str(i)) for i in range(1000)]
        kept, dropped = _trim(events, 50)
        assert dropped == 950
        assert kept[0].tick == 0, "earliest event must survive"
        assert kept[-1].tick == 999, "latest event must survive"
        early = sum(1 for e in kept if e.tick < 333)
        late = sum(1 for e in kept if e.tick >= 666)
        assert abs(early - late) <= 1, (early, late)

    def test_trimming_keeps_early_objectives(self):
        events = [Event(tick=1, kind="objective", text="early tower")]
        events += [Event(tick=i, kind="kill", text="k") for i in range(10, 1000)]
        kept, _ = _trim(events, 20)
        assert any(e.kind == "objective" and e.tick == 1 for e in kept)

    def test_trimmed_output_stays_in_tick_order(self):
        events = [Event(tick=i, kind="kill", text="k") for i in range(100)]
        events += [Event(tick=5, kind="objective", text="o")]
        kept, _ = _trim(events, 10)
        assert [e.tick for e in kept] == sorted(e.tick for e in kept)

    def test_report_notes_when_it_trimmed(self, match, names):
        report = render_report(match, names, max_timeline_events=2)
        assert "trimmed to fit" in report


class TestJson:
    def test_is_valid_json_with_the_expected_keys(self, match, names):
        payload = json.loads(render_json(match, names))
        assert set(payload) == {
            "summary", "match", "phases", "phase_stats", "fight_record_by_phase",
            "players", "fights", "kills", "economy", "advantage", "opportunities",
            "physics", "mechanics", "timeline",
        }
        # Unfocused reports have no "you" to summarise.
        assert payload["summary"] is None
        assert "conversion_assessment" in payload["fights"][0]

    def test_timeline_entries_carry_clock_and_tick(self, match, names):
        payload = json.loads(render_json(match, names))
        first = payload["timeline"][0]
        assert first["tick"] == 5 * MINUTE
        assert first["clock"] == "05:00"

    def test_survives_an_empty_match(self, empty_match, names):
        payload = json.loads(render_json(empty_match, names))
        assert payload["players"] == []
        assert payload["timeline"] == []

    def test_every_window_type_reaches_the_sidecar(self, match, names):
        payload = json.loads(render_json(match, names))
        assert set(payload["opportunities"]) == {
            "vision_model", "warning",
            "kill_windows", "jungle_windows", "rotation_windows", "macro_windows",
        }

    def test_teams_are_resolved_to_names(self, match, names):
        payload = json.loads(render_json(match, names))
        assert "Hidden King" in payload["match"]["teams"].values()
