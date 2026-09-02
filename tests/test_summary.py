"""Tests for the ranked front page.

The summary is the one section a player is guaranteed to read, so the rules it
must not break are all about honesty rather than formatting: no percentage
without a counted denominator, no population comparison without a population,
and no claim whose working is missing from the rest of the report.
"""

from __future__ import annotations

import json

import pytest

from deadlock_coach import gamedata, summary as summary_mod
from deadlock_coach.benchmark import benchmark_player
from deadlock_coach.opportunities import analyze_opportunities
from deadlock_coach.render import render_json, render_report
from deadlock_coach.skillstats import PlayerSkillStats
from deadlock_coach.tactics import (
    analyze_fights,
    economy_curve,
    kill_contexts,
    phases,
    player_reports,
)

from .conftest import A_HEROES


@pytest.fixture
def constants():
    return gamedata.load_constants(offline=True)


@pytest.fixture
def stat():
    """A post-match line for hero 1, deliberately below the population."""
    return PlayerSkillStats(
        account_id=999,
        hero_id=1,
        shots_hit=400,
        shots_missed=600,
        hero_bullets_hit=80,
        hero_bullets_hit_crit=8,
        creep_kills=90,
        possible_creeps=200,
        badge=62,
        kills=4,
        deaths=9,
        assists=6,
        net_worth=30_000,
        last_hits=150,
        denies=0,
        player_damage=25_000,
        player_damage_taken=45_000,
        boss_damage=2_000,
        creep_damage=40_000,
        neutral_damage=5_000,
        duration_seconds=1800,
    )


def build(match, names, constants, skill_stats=None, hero_id=1):
    return summary_mod.build_summary(
        match,
        names,
        hero_id=hero_id,
        fights=analyze_fights(match, names),
        reports=player_reports(match),
        kills=kill_contexts(match, constants),
        phase_list=phases(match),
        curve=economy_curve(match),
        opportunities=analyze_opportunities(match, constants=constants),
        constants=constants,
        skill_stats=skill_stats or {},
    )


class TestBenchmark:
    def test_compares_against_the_players_own_hero_at_two_ranks(self, constants, stat):
        result = benchmark_player(stat, constants)
        assert result.available
        assert result.peer is not None and result.top is not None
        # The top row must be a higher rank than the peer row, or it is not a
        # comparison against better players.
        assert result.top.badge > result.peer.badge
        accuracy = result.by_key("accuracy")
        assert accuracy.player == pytest.approx(0.4)
        assert accuracy.verdict == "behind"

    def test_style_metrics_are_never_scored(self, constants, stat):
        result = benchmark_player(stat, constants)
        assert result.by_key("hero_damage_per_1k_souls").verdict == "style"
        assert all(m.key != "hero_damage_per_1k_souls" for m in result.deficits())

    def test_unknown_hero_yields_no_comparison(self, constants, stat):
        from dataclasses import replace

        result = benchmark_player(replace(stat, hero_id=9_999), constants)
        assert not result.available
        assert result.peer is None and result.top is None

    def test_ratios_do_not_move_when_the_match_runs_longer(self, constants, stat):
        """Match length must cancel, or ranks are not comparable at all."""
        from dataclasses import replace

        longer = replace(
            stat,
            deaths=stat.deaths * 2,
            net_worth=stat.net_worth * 2,
            kills=stat.kills * 2,
            assists=stat.assists * 2,
        )
        before = benchmark_player(stat, constants).by_key("deaths_per_10k_souls")
        after = benchmark_player(longer, constants).by_key("deaths_per_10k_souls")
        assert before.player == pytest.approx(after.player)


class TestSummary:
    def test_ranks_macro_above_mechanics(self, match, names, constants, stat):
        result = build(match, names, constants, {1: stat})
        keys = [f.key for f in result.findings]
        assert keys, "a losing player with 9 deaths should produce findings"
        macro = {"fight_conversion", "macro_windows", "rotation_windows", "pickoffs"}
        mechanical = {m.key for m in result.benchmark.metrics}
        first_mechanic = next((i for i, k in enumerate(keys) if k in mechanical), len(keys))
        last_macro = max((i for i, k in enumerate(keys) if k in macro), default=-1)
        assert last_macro < first_mechanic

    def test_every_rate_carries_its_denominator(self, match, names, constants, stat):
        result = build(match, names, constants, {1: stat})
        assert result.rates
        for rate in result.rates:
            assert rate.denominator > 0
            assert f"/{rate.denominator:,.0f}" in rate.text()

    def test_accuracy_is_stated_as_a_percentage(self, match, names, constants, stat):
        result = build(match, names, constants, {1: stat})
        accuracy = next(r for r in result.rates if r.key == "accuracy")
        assert accuracy.value == pytest.approx(0.4)
        assert "40%" in accuracy.text()

    def test_missing_post_match_record_reads_as_unknown(self, match, names, constants):
        result = build(match, names, constants, {})
        assert result.benchmark is None
        assert result.unavailable
        assert "missing data, not a low score" in result.unavailable[0]
        # And no fabricated mechanical findings.
        assert not result.mechanics

    def test_findings_name_the_section_that_holds_the_working(
        self, match, names, constants, stat
    ):
        report = render_report(
            match, names, focus_hero_ids=[1], skill_stats={1: stat}, constants=constants
        )
        result = build(match, names, constants, {1: stat})
        for finding in result.findings:
            assert f"## {finding.section}" in report

    def test_serialises_for_the_sidecar(self, match, names, constants, stat):
        result = build(match, names, constants, {1: stat})
        payload = json.loads(json.dumps(result.to_dict()))
        assert payload["hero"] == "Infernus"
        assert payload["benchmark"]["top"]["matches"] > 0
        assert payload["rates"][0]["value"] is not None


class TestReportIntegration:
    def test_focused_report_leads_with_the_summary(self, match, names, constants, stat):
        report = render_report(
            match, names, focus_hero_ids=[1], skill_stats={1: stat}, constants=constants
        )
        assert report.index("## Bottom line") < report.index("## Match")

    def test_unfocused_report_has_no_summary(self, match, names):
        assert "## Bottom line" not in render_report(match, names)

    def test_sidecar_carries_the_summary_for_a_focused_player(
        self, match, names, constants, stat
    ):
        payload = json.loads(
            render_json(
                match, names, focus_hero_ids=[1], skill_stats={1: stat}, constants=constants
            )
        )
        assert payload["summary"]["hero_id"] == 1
        assert payload["summary"]["shape"]["team"] == "Hidden King"

    def test_summary_hero_is_the_focused_one(self, match, names, constants):
        result = build(match, names, constants, hero_id=A_HEROES[1])
        assert result.hero_name == "Seven"
        assert result.shape.team_name == "Hidden King"
