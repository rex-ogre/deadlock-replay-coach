from __future__ import annotations

import json

import pytest

import deadlock_coach.cli as cli
from deadlock_coach.cli import _resolve_player, build_parser


def test_player_option_is_parsed():
    args = build_parser().parse_args(["match.dem", "--player", "Yamato"])
    assert args.player == "Yamato"


def test_player_selector_accepts_hero_or_player_name(match, names):
    assert _resolve_player(match, names, "Infernus") == 1
    assert _resolve_player(match, names, "a0") == 1
    assert _resolve_player(match, names, "1") == 1


def test_player_selector_reports_available_names(match, names):
    with pytest.raises(ValueError, match="available:.*Infernus"):
        _resolve_player(match, names, "nobody")


def test_cli_builds_the_advantage_ledger_once_for_both_outputs(
    tmp_path, monkeypatch, match, names
):
    demo = tmp_path / "match.dem"
    demo.touch()
    out = tmp_path / "out"
    ledger = object()
    opportunities = object()
    received = []

    monkeypatch.setattr("deadlock_coach.source.load_demo", lambda *args, **kwargs: match)
    monkeypatch.setattr(cli.Names, "from_boon", classmethod(lambda cls: names))
    monkeypatch.setattr(
        cli, "analyze_opportunities", lambda md, **kwargs: opportunities
    )
    monkeypatch.setattr(cli, "analyze_advantage", lambda *args, **kwargs: ledger)

    def fake_report(*args, **kwargs):
        received.append((kwargs["opportunity_analysis"], kwargs["advantage_ledger"]))
        return "report\n"

    def fake_json(*args, **kwargs):
        received.append((kwargs["opportunity_analysis"], kwargs["advantage_ledger"]))
        return "{}"

    monkeypatch.setattr(cli, "render_report", fake_report)
    monkeypatch.setattr(cli, "render_json", fake_json)
    # The stub opportunities object above is not a real analysis, so the summary
    # pass -- which reads windows off it -- is stubbed with it.
    monkeypatch.setattr(cli, "build_summaries", lambda *args, **kwargs: {})

    assert cli.main([str(demo), "--out", str(out), "--quiet", "--offline"]) == 0
    assert received == [(opportunities, ledger), (opportunities, ledger)]
    assert (out / "match.report.md").read_text() == "report\n"
    assert (out / "match.match.json").read_text() == "{}"
    assert json.loads((out / "match.viewer.json").read_text())["schema_version"] == 2


def test_cli_prefers_replay_accuracy_without_calling_match_api(
    tmp_path, monkeypatch, match, names
):
    demo = tmp_path / "match.dem"
    demo.touch()
    out = tmp_path / "out"
    replay_metadata = {
        "match_info": {
            "players": [{
                "account_id": 100,
                "hero_id": 1,
                "stats": [{
                    "time_stamp_s": 1500,
                    "shots_hit": 750,
                    "shots_missed": 250,
                    "hero_bullets_hit": 200,
                    "hero_bullets_hit_crit": 20,
                }],
            }],
        },
        "stats_source": "replay-post-match-details",
    }
    seen = {}

    monkeypatch.setattr("deadlock_coach.source.load_demo", lambda *args, **kwargs: match)
    monkeypatch.setattr(cli.Names, "from_boon", classmethod(lambda cls: names))
    monkeypatch.setattr(cli, "read_replay_metadata", lambda path: replay_metadata)
    monkeypatch.setattr(
        cli,
        "fetch_match_stats",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("API fallback called")),
    )

    def report(*args, **kwargs):
        seen.update(kwargs["skill_stats"])
        return "report\n"

    monkeypatch.setattr(cli, "render_report", report)
    monkeypatch.setattr(cli, "render_json", lambda *args, **kwargs: "{}")
    monkeypatch.setattr(cli, "build_summaries", lambda *args, **kwargs: {})

    assert cli.main([str(demo), "--out", str(out), "--quiet", "--offline"]) == 0
    assert seen[1].accuracy == pytest.approx(0.75)


def test_cli_writes_a_summary_for_every_seat(tmp_path, monkeypatch, match, names):
    """The web flow picks a hero after decoding, so every seat is summarised."""
    demo = tmp_path / "match.dem"
    demo.touch()
    out = tmp_path / "out"
    monkeypatch.setattr("deadlock_coach.source.load_demo", lambda *args, **kwargs: match)
    monkeypatch.setattr(cli.Names, "from_boon", classmethod(lambda cls: names))

    assert cli.main([str(demo), "--out", str(out), "--quiet", "--offline"]) == 0
    payload = json.loads((out / "match.summary.json").read_text())
    assert [row["hero_id"] for row in payload["heroes"]] == match.hero_ids
    assert all(row["markdown"].startswith("## Bottom line") for row in payload["heroes"])


def test_focused_cli_summarises_only_that_seat(tmp_path, monkeypatch, match, names):
    demo = tmp_path / "match.dem"
    demo.touch()
    out = tmp_path / "out"
    monkeypatch.setattr("deadlock_coach.source.load_demo", lambda *args, **kwargs: match)
    monkeypatch.setattr(cli.Names, "from_boon", classmethod(lambda cls: names))

    assert cli.main([str(demo), "--out", str(out), "--quiet", "--offline", "--player", "Seven"]) == 0
    payload = json.loads((out / "match.summary.json").read_text())
    assert [row["hero"] for row in payload["heroes"]] == ["Seven"]
