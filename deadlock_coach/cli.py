"""``deadlock-coach match.dem -o out/`` — one demo in, an AI-ready briefing out."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from time import perf_counter

from .advantage import analyze_advantage
from .gamedata import load_constants
from .names import Names
from .opportunities import analyze_opportunities
from .render import DEFAULT_MAX_TIMELINE_EVENTS, render_json, render_report
from .replay_stats import read_replay_metadata
from .skillstats import by_hero, fetch_match_stats, stats_from_metadata
from .source import DEFAULT_SAMPLE_SECONDS
from .summary import build_summaries, render_summary_section
from .viewer import load_visual_assets, render_viewer_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deadlock-coach",
        description="Decode a Deadlock .dem replay into a Markdown coaching report "
        "and a JSON sidecar suitable for feeding to an LLM.",
    )
    parser.add_argument("demo", type=Path, help="path to the .dem replay")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("out"),
        help="output directory (default: ./out)",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=DEFAULT_SAMPLE_SECONDS,
        help="positional sampling interval; kill ticks are always sampled exactly "
        f"(default: {DEFAULT_SAMPLE_SECONDS:g})",
    )
    parser.add_argument(
        "--min-fight-players",
        type=int,
        default=4,
        help="distinct heroes required for a damage cluster to count as a teamfight "
        "rather than a lane trade (default: 4)",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=DEFAULT_MAX_TIMELINE_EVENTS,
        help=f"cap on timeline lines in the report (default: {DEFAULT_MAX_TIMELINE_EVENTS})",
    )
    parser.add_argument(
        "--dump-frames",
        action="store_true",
        help="also write the underlying Polars frames as parquet, for tool-call use",
    )
    parser.add_argument(
        "--player",
        help="focus the Markdown and JSON on one hero or player name (for example: Yamato)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "never call the Deadlock API; use bundled movement constants and the replay's "
            "embedded accuracy stats. Makes a run reproducible and airgap-safe."
        ),
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress progress logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    pipeline_started = perf_counter()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    log = logging.getLogger("deadlock-coach")

    if not args.demo.exists():
        print(f"error: no such file: {args.demo}", file=sys.stderr)
        return 2

    from .source import load_demo

    log.info("parsing %s", args.demo)
    md = load_demo(
        args.demo,
        sample_seconds=args.sample_seconds,
        min_fight_players=args.min_fight_players,
    )
    decoded_at = perf_counter()
    names = Names.from_boon()
    focus_hero_ids: tuple[int, ...] | None = None
    if args.player:
        try:
            focus_hero_ids = (_resolve_player(md, names, args.player),)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    # Accuracy comes from the completed replay itself. The network lookup is
    # only a compatibility fallback for a recording that lacks its final
    # PostMatchDetails packet; --offline therefore still retains accuracy.
    constants = load_constants(offline=args.offline)
    if constants.is_stale and not args.offline:
        log.warning("using bundled movement constants; the assets API was unreachable")
    account_stats = stats_from_metadata(read_replay_metadata(args.demo), constants)
    if not account_stats:
        account_stats = fetch_match_stats(md.match_id, constants, offline=args.offline)
    skill_stats = by_hero(
        account_stats,
        {
            int(row["hero_id"]): row["steam_id"]
            for row in md.players.iter_rows(named=True)
            if row.get("hero_id") is not None
        },
    )
    if not skill_stats:
        log.info("replay has no post-match record; accuracy will be omitted")
    visual_assets = load_visual_assets(offline=args.offline)
    external_data_at = perf_counter()

    opportunities = analyze_opportunities(md, constants=constants)
    # Built once and handed to both renderers: the ledger walks every neutral
    # health row and every positional sample, which is not free on a real demo.
    ledger = analyze_advantage(md, names, focus_hero_ids=focus_hero_ids)
    analyzed_at = perf_counter()

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.demo.stem

    report_path = args.out / f"{stem}.report.md"
    report_path.write_text(
        render_report(
            md,
            names,
            max_timeline_events=args.max_events,
            opportunity_analysis=opportunities,
            focus_hero_ids=focus_hero_ids,
            advantage_ledger=ledger,
            constants=constants,
            skill_stats=skill_stats,
        ),
        encoding="utf-8",
    )
    log.info("wrote %s", report_path)

    json_path = args.out / f"{stem}.match.json"
    json_path.write_text(
        render_json(
            md,
            names,
            opportunity_analysis=opportunities,
            focus_hero_ids=focus_hero_ids,
            advantage_ledger=ledger,
            constants=constants,
            skill_stats=skill_stats,
        ),
        encoding="utf-8",
    )
    log.info("wrote %s", json_path)

    # One per seat, not one for the focused player: the web flow decodes a
    # replay before the user has said which hero they played, and re-running
    # the pipeline once they do would cost minutes to save milliseconds.
    summaries = build_summaries(
        md,
        names,
        focus_hero_ids or md.hero_ids,
        opportunities=opportunities,
        constants=constants,
        skill_stats=skill_stats,
    )
    summary_path = args.out / f"{stem}.summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "match_id": md.match_id,
                "heroes": [
                    {**summary.to_dict(), "markdown": render_summary_section(summary)}
                    for summary in summaries.values()
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log.info("wrote %s", summary_path)

    viewer_path = args.out / f"{stem}.viewer.json"
    viewer_path.write_text(
        render_viewer_json(md, names, assets=visual_assets, offline=args.offline),
        encoding="utf-8",
    )
    log.info("wrote %s", viewer_path)
    rendered_at = perf_counter()

    if args.dump_frames:
        frames_dir = args.out / f"{stem}.frames"
        frames_dir.mkdir(exist_ok=True)
        for name in (
            "players",
            "kills",
            "damage",
            "objectives",
            "teamfights",
            "trooper_samples",
            "player_samples",
            "neutrals",
            # Macro windows are built from these, so a frames dump that omits
            # them cannot reproduce the report it came with.
            "mid_boss",
            "urn",
            "rift",
            "item_purchases",
            "ability_upgrades",
            "ability_ticks",
            "ability_uses",
        ):
            frame = getattr(md, name)
            frame.write_parquet(frames_dir / f"{name}.parquet")
        log.info("wrote frames to %s", frames_dir)

    log.info(
        "pipeline timing: decode %.1fs, APIs %.1fs, analysis %.1fs, render %.1fs, total %.1fs",
        decoded_at - pipeline_started,
        external_data_at - decoded_at,
        analyzed_at - external_data_at,
        rendered_at - analyzed_at,
        perf_counter() - pipeline_started,
    )

    print(report_path)
    return 0


def _resolve_player(md, names: Names, query: str) -> int:
    """Resolve an exact hero name, player name, or numeric hero id."""
    wanted = query.strip().casefold()
    matches: list[int] = []
    for hero in md.hero_ids:
        labels = {str(hero).casefold(), names.hero(hero).casefold()}
        player = md.player_name(hero)
        if player:
            labels.add(player.casefold())
        if wanted in labels:
            matches.append(hero)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        labels = ", ".join(f"{names.hero(hero)} ({md.player_name(hero) or '—'})" for hero in matches)
        raise ValueError(f"player selector {query!r} is ambiguous: {labels}")
    available = ", ".join(
        f"{names.hero(hero)} ({md.player_name(hero) or '—'})" for hero in md.hero_ids
    )
    raise ValueError(f"no hero or player named {query!r}; available: {available}")


if __name__ == "__main__":
    raise SystemExit(main())
