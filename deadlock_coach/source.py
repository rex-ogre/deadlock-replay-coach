"""The only module that knows ``boon`` exists.

Loading a demo is the one step that can fail in a dozen game-specific ways: a
dataset missing on an old build, a mode that has no Urn, a truncated file. All
of that is contained here, so the analysis layer only ever sees well-formed
frames.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import polars as pl

from .match import SCHEMAS, Clock, MatchData, empty

log = logging.getLogger(__name__)

# Opportunity windows often last only a few seconds. One sample per second is
# still compact (~30k rows for a 40-minute 6v6), while a 15-second stride can
# skip an entire chase or invade decision. Kill ticks remain exact.
DEFAULT_SAMPLE_SECONDS = 1.0

# boon's default of 3 counts a 2v1 lane poke as a "teamfight" — on match
# 12345678 that produced 141 fights, 84 of them with no kills at all. Four
# distinct heroes is the smallest cluster that is meaningfully a group fight
# rather than a lane trade.
DEFAULT_MIN_FIGHT_PLAYERS = 4

# A fight winner normally has only a short respawn window. Sampling the wave at
# the fight end and every five seconds afterwards is enough to tell "push now"
# from "the lane will not arrive", while keeping a multi-million-row dataset to
# a few thousand rows.
TROOPER_SAMPLE_SECONDS = 5.0
TROOPER_LOOKAHEAD_SECONDS = 20.0

_SAMPLE_COLUMNS = list(SCHEMAS["player_samples"].keys())

# Accessing a lazy boon dataset property parses the replay when that dataset has
# not been loaded yet.  Asking for these together is materially different from
# touching the properties one by one: boon can collect all of them during one
# shared pass over a 400--600 MB demo.
_EVENT_DATASETS = (
    "kills",
    "damage",
    "objectives",
    "mid_boss",
    "urn",
    "rift",
    "item_purchases",
    "ability_upgrades",
    "ability_ticks",
    "abilities",
    "neutrals",
    # Clock conversion needs pause state.  Loading it here prevents the first
    # render-time tick_to_seconds call from starting another full parse.
    "world_ticks",
)


def _preload_event_datasets(demo) -> None:
    """Load all event frames in one boon parse, with old-version fallback."""
    loader = getattr(demo, "load", None)
    if not callable(loader):
        return
    try:
        loader(*_EVENT_DATASETS)
    except Exception as exc:
        # Individual property access below retains the adapter's original
        # graceful-degradation behaviour if a boon/game build rejects a batch.
        log.warning(
            "batch dataset load failed; falling back to individual loads (%s: %s)",
            type(exc).__name__,
            exc,
        )


def _frame(demo, name: str, schema_name: str | None = None) -> pl.DataFrame:
    """Pull one dataset, degrading to an empty frame rather than failing.

    A missing Urn frame should cost you the Urn section of the report, not the
    whole report.
    """
    schema_name = schema_name or name
    try:
        df = getattr(demo, name)
    except Exception as exc:
        log.warning("dataset %r unavailable (%s: %s)", name, type(exc).__name__, exc)
        return empty(schema_name)
    if df is None:
        return empty(schema_name)
    return df


def _sample_ticks(demo, kills: pl.DataFrame, sample_seconds: float) -> list[int]:
    """Stride ticks plus every kill tick, deduplicated and sorted."""
    tick_rate = int(demo.tick_rate or 60)
    total = int(demo.total_ticks or 0)
    stride = max(1, int(sample_seconds * tick_rate))
    ticks = set(range(0, total + 1, stride)) if total else set()
    if not kills.is_empty() and "tick" in kills.columns:
        ticks.update(int(t) for t in kills["tick"].drop_nulls().to_list())
    return sorted(t for t in ticks if t >= 0)


def _player_samples(demo, kills: pl.DataFrame, sample_seconds: float) -> pl.DataFrame:
    ticks = _sample_ticks(demo, kills, sample_seconds)
    if not ticks:
        return empty("player_samples")
    try:
        df = demo.snapshots("player_ticks", ticks=ticks)
    except Exception as exc:
        log.warning("could not sample player_ticks (%s: %s)", type(exc).__name__, exc)
        return empty("player_samples")
    if df is None or df.is_empty():
        return empty("player_samples")
    keep = [c for c in _SAMPLE_COLUMNS if c in df.columns]
    return df.select(keep)


def _teamfights(demo, min_players: int = DEFAULT_MIN_FIGHT_PLAYERS) -> pl.DataFrame:
    """Teamfight detection is a *method* on Demo, not a lazy dataset property.

    Tolerate both shapes: boon has moved things between the two before, and
    guessing wrong here silently costs the entire fight analysis.
    """
    try:
        result = demo.teamfights
        if callable(result):
            result = result(min_players=min_players)
    except Exception as exc:
        log.warning("teamfight detection failed (%s: %s)", type(exc).__name__, exc)
        return empty("teamfights")
    return empty("teamfights") if result is None else result


def _trooper_samples(demo, teamfights: pl.DataFrame) -> pl.DataFrame:
    """Sample lane creeps around fight endings without loading ``demo.troopers``.

    Raw trooper frames are commonly 4–5 million rows. ``snapshots`` lets boon
    decode only the ticks where a post-fight conversion decision is possible.
    """
    if teamfights.is_empty() or "end_tick" not in teamfights.columns:
        return empty("trooper_samples")
    ticks = _trooper_sample_ticks(demo, teamfights)
    if not ticks:
        return empty("trooper_samples")
    try:
        df = demo.snapshots("troopers", ticks=sorted(ticks))
    except Exception as exc:
        log.warning("could not sample troopers (%s: %s)", type(exc).__name__, exc)
        return empty("trooper_samples")
    if df is None or df.is_empty():
        return empty("trooper_samples")
    keep = [c for c in SCHEMAS["trooper_samples"] if c in df.columns]
    return df.select(keep)


def _trooper_sample_ticks(demo, teamfights: pl.DataFrame) -> list[int]:
    if teamfights.is_empty() or "end_tick" not in teamfights.columns:
        return []
    tick_rate = int(demo.tick_rate or 60)
    total = int(demo.total_ticks or 0)
    stride = max(1, int(TROOPER_SAMPLE_SECONDS * tick_rate))
    lookahead = max(stride, int(TROOPER_LOOKAHEAD_SECONDS * tick_rate))
    ticks: set[int] = set()
    for raw in teamfights["end_tick"].drop_nulls().to_list():
        end = int(raw)
        ticks.update(range(end, min(total, end + lookahead) + 1, stride))
    return sorted(ticks)


def _sample_frames(
    demo,
    kills: pl.DataFrame,
    teamfights: pl.DataFrame,
    sample_seconds: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Collect player and post-fight wave samples in one replay pass.

    Boon applies one tick selector to every requested snapshot dataset, so the
    union is requested and each returned frame is filtered back to its own
    plan.  This collects more temporary trooper rows than the old two-pass
    implementation, but still only about one row per second rather than the
    multi-million-row full trooper frame.
    """
    player_ticks = _sample_ticks(demo, kills, sample_seconds)
    trooper_ticks = _trooper_sample_ticks(demo, teamfights)
    if not player_ticks:
        return empty("player_samples"), _trooper_samples(demo, teamfights)
    if not trooper_ticks:
        return _player_samples(demo, kills, sample_seconds), empty("trooper_samples")

    requested_ticks = sorted(set(player_ticks).union(trooper_ticks))
    try:
        result = demo.snapshots(
            ["player_ticks", "troopers"],
            ticks=requested_ticks,
        )
        player_df = result.get("player_ticks")
        trooper_df = result.get("troopers")
    except Exception as exc:
        log.warning(
            "combined snapshot decode failed; falling back to separate passes (%s: %s)",
            type(exc).__name__,
            exc,
        )
        return (
            _player_samples(demo, kills, sample_seconds),
            _trooper_samples(demo, teamfights),
        )

    players = empty("player_samples")
    if player_df is not None and not player_df.is_empty():
        keep = [c for c in _SAMPLE_COLUMNS if c in player_df.columns]
        players = player_df.filter(pl.col("tick").is_in(player_ticks)).select(keep)

    troopers = empty("trooper_samples")
    if trooper_df is not None and not trooper_df.is_empty():
        keep = [c for c in SCHEMAS["trooper_samples"] if c in trooper_df.columns]
        troopers = trooper_df.filter(pl.col("tick").is_in(trooper_ticks)).select(keep)
    return players, troopers


def load_demo(
    path: str | Path,
    *,
    sample_seconds: float = DEFAULT_SAMPLE_SECONDS,
    min_fight_players: int = DEFAULT_MIN_FIGHT_PLAYERS,
) -> MatchData:
    """Parse a Deadlock ``.dem`` into a :class:`MatchData`."""
    import boon  # imported lazily so the analysis layer stays importable without it

    started = perf_counter()
    demo = boon.Demo(str(path))
    _preload_event_datasets(demo)
    events_finished = perf_counter()
    kills = _frame(demo, "kills")

    teamfights = _teamfights(demo, min_fight_players)
    player_samples, trooper_samples = _sample_frames(
        demo,
        kills,
        teamfights,
        sample_seconds,
    )
    samples_finished = perf_counter()
    log.info(
        "demo decode timing: events %.1fs, snapshots %.1fs, total %.1fs",
        events_finished - started,
        samples_finished - events_finished,
        samples_finished - started,
    )
    return MatchData(
        map_name=str(demo.map_name or "unknown"),
        match_id=demo.match_id,
        build=getattr(demo, "build", None),
        game_mode=demo.game_mode,
        tick_rate=int(demo.tick_rate or 60),
        total_ticks=int(demo.total_ticks or 0),
        game_over_tick=demo.game_over_tick,
        winning_team_num=demo.winning_team_num,
        players=_frame(demo, "players"),
        kills=kills,
        damage=_frame(demo, "damage"),
        objectives=_frame(demo, "objectives"),
        mid_boss=_frame(demo, "mid_boss"),
        urn=_frame(demo, "urn"),
        rift=_frame(demo, "rift"),
        teamfights=teamfights,
        trooper_samples=trooper_samples,
        player_samples=player_samples,
        item_purchases=_frame(demo, "item_purchases"),
        ability_upgrades=_frame(demo, "ability_upgrades"),
        ability_ticks=_frame(demo, "ability_ticks"),
        ability_uses=_frame(demo, "abilities", "ability_uses"),
        neutrals=_frame(demo, "neutrals"),
        # boon's clock accounts for pauses; the linear default does not.
        clock=Clock(int(demo.tick_rate or 60), seconds_fn=demo.tick_to_seconds),
    )
