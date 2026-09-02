"""Decode Deadlock ``.dem`` replays into AI-readable coaching material.

    from deadlock_coach import load_demo, render_report
    print(render_report(load_demo("match.dem")))

The pipeline is independently testable in layers:

    source   .dem  -> MatchData (Polars frames)      [needs boon]
    events   frames -> clock-stamped narrative events
    tactics  frames -> fights, pickoffs, phases, economy, coaching notes
    opportunities -> player-perspective decision windows
    advantage -> resource streams, map control, and ranked win-condition levers
    benchmark -> one player against their rank and against the top rank
    summary  all of it -> the ranked front page for one player
    render   all of it -> Markdown briefing + JSON sidecar
"""

from .advantage import AdvantageLedger, analyze_advantage
from .benchmark import PlayerBenchmark, benchmark_player
from .events import Event, build_timeline, format_timeline
from .match import Clock, MatchData
from .names import Names
from .opportunities import analyze_opportunities, observation_frames
from .render import render_json, render_report
from .summary import QuickSummary, build_summaries, build_summary
from .tactics import (
    analyze_fights,
    economy_curve,
    kill_contexts,
    phases,
    player_reports,
)

__all__ = [
    "AdvantageLedger",
    "Clock",
    "Event",
    "MatchData",
    "Names",
    "PlayerBenchmark",
    "QuickSummary",
    "analyze_advantage",
    "analyze_fights",
    "analyze_opportunities",
    "benchmark_player",
    "build_summaries",
    "build_summary",
    "build_timeline",
    "economy_curve",
    "format_timeline",
    "kill_contexts",
    "load_demo",
    "observation_frames",
    "phases",
    "player_reports",
    "render_json",
    "render_report",
]

__version__ = "0.1.0"


def load_demo(*args, **kwargs):
    """Lazy re-export so importing the package never requires ``boon``."""
    from .source import load_demo as _load_demo

    return _load_demo(*args, **kwargs)
