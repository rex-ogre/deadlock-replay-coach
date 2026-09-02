"""Render a match into the two things a model can actually consume: a compact
Markdown briefing, and a JSON sidecar for tool calls.

The Markdown is written for a reader, not a parser. Every number carries its
unit, every judgement carries the evidence behind it, and anything the data
could not establish is stated as unknown rather than omitted — a model told
"isolated deaths: unknown" will hedge, while one shown a silent gap will
happily invent a number.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable

from .advantage import AdvantageLedger, analyze_advantage
from .events import Event, build_timeline
from .gamedata import GameConstants, load_constants
from .match import MatchData
from .names import Names
from .opportunities import OpportunityAnalysis, analyze_opportunities
from .physics import HU_PER_METER, reference_speeds
from .skillstats import PlayerSkillStats
from .summary import QuickSummary, build_summary, render_summary_section
from .tactics import (
    CONVERSION_HERO_READY_DISTANCE,
    UNSUPPORTABLE_SECONDS,
    CONVERSION_HERO_SETUP_DISTANCE,
    EconomySnapshot,
    FightSummary,
    PhaseStat,
    PlayerReport,
    analyze_fights,
    biggest_swing,
    economy_curve,
    fight_record_by_phase,
    kill_contexts,
    phase_at,
    phase_stats,
    phases,
    player_reports,
)

# A full match is ~600-1200 events. Trimming below this loses the plot; above
# it, the timeline starts crowding out the analysis in a fixed context window.
DEFAULT_MAX_TIMELINE_EVENTS = 400

# Dropped first when trimming: economically real, narratively noisy.
LOW_VALUE_KINDS = ("purchase", "urn", "rift")

HEADER_NOTE = """\
This is a decoded Deadlock replay. Factual match sections are measured from the
`.dem` file. The player-perspective opportunity section is an explicitly
labeled estimate, not a guarantee. Sections are ordered analysis-first; the raw
event timeline is last.

Reading notes:
- Times are match clock (mm:ss) with paused time excluded.
- Souls figures are net worth (current souls + souls already spent).
- "unknown" means the demo did not carry that data. Do not fill it in.
- Fight verdicts are kill difference inside the fight window. Winning a fight
  and converting it into an objective are scored separately, on purpose.
- Post-fight conversion reads check the actual creep wave first. "No reliable
  structure window" means the report will not blame a player for failing to
  push a lane that had no usable troopers.
- Opportunity counts are independent decision windows. Do not add them to
  actual kills as a theoretical maximum.
- The advantage ledger answers "how is this match won": it splits the soul
  lead into the streams that produced it, for both teams, so the gap in each
  stream is visible. Streams marked estimated carry an inference; say so.
- "Bottom line" is the ranked summary for the focused player. It contains no
  new evidence: every line in it names the later section that holds the
  working. Lead with it, and keep its ordering — macro before mechanics.\
"""


def render_report(
    md: MatchData,
    names: Names | None = None,
    *,
    max_timeline_events: int = DEFAULT_MAX_TIMELINE_EVENTS,
    opportunity_analysis: OpportunityAnalysis | None = None,
    focus_hero_ids: Iterable[int] | None = None,
    advantage_ledger: AdvantageLedger | None = None,
    constants: GameConstants | None = None,
    skill_stats: dict[int, PlayerSkillStats] | None = None,
) -> str:
    names = names or Names.from_boon()
    constants = constants or load_constants()
    timeline = build_timeline(md, names)
    fights = analyze_fights(md, names)
    focus = set(focus_hero_ids) if focus_hero_ids is not None else None
    reports = player_reports(md)
    if focus is not None:
        reports = [report for report in reports if report.hero_id in focus]
    curve = economy_curve(md)
    opportunities = opportunity_analysis or analyze_opportunities(md, constants=constants)
    ledger = advantage_ledger or analyze_advantage(
        md,
        names,
        focus_hero_ids=tuple(focus) if focus else None,
    )

    summary = build_match_summary(
        md,
        names,
        focus=focus,
        fights=fights,
        reports=reports,
        curve=curve,
        opportunities=opportunities,
        constants=constants,
        skill_stats=skill_stats,
    )

    blocks = [
        HEADER_NOTE,
        render_summary_section(summary) if summary else "",
        _match_header(md, names),
        _focus_note(md, names, focus),
        _roster(md, names, reports, focused=focus is not None),
        _phases(md, names),
        _economy(md, names, curve),
        _mechanics(md, names, constants, skill_stats or {}, focus),
        _win_conditions(md, names, ledger),
        _fights(md, names, fights, focus),
        _opportunities(md, names, reports, opportunities, focus),
        _coaching(md, names, reports),
        _kill_patterns(md, names, focus, constants),
        _timeline(md, timeline, max_timeline_events),
    ]
    return "\n\n".join(b for b in blocks if b).rstrip() + "\n"


def build_match_summary(
    md: MatchData,
    names: Names,
    *,
    focus: set[int] | None,
    fights: list[FightSummary],
    reports: list[PlayerReport],
    curve: list[EconomySnapshot],
    opportunities: OpportunityAnalysis,
    constants: GameConstants,
    skill_stats: dict[int, PlayerSkillStats] | None,
) -> QuickSummary | None:
    """The ranked front page, for a report focused on exactly one player.

    A twelve-player report has no "you", and a summary that tried to rank
    everyone's mistakes at once would be the very wall of text it exists to
    replace. So an unfocused report gets no summary and stays a reference
    document.
    """
    if not focus or len(focus) != 1:
        return None
    return build_summary(
        md,
        names,
        hero_id=next(iter(focus)),
        fights=fights,
        reports=reports,
        kills=kill_contexts(md, constants),
        phase_list=phases(md),
        curve=curve,
        opportunities=opportunities,
        constants=constants,
        skill_stats=skill_stats,
    )


# ------------------------------------------------------------- sections


def _match_header(md: MatchData, names: Names) -> str:
    assert md.clock is not None
    winner = names.team(md.winning_team_num) if md.winning_team_num else "unknown"
    lines = [
        "## Match",
        "",
        f"- Map: `{md.map_name}`",
        f"- Match id: {md.match_id if md.match_id is not None else 'unknown'}",
        f"- Duration: {md.clock.mmss(md.end_tick)} ({md.tick_rate} tick)",
        f"- Winner: {winner}",
        f"- Teams: {', '.join(names.team(t) for t in md.team_nums) or 'unknown'}",
    ]
    return "\n".join(lines)


def _focus_note(md: MatchData, names: Names, focus: set[int] | None) -> str:
    if focus is None:
        return ""
    labels = [
        f"{names.hero(hero)} ({md.player_name(hero) or '—'})" for hero in sorted(focus)
    ]
    return "## Report focus\n\n- Player: " + ", ".join(labels)


def _roster(
    md: MatchData,
    names: Names,
    reports: list[PlayerReport],
    *,
    focused: bool = False,
) -> str:
    if not reports:
        return ""
    lines = ["## Focused player final line" if focused else "## Roster and final line", ""]
    for team in md.team_nums:
        team_reports = [report for report in reports if report.team_num == team]
        if not team_reports:
            continue
        lines.append(f"**{names.team(team)}**")
        lines.append("")
        lines.append("| Hero | Player | Lane | K/D/A | Net worth | Level | Last hits | Hero dmg |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for r in team_reports:
            lines.append(
                f"| {names.hero(r.hero_id)} | {r.player_name or '—'} | {names.lane(r.start_lane)} "
                f"| {r.kills}/{r.deaths}/{r.assists} | {_num(r.final_net_worth)} "
                f"| {_num(r.final_level)} | {_num(r.last_hits)} | {_num(r.hero_damage)} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def _phases(md: MatchData, names: Names) -> str:
    assert md.clock is not None
    phase_list = phases(md)
    if not phase_list:
        return ""
    record = fight_record_by_phase(md, names)
    lines = ["## Phases", ""]
    for p in phase_list:
        lines.append(
            f"- **{p.name}** {md.clock.mmss(p.start_tick)}–{md.clock.mmss(p.end_tick)} "
            f"(boundary: {p.trigger})"
        )
    if any(record.values()):
        lines.append("")
        lines.append("Decisive fights won, by phase — this is the shape of the match:")
        lines.append("")
        lines.append("| Phase | " + " | ".join(names.team(t) for t in md.team_nums) + " |")
        lines.append("| --- |" + " --- |" * len(md.team_nums))
        for p in phase_list:
            won = record.get(p.name, {})
            lines.append(
                f"| {p.name} ({md.clock.mmss(p.start_tick)}–{md.clock.mmss(p.end_tick)}) | "
                + " | ".join(str(won.get(t, 0)) for t in md.team_nums)
                + " |"
            )
    return "\n".join(lines)


def _economy(md: MatchData, names: Names, curve: list[EconomySnapshot]) -> str:
    if not curve:
        return "## Soul economy\n\nNo sampled economy data in this demo."
    assert md.clock is not None
    lines = ["## Soul economy", ""]

    final = curve[-1]
    lines.append(
        "- Final net worth: "
        + ", ".join(f"{names.team(t)} {v:,}" for t, v in sorted(final.net_worth_by_team.items()))
    )

    peak = max(curve, key=lambda s: s.lead)
    if peak.lead > 0:
        lines.append(
            f"- Largest lead: {names.team(peak.lead_team)} +{peak.lead:,} at "
            f"{md.clock.mmss(peak.tick)}"
        )

    swing = biggest_swing(curve)
    if swing is not None:
        before, after = swing
        lines.append(
            f"- Sharpest swing: {md.clock.mmss(before.tick)} → {md.clock.mmss(after.tick)} "
            + " / ".join(
                f"{names.team(t)} {before.net_worth_by_team.get(t, 0):,}→"
                f"{after.net_worth_by_team.get(t, 0):,}"
                for t in sorted(after.net_worth_by_team)
            )
        )

    lines.append("")
    lines.append("| Time | " + " | ".join(names.team(t) for t in md.team_nums) + " | Lead |")
    lines.append("| --- |" + " --- |" * (len(md.team_nums) + 1))
    for snapshot in _thin(curve, 12):
        row = " | ".join(f"{snapshot.net_worth_by_team.get(t, 0):,}" for t in md.team_nums)
        lead = (
            f"{names.team(snapshot.lead_team)} +{snapshot.lead:,}" if snapshot.lead_team else "even"
        )
        lines.append(f"| {md.clock.mmss(snapshot.tick)} | {row} | {lead} |")
    return "\n".join(lines)


WIN_CONDITION_NOTE = """\
Deadlock is not won by one play; it is won by compounding resources faster than
the other side. There are only two ways to compound: gather more of your own,
and take more of theirs away. Every row below is one such stream, reported for
both teams, because a stream you cannot compare is not a lever.\
"""


def _win_conditions(md: MatchData, names: Names, ledger: AdvantageLedger) -> str:
    if not ledger.teams:
        return ""
    assert md.clock is not None
    teams = list(ledger.teams)
    lines = ["## Win conditions — the advantage ledger", "", WIN_CONDITION_NOTE, ""]

    if ledger.focus_team is not None:
        lines.append(
            f"Levers below are written for **{names.team(ledger.focus_team)}** "
            f"({ledger.focus_reason})."
        )
        lines.append("")

    if ledger.streams:
        lines.append("### Resource streams, whole match")
        lines.append("")
        lines.append("| Stream | " + " | ".join(names.team(t) for t in teams) + " | Edge | Basis |")
        lines.append("| --- |" + " --- |" * (len(teams) + 2))
        for stream in ledger.streams:
            cells = " | ".join(_stream_cell(stream.by_team.get(t), stream.unit) for t in teams)
            leader = stream.leader
            # "+14" on a lower-is-better row (time dead, camps conceded) reads
            # as the leader having *more* of it, which is the opposite.
            edge = (
                f"{names.team(leader)} ahead by "
                f"{_stream_cell(stream.margin, stream.unit)}"
                if leader is not None and stream.margin
                else "level"
            )
            basis = "estimated" if stream.estimated else "measured"
            if stream.source not in ("measured", "estimated"):
                basis = stream.source
            lines.append(f"| {stream.label} | {cells} | {edge} | {basis} |")
        lines.append("")

    if ledger.windows:
        lines.append("### Where the gap opened, five minutes at a time")
        lines.append("")
        lines.append(
            "Souls gained inside each window, not running totals — a team can lead the match "
            "and still be losing the window you are reading."
        )
        lines.append("")
        header = " | ".join(f"{names.team(t)} souls" for t in teams)
        lines.append(f"| Window | {header} | Kills | Structures | Camps (raids) | Prizes | Ground |")
        lines.append("| --- |" + " --- |" * (len(teams) + 5))
        for window in ledger.windows:
            gained = " | ".join(
                f"{window.gained_by_team.get(t, 0):,} ({window.per_min_by_team.get(t, 0):,.0f}/min)"
                for t in teams
            )
            kills = "–".join(str(window.kills_by_team.get(t, 0)) for t in teams)
            structures = "–".join(str(window.structures_by_team.get(t, 0)) for t in teams)
            camps = "–".join(
                f"{window.camps_by_team.get(t, 0)} ({window.raids_by_team.get(t, 0)})"
                for t in teams
            )
            prizes = "–".join(str(window.prizes_by_team.get(t, 0)) for t in teams)
            ground = (
                f"{names.team(window.leader)} +{window.margin:,} ({window.driver})"
                if window.leader is not None
                else "even"
            )
            lines.append(
                f"| {window.label} | {gained} | {kills} | {structures} | {camps} | "
                f"{prizes} | {ground} |"
            )
        lines.append("")
        lines.append(
            "Column order inside each cell follows the team column order. "
            "`Camps (raids)` counts camps cleared, and in brackets how many of those were "
            "inside the *enemy* half."
        )
        lines.append("")

    lines.extend(_territory_lines(names, ledger))
    lines.extend(_downtime_lines(md, names, ledger))
    lines.extend(_lever_lines(ledger))

    if ledger.caveats:
        lines.append("What this section cannot see:")
        lines.append("")
        lines.extend(f"- {caveat}" for caveat in ledger.caveats)
    return "\n".join(lines).rstrip()


def _territory_lines(names: Names, ledger: AdvantageLedger) -> list[str]:
    rows = [t for t in ledger.territory if t.share is not None]
    if not rows:
        return []
    lines = [
        "### Map control — whose ground the match was played on",
        "",
        "Position samples only. No vision model is involved: a hero was either standing in the "
        "enemy half or was not.",
        "",
        "| Team | Time in the enemy half | Kills scored there | Deaths at home |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {names.team(row.team_num)} | {row.share:.1%} | {row.kills_in_enemy_half} "
            f"| {row.deaths_in_own_half} |"
        )
    lines.append("")
    return lines


def _downtime_lines(md: MatchData, names: Names, ledger: AdvantageLedger) -> list[str]:
    rows = [d for d in ledger.downtime if d.deaths]
    if not rows:
        return []
    lines = [
        "### Death downtime — the resource nobody counts",
        "",
        "Respawn is not recorded in the demo; a dead hero simply stops emitting positional "
        "rows, so time dead is the gap until they reappear. Souls forgone prices that gap at "
        "the player's *own* earning rate for this match — an estimate, and only meaningful "
        "next to the other rows here.",
        "",
        "| Player | Team | Deaths | Time dead | Own souls/min | Souls forgone (est) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        label = names.hero(row.hero_id)
        if row.player_name:
            label += f" ({row.player_name})"
        rate = "unknown" if row.souls_per_min is None else f"{row.souls_per_min:,.0f}"
        forgone = "unknown" if row.souls_forgone is None else f"{row.souls_forgone:,}"
        lines.append(
            f"| {label} | {names.team(row.team_num)} | {row.deaths} "
            f"| {_mmss(row.seconds)} | {rate} | {forgone} |"
        )
    lines.append("")
    return lines


def _lever_lines(ledger: AdvantageLedger) -> list[str]:
    if not ledger.levers:
        return []
    lines = [
        "### Levers — ranked by the souls at stake",
        "",
        "Ordered by measurable size, not by how bad it looked. A lever with no soul figure is "
        "not a smaller lever; it is one whose soul value the demo does not carry.",
        "",
    ]
    for index, lever in enumerate(ledger.levers, start=1):
        stake = (
            f"{lever.souls_at_stake:,} souls"
            if lever.souls_at_stake is not None
            else "souls not derivable"
        )
        lines.append(f"{index}. **{lever.title}** — {stake} (confidence: {lever.confidence})")
        lines.extend(f"   - {line}" for line in lever.evidence)
        lines.append(f"   - **Do instead:** {lever.action}")
        lines.append("")
    return lines


def _stream_cell(value: float | None, unit: str) -> str:
    if value is None:
        return "unknown"
    if unit == "%":
        return f"{value:,.1f}%"
    if unit == "s":
        return _mmss(value)
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.1f}"


def _mmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 60:02d}:{total % 60:02d}"


def _fights(
    md: MatchData,
    names: Names,
    fights: list[FightSummary],
    focus: set[int] | None = None,
) -> str:
    if not fights:
        return "## Teamfights\n\nNo teamfights detected (demo may lack damage data)."
    assert md.clock is not None

    converted = sum(1 for f in fights if f.converted_into)
    decisive = [f for f in fights if f.winner is not None]
    unconverted_wins = [f for f in decisive if not f.converted_into]
    push_now = sum(
        1
        for f in unconverted_wins
        if f.conversion_assessment is not None
        and f.conversion_assessment.status == "push_now"
    )
    setup = sum(
        1
        for f in unconverted_wins
        if f.conversion_assessment is not None
        and f.conversion_assessment.status == "setup_required"
    )
    unavailable = sum(
        1
        for f in unconverted_wins
        if f.conversion_assessment is not None
        and f.conversion_assessment.status == "no_structure_window"
    )
    # Tabling every cluster buries the match in lane poke: most detected fights
    # trade no kills at all. List the ones that resolved, count the rest.
    standoffs = len(fights) - len(decisive)
    lines = [
        "## Teamfights",
        "",
        f"{len(fights)} fights detected; {len(decisive)} had a kill winner; "
        f"{converted} converted into an objective within 45s. "
        f"The other {standoffs} traded no kills (poke and standoffs) and are "
        f"omitted from the table below.",
        f"Among unconverted wins, {push_now} were immediately pushable, {setup} needed "
        f"wave setup or a rotation, and {unavailable} had no reliable structure window. "
        f"Unknown wave data is kept unknown rather than scored as a missed conversion.",
        "",
    ]
    for team in md.team_nums:
        won = [f for f in fights if f.winner == team]
        conv = sum(1 for f in won if f.converted_into)
        immediate = sum(
            1
            for f in won
            if not f.converted_into
            and f.conversion_assessment is not None
            and f.conversion_assessment.status == "push_now"
        )
        needs_setup = sum(
            1
            for f in won
            if not f.converted_into
            and f.conversion_assessment is not None
            and f.conversion_assessment.status == "setup_required"
        )
        no_window = sum(
            1
            for f in won
            if not f.converted_into
            and f.conversion_assessment is not None
            and f.conversion_assessment.status == "no_structure_window"
        )
        rate = f"{conv}/{len(won)}" if won else "0/0"
        noun = "fight" if len(won) == 1 else "fights"
        lines.append(
            f"- {names.team(team)}: won {len(won)} {noun}, converted {rate}; "
            f"unconverted reads: {immediate} push now, {needs_setup} setup, "
            f"{no_window} no structure window."
        )
    lines.append("")
    lines.append(
        "| # | Time | Phase | Size | Result | Hero dmg | Actual conversion | Available next move |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for f in decisive:
        winner = names.team(f.winner) if f.winner is not None else "—"
        result = f.verdict if f.winner is None else f"{winner} {f.verdict}"
        lines.append(
            f"| {f.fight_id} | {md.clock.mmss(f.start_tick)} | {f.phase} | {f.engagement} "
            f"| {result} | {f.hero_damage:,} | {f.converted_into or '—'} "
            f"| {_conversion_read(md, names, f, focus)} |"
        )
    return "\n".join(lines)


def _conversion_read(
    md: MatchData,
    names: Names,
    fight: FightSummary,
    focus: set[int] | None = None,
) -> str:
    read = fight.conversion_assessment
    if read is None:
        return "—"
    if read.status == "converted":
        return f"Converted: {read.target}"
    if read.status == "push_now":
        prefix = "Push now"
    elif read.status == "setup_required":
        prefix = "Setup required"
    elif read.status == "no_structure_window":
        prefix = "No reliable structure window"
    else:
        prefix = "Unknown"

    if read.target:
        detail = f"{prefix}: {read.target}"
    else:
        detail = prefix
    if read.status in ("push_now", "setup_required"):
        facts = []
        if read.allied_front_troopers is not None:
            facts.append(
                f"{read.allied_front_troopers} allied/{read.enemy_contesting_troopers or 0} contesting troopers"
            )
        if read.wave_distance is not None:
            facts.append(f"wave {read.wave_distance:.0f} away")
        if read.nearest_winner_distance is not None:
            facts.append(f"nearest healthy winner {read.nearest_winner_distance:.0f} away")
        if focus:
            for hero in sorted(focus):
                distance = read.winner_distances.get(hero)
                if distance is None:
                    facts.append(f"{names.hero(hero)} was not a healthy conversion participant")
                elif distance <= CONVERSION_HERO_READY_DISTANCE:
                    facts.append(f"{names.hero(hero)} could join now ({distance:.0f} away)")
                elif distance <= CONVERSION_HERO_SETUP_DISTANCE:
                    facts.append(f"{names.hero(hero)} needed a rotation ({distance:.0f} away)")
                else:
                    facts.append(f"{names.hero(hero)} was too far ({distance:.0f} away)")
        if facts:
            detail += " (" + ", ".join(facts) + ")"
    elif read.reason:
        detail += f": {read.reason}"
    if read.alternatives:
        detail += "; " + "; ".join(read.alternatives)
    return detail


def _opportunities(
    md: MatchData,
    names: Names,
    reports: list[PlayerReport],
    analysis: OpportunityAnalysis,
    focus: set[int] | None = None,
) -> str:
    """Render estimates with their evidence and uncertainty attached."""
    assert md.clock is not None
    kills_by_hero: dict[int, list] = {}
    jungle_by_hero: dict[int, list] = {}
    rotation_by_hero: dict[int, list] = {}
    macro_by_hero: dict[int, list] = {}
    for window in analysis.kill_windows:
        if focus is not None and window.observer_hero_id not in focus:
            continue
        kills_by_hero.setdefault(window.observer_hero_id, []).append(window)
    for window in analysis.jungle_windows:
        if focus is not None and window.observer_hero_id not in focus:
            continue
        jungle_by_hero.setdefault(window.observer_hero_id, []).append(window)
    for window in analysis.rotation_windows:
        if focus is not None and window.observer_hero_id not in focus:
            continue
        rotation_by_hero.setdefault(window.observer_hero_id, []).append(window)
    for window in analysis.macro_windows:
        if focus is not None and window.observer_hero_id not in focus:
            continue
        macro_by_hero.setdefault(window.observer_hero_id, []).append(window)

    lines = [
        "## Player-perspective opportunities",
        "",
        "These are conservative decision windows from information available to that "
        "player's team at the time. They are not guaranteed outcomes and are not an "
        "estimate of maximum possible kills.",
        "",
        "Four kinds of window are detected:",
        "",
        "- **Kill pressure** — an opponent you could have finished where you stood.",
        "- **Cross-lane rotation** — an actual one-kill loss elsewhere that your "
        "estimated arrival could have reached before the first allied death.",
        "- **Macro** — what the map was offering (a man advantage, the Mid Boss, the Urn) "
        "while your team spent it on nothing, plus structures you left undefended.",
        "- **Invade / scout** — enemy jungle you could have taken or checked.",
        "",
        f"- Information boundary: {analysis.vision_model}.",
        "- Hidden live enemy positions and hidden camp state do not enter scoring.",
        "- Macro windows use only announced information — the kill feed, fallen "
        "structures, and Mid Boss / Urn events — so they need no vision model at all.",
        "- `high`/`medium`/`low` describe evidence quality, not success probability.",
        "- Each player's table lists **every** window found, ranked by importance. "
        "Confidence, detector score, and decision type determine rank; time breaks ties. The "
        "\"Also note\" column carries only what is specific to that moment; the "
        "caveats below are true of every row and are not repeated on each one.",
        "",
        *_standing_caveats(analysis),
    ]
    if not reports:
        lines.append("No roster data was available for player-perspective analysis.")
        return "\n".join(lines)

    lines.extend(
        [
            "| Player | Actual kills | Kill-pressure signals | Cross-lane rotations "
            "| Macro windows | Actionable invade / scout-only |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for report in reports:
        lines.append(
            f"| {names.hero(report.hero_id)} ({report.player_name or '—'}) | {report.kills} "
            f"| {_confidence_counts(kills_by_hero.get(report.hero_id, []))} "
            f"| {_confidence_counts(rotation_by_hero.get(report.hero_id, []))} "
            f"| {_macro_counts(macro_by_hero.get(report.hero_id, []))} "
            f"| {_jungle_counts(jungle_by_hero.get(report.hero_id, []))} |"
        )

    detailed = sorted(
        set(kills_by_hero) | set(jungle_by_hero) | set(rotation_by_hero) | set(macro_by_hero)
    )
    if not detailed:
        lines.extend(
            [
                "",
                "No window met the conservative evidence threshold. This means the replay "
                "data could not establish one, not that no opportunity existed.",
            ]
        )
        return "\n".join(lines)

    for hero in detailed:
        rows = _opportunity_rows(
            md,
            names,
            kills_by_hero.get(hero, []),
            rotation_by_hero.get(hero, []),
            macro_by_hero.get(hero, []),
            jungle_by_hero.get(hero, []),
        )
        lines.extend(
            [
                "",
                f"### {names.hero(hero)} — every window, ranked by importance ({len(rows)})",
                "",
                "| Rank | Time | Phase | Type | What was on | Confidence | Why | Also note |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for rank, row in enumerate(rows, start=1):
            lines.append(
                f"| {rank} | {_clock_window(md, row.start_tick, row.end_tick)} | {row.phase} "
                f"| {row.category} | {row.action} | {row.confidence} "
                f"| {'; '.join(row.evidence)} "
                f"| {'; '.join(row.limitations) or '—'} |"
            )
    return "\n".join(lines)


@dataclass(frozen=True)
class _Row:
    """One window of any kind, flattened into one importance-ranked table."""

    start_tick: int
    end_tick: int
    phase: str
    category: str
    action: str
    confidence: str
    priority: int
    evidence: tuple[str, ...]
    limitations: tuple[str, ...]


def _opportunity_rows(
    md: MatchData,
    names: Names,
    kills: list,
    rotations: list,
    macro: list,
    jungle: list,
) -> list[_Row]:
    phase_list = phases(md)
    rows: list[_Row] = []

    for w in kills:
        rows.append(
            _Row(
                start_tick=w.start_tick,
                end_tick=w.end_tick,
                phase=w.phase,
                category="kill pressure",
                action=(
                    f"finish {names.hero(w.target_hero_id)} "
                    f"({w.target_health_pct:.0%} hp, {w.distance:.0f} units away)"
                ),
                confidence=w.confidence,
                priority=_row_priority("kill", w),
                evidence=w.evidence,
                limitations=w.limitations,
            )
        )

    for w in rotations:
        origin = names.lane(w.from_lane) if w.from_lane is not None else "off-lane"
        toward = names.lane(w.to_lane) if w.to_lane is not None else "off-lane"
        rows.append(
            _Row(
                start_tick=w.start_tick,
                end_tick=w.end_tick,
                phase=w.phase,
                category="rotation",
                action=(
                    f"rotate {origin} → {toward} onto {names.hero(w.target_hero_id)} "
                    f"({w.target_health_pct:.0%} hp, {w.travel_distance:.0f} units away)"
                ),
                confidence=w.confidence,
                priority=_row_priority("rotation", w),
                evidence=w.evidence,
                limitations=w.limitations,
            )
        )

    for w in macro:
        rows.append(
            _Row(
                start_tick=w.start_tick,
                end_tick=w.end_tick,
                phase=w.phase,
                category="macro",
                action=_macro_action(names, w),
                confidence=w.confidence,
                priority=_row_priority("macro", w),
                evidence=w.evidence,
                limitations=w.limitations,
            )
        )

    for w in jungle:
        verb = "take" if w.confidence in ("high", "medium") else "scout"
        lane = names.lane(w.lane) if w.lane is not None else "unknown"
        rows.append(
            _Row(
                start_tick=w.start_tick,
                end_tick=w.end_tick,
                phase=phase_at(phase_list, w.start_tick),
                category="invade" if verb == "take" else "scout",
                action=(
                    f"{verb} enemy camp #{w.camp_id} near {lane} lane "
                    f"({w.distance:.0f} units away)"
                ),
                confidence=w.confidence,
                priority=_row_priority("jungle", w),
                evidence=w.evidence,
                limitations=w.limitations,
            )
        )

    return sorted(rows, key=lambda r: (-r.priority, r.start_tick, r.action))


def _row_priority(kind: str, window) -> int:
    """Comparable importance within one player's report; no rows are dropped."""
    confidence = {"low": 1, "medium": 2, "high": 3}.get(window.confidence, 0)
    type_weight = {"rotation": 4, "macro": 3, "kill": 2, "jungle": 1}[kind]
    if kind == "jungle":
        intrinsic = window.enemies_last_seen_away
        intrinsic += 2 if window.camp_status == "visible and available" else 0
        intrinsic += 1 if window.distance <= 3_500 else 0
    else:
        intrinsic = int(getattr(window, "score", 0))
    return confidence * 100 + intrinsic * 5 + type_weight


def _clock_window(md: MatchData, start: int, end: int) -> str:
    assert md.clock is not None
    first = md.clock.mmss(start)
    last = md.clock.mmss(end)
    return first if first == last else f"{first}–{last}"


def _confidence_counts(items: list) -> str:
    if not items:
        return "none established"
    counts: dict[str, int] = {}
    for item in items:
        counts[item.confidence] = counts.get(item.confidence, 0) + 1
    return " / ".join(
        f"{counts[level]} {level}"
        for level in ("high", "medium", "low")
        if counts.get(level)
    )


def _standing_caveats(analysis: OpportunityAnalysis) -> list[str]:
    """The always-true caveats, stated once instead of on several hundred rows."""
    labels = {
        "kill": "Kill pressure",
        "rotation": "Cross-lane rotation",
        "macro": "Man-advantage / Mid Boss / Urn",
        "defend": "Undefended structure",
        "jungle": "Invade / scout",
    }
    caveats = analysis.standing_caveats or {}
    if not caveats:
        return []
    out = ["Always true of these estimates, whatever the row says:", ""]
    for key, label in labels.items():
        for text in caveats.get(key, ()):
            out.append(f"- *{label}* — {text}.")
    out.append("")
    return out


def _target_label(names: Names, target_kind: str) -> str:
    if target_kind == "mid_boss":
        return "Mid Boss"
    if target_kind == "urn":
        return "Urn"
    return names.objective(target_kind)


def _macro_target(names: Names, window) -> str:
    """Name a macro target from its structured fields, never a pre-baked string."""
    label = _target_label(names, window.target_kind)
    if window.target_kind not in ("mid_boss", "urn"):
        if window.target_lane is not None:
            label = f"{names.lane(window.target_lane)}-lane {label}"
        if window.target_team_num is not None:
            label = f"{label} ({names.team(window.target_team_num)})"
    if window.alternatives:
        others = ", ".join(_target_label(names, kind) for kind in window.alternatives)
        label = f"{label} — also on: {others}"
    return label


def _macro_action(names: Names, window) -> str:
    """The macro window as an instruction, so the row reads as advice."""
    target = _macro_target(names, window)
    verbs = {"siege": "push", "defend": "defend", "mid_boss": "take", "urn": "run"}
    verb = verbs.get(window.kind, "contest")
    if window.distance is not None:
        target = f"{target} ({window.distance:.0f} units away)"
    return f"{verb} {target}"


def _macro_counts(items: list) -> str:
    if not items:
        return "none established"
    counts: dict[str, int] = {}
    for item in items:
        counts[item.kind] = counts.get(item.kind, 0) + 1
    order = ("mid_boss", "siege", "urn", "defend")
    labels = {"mid_boss": "Mid Boss", "siege": "siege", "urn": "Urn", "defend": "defend"}
    return " / ".join(f"{counts[k]} {labels[k]}" for k in order if counts.get(k))


def _jungle_counts(items: list) -> str:
    actionable = sum(item.confidence in ("high", "medium") for item in items)
    scout = sum(item.confidence == "low" for item in items)
    parts = []
    if actionable:
        parts.append(f"{actionable} invade")
    if scout:
        parts.append(f"{scout} scout-only")
    return " / ".join(parts) if parts else "none established"


def _phase_lines(md: MatchData, stats: list[PhaseStat], hero_id: int) -> list[str]:
    """One line per phase for a player, so the reader can see the whole arc and
    not just the totals."""
    assert md.clock is not None
    mine = [s for s in stats if s.hero_id == hero_id]
    if not mine:
        return []
    out = ["- Phase by phase:"]
    for s in mine:
        window = f"{md.clock.mmss(s.start_tick)}–{md.clock.mmss(s.end_tick)}"
        share = f", {s.damage_share:.0%} of team dmg" if s.damage_share is not None else ""
        out.append(
            f"  - **{s.phase}** ({window}): {s.kills}/{s.deaths}/{s.assists}, "
            f"+{_num(s.net_worth_gained)} souls, {_num(s.hero_damage)} hero dmg{share}, "
            f"{_num(s.last_hits)} last hits, {_num(s.objective_damage)} objective dmg"
        )
    return out


def _coaching(md: MatchData, names: Names, reports: list[PlayerReport]) -> str:
    if not reports:
        return ""
    stats = phase_stats(md)
    lines = ["## Per-player review", ""]
    for r in reports:
        head = f"### {names.hero(r.hero_id)}"
        if r.player_name:
            head += f" — {r.player_name}"
        head += f" ({names.team(r.team_num)}, {names.lane(r.start_lane)} lane)"
        lines.append(head)
        lines.append("")
        kp = f"{r.kill_participation:.0%}" if r.kill_participation is not None else "unknown"
        lines.append(f"- K/D/A {r.kills}/{r.deaths}/{r.assists}, kill participation {kp}")
        if r.deaths == 0:
            # Saying "deaths by phase: —, isolated: unknown" for a player who
            # never died reads as missing data rather than a clean sheet.
            lines.append("- Did not die once all match.")
        else:
            iso = "unknown" if r.isolated_deaths is None else str(r.isolated_deaths)
            deaths_split = ", ".join(f"{k} {v}" for k, v in r.deaths_by_phase.items())
            lines.append(f"- Deaths by phase: {deaths_split}; died with no ally nearby: {iso}")
        lines.append(
            f"- Net worth {_num(r.final_net_worth)}, last hits {_num(r.last_hits)}, "
            f"denies {_num(r.denies)}, objective damage {_num(r.objective_damage)}"
        )
        lines.extend(_phase_lines(md, stats, r.hero_id))
        for note in r.notes:
            lines.append(f"- **Coaching note:** {note}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _kill_patterns(
    md: MatchData,
    names: Names,
    focus: set[int] | None = None,
    constants: GameConstants | None = None,
) -> str:
    contexts = kill_contexts(md, constants)
    if focus is not None:
        contexts = [context for context in contexts if context.victim_hero_id in focus]
    if not contexts:
        return ""
    known = [c for c in contexts if c.outnumbered is not None]
    if not known:
        return (
            "## Kill patterns\n\n"
            f"{len(contexts)} kills recorded, but no positional samples were available, "
            "so pickoffs cannot be distinguished from even fights."
        )
    picks = [c for c in known if c.outnumbered]
    subject = "focused-player deaths" if focus is not None else "classified kills"
    lines = [
        "## Kill patterns",
        "",
        f"{len(picks)} of {len(known)} {subject} were outnumbered pickoffs "
        f"(attackers exceeded the victim's nearby living allies by 2+).",
        "",
    ]
    by_phase: dict[str, int] = {}
    for c in picks:
        by_phase[c.phase] = by_phase.get(c.phase, 0) + 1
    if by_phase:
        lines.append("- Pickoffs by phase: " + ", ".join(f"{k} {v}" for k, v in by_phase.items()))

    victims: dict[int, int] = {}
    for c in picks:
        if c.victim_hero_id is not None:
            victims[c.victim_hero_id] = victims.get(c.victim_hero_id, 0) + 1
    if victims:
        worst = sorted(victims.items(), key=lambda kv: kv[1], reverse=True)[:3]
        lines.append(
            "- Most picked off: " + ", ".join(f"{names.hero(h)} ({n})" for h, n in worst)
        )

    # A death with nobody nearby is two different mistakes depending on whether
    # help existed. Splitting them keeps the model from calling every solo
    # death a positioning error, when some of them are a correct split and the
    # error belongs to whoever was meant to be covering the map.
    alone = [c for c in known if c.isolated and c.support_seconds is not None]
    if alone:
        stranded = [c for c in alone if c.support_seconds >= UNSUPPORTABLE_SECONDS]
        lines.append("")
        lines.append(
            f"- Of {len(alone)} deaths with no living ally nearby, {len(stranded)} had "
            f"no teammate who could have arrived within {UNSUPPORTABLE_SECONDS:.0f}s "
            "even at best-case movement speed. Those are a team-shape read, not a "
            "positioning read on the victim."
        )
        for context in sorted(stranded, key=lambda c: -(c.support_seconds or 0))[:3]:
            assert md.clock is not None
            lines.append(
                f"  - {md.clock.mmss(context.tick)} {names.hero(context.victim_hero_id)}: "
                f"{context.support_read}"
            )
    return "\n".join(lines)


def _timeline(md: MatchData, events: list[Event], limit: int) -> str:
    assert md.clock is not None
    shown, trimmed = _trim(events, limit)
    lines = ["## Event timeline", ""]
    if trimmed:
        lines.append(f"_{trimmed} lower-value events trimmed to fit; kills and objectives kept._")
        lines.append("")
    lines.append("```")
    lines.extend(f"[{md.clock.mmss(e.tick)}] {e.text}" for e in shown)
    lines.append("```")
    return "\n".join(lines)


# --------------------------------------------------------------- helpers


def _num(value) -> str:
    return "unknown" if value is None else f"{int(value):,}"


def _thin(items: list, target: int) -> list:
    """Evenly spaced subset, always keeping the first and last."""
    if len(items) <= target or target < 2:
        return items
    step = (len(items) - 1) / (target - 1)
    indices = sorted({round(i * step) for i in range(target)})
    return [items[i] for i in indices]


def _trim(events: list[Event], limit: int) -> tuple[list[Event], int]:
    if len(events) <= limit:
        return events, 0
    important = [e for e in events if e.kind not in LOW_VALUE_KINDS]
    if len(important) <= limit:
        return important, len(events) - len(important)
    # Still too long. Keep every objective — they are the spine of the match —
    # and sample the remaining events EVENLY ACROSS THE MATCH. An earlier
    # version kept the most recent N instead, on the theory that late-game
    # kills explain more; that silently deleted the entire laning and mid game
    # from any bloody match, which is exactly the wrong thing to hide from a
    # reader trying to understand how the game got to where it ended.
    objectives = [e for e in important if e.kind in ("objective", "patron_phase")]
    if len(objectives) >= limit:
        keep = _thin(objectives, limit)
    else:
        others = [e for e in important if e.kind not in ("objective", "patron_phase")]
        keep = objectives + _thin(others, limit - len(objectives))
    keep.sort(key=Event.sort_key)
    return keep, len(events) - len(keep)


def render_json(
    md: MatchData,
    names: Names | None = None,
    *,
    opportunity_analysis: OpportunityAnalysis | None = None,
    focus_hero_ids: Iterable[int] | None = None,
    advantage_ledger: AdvantageLedger | None = None,
    constants: GameConstants | None = None,
    skill_stats: dict[int, PlayerSkillStats] | None = None,
) -> str:
    """Machine-readable sidecar, for a model that wants to compute rather than read."""
    names = names or Names.from_boon()
    constants = constants or load_constants()
    assert md.clock is not None
    opportunities = opportunity_analysis or analyze_opportunities(md, constants=constants)
    focus = set(focus_hero_ids) if focus_hero_ids is not None else None
    ledger = advantage_ledger or analyze_advantage(
        md,
        names,
        focus_hero_ids=tuple(focus) if focus else None,
    )

    def focused(items: Iterable, attr: str = "hero_id") -> list:
        values = list(items)
        if focus is None:
            return values
        return [item for item in values if getattr(item, attr) in focus]

    summary = build_match_summary(
        md,
        names,
        focus=focus,
        fights=analyze_fights(md, names),
        reports=player_reports(md),
        curve=economy_curve(md),
        opportunities=opportunities,
        constants=constants,
        skill_stats=skill_stats,
    )

    payload = {
        # First key on purpose: a model that reads the sidecar top-down gets the
        # ranked answer before the raw material, in the same order as the report.
        "summary": summary.to_dict() if summary else None,
        "match": {
            "map_name": md.map_name,
            "match_id": md.match_id,
            "build": md.build,
            "tick_rate": md.tick_rate,
            "end_tick": md.end_tick,
            "duration": md.clock.mmss(md.end_tick),
            "winning_team_num": md.winning_team_num,
            "teams": {str(t): names.team(t) for t in md.team_nums},
        },
        "phases": [asdict(p) for p in phases(md)],
        "phase_stats": [asdict(s) for s in focused(phase_stats(md))],
        "fight_record_by_phase": fight_record_by_phase(md, names),
        "players": [asdict(r) for r in focused(player_reports(md))],
        "fights": [asdict(f) for f in analyze_fights(md, names)],
        "kills": [
            asdict(c)
            for c in kill_contexts(md, constants)
            if focus is None or c.victim_hero_id in focus or c.attacker_hero_id in focus
        ],
        "economy": [asdict(s) for s in economy_curve(md)],
        "advantage": {
            "focus_team": ledger.focus_team,
            "focus_reason": ledger.focus_reason,
            "streams": [
                {
                    **asdict(s),
                    # Properties are not fields, and the whole point of a stream
                    # is the comparison — so materialise it into the sidecar.
                    "leader": s.leader,
                    "margin": s.margin,
                }
                for s in ledger.streams
            ],
            "windows": [asdict(w) for w in ledger.windows],
            "territory": [{**asdict(t), "share": t.share} for t in ledger.territory],
            "downtime": [asdict(d) for d in ledger.downtime],
            "camp_clears": [asdict(c) for c in ledger.camp_clears],
            "neutral_prizes": [asdict(p) for p in ledger.prizes],
            "urn_runs": [asdict(r) for r in ledger.urn_runs],
            "levers": [asdict(lever) for lever in ledger.levers],
            "caveats": list(ledger.caveats),
        },
        "opportunities": {
            "vision_model": opportunities.vision_model,
            "warning": (
                "Decision windows are estimates, not guaranteed outcomes or a theoretical "
                "maximum kill count."
            ),
            "kill_windows": [
                asdict(w) for w in focused(opportunities.kill_windows, "observer_hero_id")
            ],
            "jungle_windows": [
                asdict(w) for w in focused(opportunities.jungle_windows, "observer_hero_id")
            ],
            "rotation_windows": [
                asdict(w) for w in focused(opportunities.rotation_windows, "observer_hero_id")
            ],
            "macro_windows": [
                asdict(w) for w in focused(opportunities.macro_windows, "observer_hero_id")
            ],
        },
        # Movement constants travel with the sidecar so a model can convert a
        # distance into a travel time itself rather than being told one number.
        "physics": {
            "hu_per_meter": HU_PER_METER,
            "constants_source": constants.source,
            "constants_fetched_at": constants.fetched_at,
            "zipline": asdict(constants.zipline),
            "heroes": {
                str(h): asdict(constants.hero(h))
                for h in (sorted(focus) if focus else md.hero_ids)
                if constants.hero(h)
            },
        },
        "mechanics": {
            str(h): {
                **asdict(stat),
                "accuracy": stat.accuracy,
                "crit_rate": stat.crit_rate,
                "hero_bullet_share": stat.hero_bullet_share,
                "creep_efficiency": stat.creep_efficiency,
                "accuracy_delta": stat.accuracy_delta,
            }
            for h, stat in (skill_stats or {}).items()
            if focus is None or h in focus
        },
        "timeline": [
            {"tick": e.tick, "clock": md.clock.mmss(e.tick), "kind": e.kind, "text": e.text}
            for e in build_timeline(md, names)
        ],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False, default=_jsonable)


def _jsonable(value):
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return list(value)
    return str(value)


def _mechanics(
    md: MatchData,
    names: Names,
    constants: GameConstants,
    stats: dict[int, PlayerSkillStats],
    focus: set[int] | None,
) -> str:
    """Movement physics and per-player mechanical stats.

    Two things a model cannot infer from a kill feed live here. The speed
    ladder makes distances elsewhere in the report interpretable -- "40 metres"
    only means something once you know it is five seconds of walking or two off
    a transit line. The accuracy table supplies the denominator the replay
    itself does not have.
    """
    lines = ["## Mechanics and movement"]

    hero_ids = sorted(focus) if focus else md.hero_ids
    sample_hero = next(
        (constants.hero(h) for h in hero_ids if constants.hero(h)), None
    )
    if sample_hero is not None:
        ladder = reference_speeds(sample_hero, constants.zipline)
        lines += [
            "",
            f"Speed ladder for {sample_hero.name} (m/s). Distances elsewhere in this "
            "report are metres, and travel times assume a straight line with no walls:",
            "",
            "| movement | m/s |",
            "| --- | ---: |",
        ]
        lines += [f"| {label} | {speed:.1f} |" for label, speed in ladder.items()]
        lines += [
            "",
            f"A transit line dismount carries up to "
            f"{constants.zipline.dismount_horizontal_max_percent:.0f}% of line speed, so a "
            f"hero can leave a line at {constants.zipline.dismount_carry_speed:.1f} m/s -- "
            f"{constants.zipline.dismount_carry_speed / sample_hero.max_move_speed:.1f}x "
            "walking. That is why rotation distances read short and gank windows read "
            "long in this game compared with a click-to-move MOBA.",
        ]

    rows = [(h, stats[h]) for h in hero_ids if h in stats]
    if rows:
        lines += [
            "",
            "Weapon mechanics, from the post-match record. Accuracy counts every "
            "target including creeps, so it is only comparable within a hero; the "
            "baseline column is that hero's population median at this lobby's rank.",
            "",
            "| hero | accuracy | vs baseline | shots | on heroes | crit | creeps secured |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for hero_id, stat in sorted(rows, key=lambda kv: -(kv[1].accuracy or 0)):
            delta = stat.accuracy_delta
            lines.append(
                f"| {names.hero(hero_id)} "
                f"| {_pct(stat.accuracy)} "
                f"| {'—' if delta is None else f'{delta:+.0%}'} "
                f"| {stat.shots_taken:,} "
                f"| {_pct(stat.hero_bullet_share)} "
                f"| {_pct(stat.crit_rate)} "
                f"| {_pct(stat.creep_efficiency)} |"
            )
    else:
        lines += [
            "",
            "No post-match record was available for this match, so accuracy, crit rate "
            "and creep efficiency are unknown. The replay records shots that landed but "
            "not shots that missed, so these cannot be recovered from the demo. Do not "
            "read the absence as a low score.",
        ]

    source = {
        "live": "fetched from the Deadlock assets API for this run",
        "cache": "from a recent cached fetch of the Deadlock assets API",
        "bundled": "from the snapshot shipped with this tool, which may predate the patch",
        "defaults": "hardcoded fallbacks; the assets API was unreachable",
    }.get(constants.source, constants.source)
    stamp = f" ({constants.fetched_at})" if constants.fetched_at else ""
    lines += [
        "",
        f"Movement constants are {source}{stamp}. Valve shipped movement, stamina, "
        "Mid Boss and Urn changes in separate 2026 patches, and dash duration now "
        "varies by hero, so figures here track the current build rather than the one "
        "this replay was recorded on.",
    ]
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"
