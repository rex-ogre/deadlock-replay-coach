"""The front page: what went wrong, by how much, and how that compares.

Everything else this tool produces is a reference document. Useful, but it
answers questions in the order the *data* has them, and a player opening a
review wants three answers in the order a *player* has them:

1. What did I do badly, and in which window?
2. What is the actual percentage?
3. What separates me from the players at the top?

So this module builds one ranked, numeric summary and puts it first. It
invents nothing: every line here is a count, a rate, or a population
comparison already established elsewhere in the report, carried up to the top
with a pointer back to the section that holds the working. Where a number has
no honest denominator it is reported as a count and said to be a count --
``3 kill windows`` is not ``3 kills you should have had``.

Findings carry a ``basis`` so the two very different kinds of claim stay
separable: a population comparison is measured against thousands of matches,
while a detector window is this tool's own read of the replay and can be wrong
about a hero's kit. A reader must be able to tell which one they are arguing
with.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .benchmark import MetricComparison, PlayerBenchmark, benchmark_player
from .gamedata import GameConstants, badge_name
from .match import MatchData
from .names import Names
from .opportunities import OpportunityAnalysis
from .skillstats import PlayerSkillStats
from .tactics import (
    EconomySnapshot,
    FightSummary,
    KillContext,
    Phase,
    PlayerReport,
    analyze_fights,
    economy_curve,
    kill_contexts,
    phases,
    player_reports,
)

#: How a finding was established. Kept on every row because the two are not
#: equally strong and a player is entitled to know which they are reading.
BASIS_POPULATION = "population baseline"
BASIS_RECORD = "post-match record"
BASIS_REPLAY = "measured from the replay"
BASIS_DETECTOR = "replay detector (estimate)"

#: What the finding says about the player. COST and MISSING are conclusions
#: from this replay; only MECHANIC is a measured population comparison. Slugs,
#: not prose: these travel into the JSON sidecar, and a client that has to
#: string-match an English sentence to group findings will eventually match
#: the wrong one.
COST = "cost"
MISSING = "missing"
MECHANIC = "mechanic"

KIND_TITLES = {
    COST: "What cost you in this replay",
    MISSING: "Windows this replay says you did not use",
    MECHANIC: "Measured gaps against the top rank",
}

MAX_FINDINGS = 8
MAX_EVIDENCE = 4


@dataclass(frozen=True)
class Rate:
    """A percentage that has a real denominator behind it."""

    key: str
    label: str
    numerator: float
    denominator: float
    note: str = ""

    @property
    def value(self) -> float | None:
        return self.numerator / self.denominator if self.denominator else None

    def text(self) -> str:
        if self.value is None:
            return f"{self.label}: unknown"
        counts = f"{self.numerator:,.0f}/{self.denominator:,.0f}"
        note = f" — {self.note}" if self.note else ""
        return f"{self.label}: **{self.value:.0%}** ({counts}){note}"


@dataclass(frozen=True)
class Finding:
    """One ranked thing to fix, with the number that makes it arguable."""

    kind: str
    key: str
    headline: str
    basis: str
    section: str
    weight: float
    rate: Rate | None = None
    evidence: tuple[str, ...] = ()
    detail: str = ""
    #: How many times this happened, where the finding is a count rather than a
    #: rate. Carried separately from the headline so a client can render the
    #: number in its own language instead of parsing English prose.
    count: int | None = None

    def text(self) -> str:
        parts = [self.headline]
        if self.evidence:
            parts.append("At " + ", ".join(self.evidence) + ".")
        if self.detail:
            parts.append(self.detail)
        return " ".join(parts)


@dataclass(frozen=True)
class MatchShape:
    """The one paragraph that says what kind of match this was."""

    won: bool | None
    team_name: str
    opponent_name: str
    final_gap: int
    worst_gap: int
    worst_clock: str
    turning_phase: str | None
    fights_by_phase: dict[str, tuple[int, int]] = field(default_factory=dict)

    def text(self) -> str:
        if self.won is None:
            outcome = "The match has no recorded winner"
        elif self.won:
            outcome = f"{self.team_name} won"
        else:
            outcome = f"{self.team_name} lost"
        if not self.final_gap:
            gap = f"level on souls with {self.opponent_name}"
        else:
            gap = (
                f"finishing {abs(self.final_gap):,} souls "
                f"{'ahead of' if self.final_gap > 0 else 'behind'} {self.opponent_name}"
            )
        turn = ""
        if self.turning_phase:
            record = self.fights_by_phase.get(self.turning_phase)
            score = f" ({record[0]}-{record[1]} on decisive fights)" if record else ""
            # Only a team that was actually behind has a deepest point; quoting
            # a "worst deficit" of zero to a team that never trailed is noise.
            deepest = (
                f", and the deficit was deepest at {self.worst_clock} "
                f"({abs(self.worst_gap):,} souls)"
                if self.worst_gap < 0
                else ""
            )
            turn = f" The {self.turning_phase} is where it went{score}{deepest}."
        return f"{outcome}, {gap}.{turn}"


@dataclass(frozen=True)
class QuickSummary:
    """Everything the first screen of a review needs, and nothing else."""

    hero_id: int | None
    hero_name: str
    player_name: str | None
    badge: int | None
    shape: MatchShape
    rates: tuple[Rate, ...]
    findings: tuple[Finding, ...]
    benchmark: PlayerBenchmark | None
    unavailable: tuple[str, ...] = ()

    @property
    def costs(self) -> list[Finding]:
        return [f for f in self.findings if f.kind == COST]

    @property
    def missing(self) -> list[Finding]:
        return [f for f in self.findings if f.kind == MISSING]

    @property
    def mechanics(self) -> list[Finding]:
        return [f for f in self.findings if f.kind == MECHANIC]

    def to_dict(self) -> dict:
        return {
            "hero_id": self.hero_id,
            "hero": self.hero_name,
            "player": self.player_name,
            "badge": self.badge,
            "badge_name": badge_name(self.badge) if self.badge else None,
            "shape": {
                "won": self.shape.won,
                "team": self.shape.team_name,
                "opponent": self.shape.opponent_name,
                "final_gap": self.shape.final_gap,
                "worst_gap": self.shape.worst_gap,
                "worst_clock": self.shape.worst_clock,
                "turning_phase": self.shape.turning_phase,
                "fights_by_phase": {
                    phase: list(counts) for phase, counts in self.shape.fights_by_phase.items()
                },
                "text": self.shape.text(),
            },
            "rates": [
                {
                    "key": rate.key,
                    "label": rate.label,
                    "value": rate.value,
                    "numerator": rate.numerator,
                    "denominator": rate.denominator,
                    "note": rate.note,
                }
                for rate in self.rates
            ],
            "findings": [
                {
                    "kind": finding.kind,
                    "key": finding.key,
                    "headline": finding.headline,
                    "detail": finding.detail,
                    "basis": finding.basis,
                    "section": finding.section,
                    "count": finding.count,
                    "evidence": list(finding.evidence),
                    "rate": (
                        {
                            "value": finding.rate.value,
                            "numerator": finding.rate.numerator,
                            "denominator": finding.rate.denominator,
                        }
                        if finding.rate
                        else None
                    ),
                }
                for finding in self.findings
            ],
            "benchmark": _benchmark_dict(self.benchmark),
            "unavailable": list(self.unavailable),
        }


def _benchmark_dict(benchmark: PlayerBenchmark | None) -> dict | None:
    if benchmark is None or not benchmark.available:
        return None
    return {
        "peer": {
            "badge": benchmark.peer.badge if benchmark.peer else None,
            "label": benchmark.peer.label if benchmark.peer else None,
            "matches": benchmark.peer.matches if benchmark.peer else 0,
        },
        "top": {
            "badge": benchmark.top.badge if benchmark.top else None,
            "label": benchmark.top.label if benchmark.top else None,
            "matches": benchmark.top.matches if benchmark.top else 0,
        },
        "metrics": [
            {
                "key": metric.key,
                "label": metric.label,
                "unit": metric.unit,
                "reads_as": metric.reads_as,
                "player": metric.player,
                "peer": metric.peer,
                "top": metric.top,
                "verdict": metric.verdict,
                "relative_to_top": metric.relative_to_top,
                "higher_is_better": metric.higher_is_better,
            }
            for metric in benchmark.metrics
        ],
    }


# ------------------------------------------------------------------ building


def build_summary(
    md: MatchData,
    names: Names,
    *,
    hero_id: int | None,
    fights: list[FightSummary],
    reports: list[PlayerReport],
    kills: list[KillContext],
    phase_list: list[Phase],
    curve: list[EconomySnapshot],
    opportunities: OpportunityAnalysis,
    constants: GameConstants,
    skill_stats: dict[int, PlayerSkillStats] | None = None,
) -> QuickSummary:
    """Rank what this player did against what their hero's population does."""
    assert md.clock is not None
    stat = (skill_stats or {}).get(hero_id) if hero_id is not None else None
    benchmark = benchmark_player(stat, constants) if stat else None
    report = next((r for r in reports if r.hero_id == hero_id), None)
    team = md.team_of(hero_id) if hero_id is not None else None

    shape = _shape(md, names, team, fights, phase_list, curve)
    rates = _rates(md, team, hero_id, report, fights, kills, stat)
    findings = _findings(
        md,
        names,
        hero_id=hero_id,
        team=team,
        report=report,
        fights=fights,
        kills=kills,
        opportunities=opportunities,
        benchmark=benchmark,
        stat=stat,
    )

    unavailable = []
    if stat is None:
        unavailable.append(
            "No post-match record for this match, so accuracy, deny rate and every "
            "population comparison are unknown. The replay records shots that landed "
            "but not shots that missed; this is missing data, not a low score."
        )
    elif benchmark is not None and not benchmark.available:
        unavailable.append(
            f"No population sample for {names.hero(hero_id)} at this rank, so the "
            "comparison columns are unknown."
        )

    return QuickSummary(
        hero_id=hero_id,
        hero_name=names.hero(hero_id) if hero_id is not None else "the match",
        player_name=md.player_name(hero_id) if hero_id is not None else None,
        badge=stat.badge if stat else None,
        shape=shape,
        rates=tuple(rates),
        findings=tuple(findings[:MAX_FINDINGS]),
        benchmark=benchmark,
        unavailable=tuple(unavailable),
    )


def build_summaries(
    md: MatchData,
    names: Names,
    hero_ids: Iterable[int],
    *,
    opportunities: OpportunityAnalysis,
    constants: GameConstants,
    skill_stats: dict[int, PlayerSkillStats] | None = None,
) -> dict[int, QuickSummary]:
    """One summary per hero, sharing the expensive analysis between them.

    The web flow decodes a replay before the user has said which hero they
    played, so every seat is summarised up front and the one they pick is
    served later. Fights, deaths and windows are computed once here rather
    than twelve times.
    """
    fights = analyze_fights(md, names)
    reports = player_reports(md)
    kills = kill_contexts(md, constants)
    phase_list = phases(md)
    curve = economy_curve(md)
    return {
        hero_id: build_summary(
            md,
            names,
            hero_id=hero_id,
            fights=fights,
            reports=reports,
            kills=kills,
            phase_list=phase_list,
            curve=curve,
            opportunities=opportunities,
            constants=constants,
            skill_stats=skill_stats,
        )
        for hero_id in hero_ids
    }


def _shape(
    md: MatchData,
    names: Names,
    team: int | None,
    fights: list[FightSummary],
    phase_list: list[Phase],
    curve: list[EconomySnapshot],
) -> MatchShape:
    others = [t for t in md.team_nums if t != team]
    opponent = others[0] if others else None
    final_gap = 0
    worst_gap = 0
    worst_clock = "--:--"
    if curve and team is not None:
        last = curve[-1].net_worth_by_team
        final_gap = int(last.get(team, 0)) - int(last.get(opponent, 0) if opponent else 0)
        worst = min(
            curve,
            key=lambda s: int(s.net_worth_by_team.get(team, 0))
            - int(s.net_worth_by_team.get(opponent, 0) if opponent else 0),
        )
        worst_gap = int(worst.net_worth_by_team.get(team, 0)) - int(
            worst.net_worth_by_team.get(opponent, 0) if opponent else 0
        )
        assert md.clock is not None
        worst_clock = md.clock.mmss(worst.tick)

    by_phase: dict[str, tuple[int, int]] = {}
    for phase in phase_list:
        mine = sum(1 for f in fights if f.phase == phase.name and f.winner == team)
        theirs = sum(1 for f in fights if f.phase == phase.name and f.winner == opponent)
        by_phase[phase.name] = (mine, theirs)

    # The phase this player's team lost hardest is the one worth opening the
    # review with; a phase they merely drew is not the story.
    turning = None
    worst_margin = 0
    for phase_name, (mine, theirs) in by_phase.items():
        margin = mine - theirs
        if margin < worst_margin:
            worst_margin, turning = margin, phase_name

    return MatchShape(
        won=None if md.winning_team_num is None else md.winning_team_num == team,
        team_name=names.team(team) if team is not None else "your team",
        opponent_name=names.team(opponent) if opponent is not None else "the opponent",
        final_gap=final_gap,
        worst_gap=worst_gap,
        worst_clock=worst_clock,
        turning_phase=turning,
        fights_by_phase=by_phase,
    )


def _rates(
    md: MatchData,
    team: int | None,
    hero_id: int | None,
    report: PlayerReport | None,
    fights: list[FightSummary],
    kills: list[KillContext],
    stat: PlayerSkillStats | None,
) -> list[Rate]:
    """Only percentages whose denominator is a real, counted population."""
    rates: list[Rate] = []
    if stat and stat.shots_taken:
        rates.append(
            Rate(
                key="accuracy",
                label="Weapon accuracy",
                numerator=stat.shots_hit,
                denominator=stat.shots_taken,
                note=f"{stat.shots_taken:,} shots taken",
            )
        )
    if stat and stat.hero_bullets_hit and stat.shots_hit:
        rates.append(
            Rate(
                key="hero_bullet_share",
                label="Landed shots that hit a hero",
                numerator=stat.hero_bullets_hit,
                denominator=stat.shots_hit,
                note="the rest hit creeps",
            )
        )
    if stat and stat.possible_creeps:
        rates.append(
            Rate(
                key="creep_efficiency",
                label="Available creeps secured",
                numerator=stat.creep_kills,
                denominator=stat.possible_creeps,
            )
        )

    won = [f for f in fights if f.winner == team] if team is not None else []
    if won:
        rates.append(
            Rate(
                key="fight_conversion",
                label="Won fights converted into an objective",
                numerator=sum(1 for f in won if f.converted_into),
                denominator=len(won),
                note="within 45 seconds of the fight ending",
            )
        )

    deaths = [c for c in kills if hero_id is not None and c.victim_hero_id == hero_id]
    scored = [c for c in deaths if c.outnumbered is not None]
    if scored:
        rates.append(
            Rate(
                key="pickoff_share",
                label="Your deaths that were pickoffs",
                numerator=sum(1 for c in scored if c.outnumbered),
                denominator=len(scored),
                note="attackers outnumbered your nearby living allies by 2+",
            )
        )
    if report is not None and team is not None:
        team_kills = sum(
            1
            for row in md.kills.iter_rows(named=True)
            if md.team_of(row.get("victim_hero_id")) not in (None, team)
        )
        if team_kills:
            rates.append(
                Rate(
                    key="kill_participation",
                    label="Kill participation",
                    numerator=report.kills + report.assists,
                    denominator=team_kills,
                    note="your team's kills you got a kill or assist on",
                )
            )
    return rates


def _findings(
    md: MatchData,
    names: Names,
    *,
    hero_id: int | None,
    team: int | None,
    report: PlayerReport | None,
    fights: list[FightSummary],
    kills: list[KillContext],
    opportunities: OpportunityAnalysis,
    benchmark: PlayerBenchmark | None,
    stat: PlayerSkillStats | None,
) -> list[Finding]:
    assert md.clock is not None
    clock = md.clock.mmss
    out: list[Finding] = []

    # -- macro first: one unconverted fight is worth more souls than a season
    # of aim practice, which is the whole ordering rule of this report.
    if team is not None:
        won = [f for f in fights if f.winner == team]
        converted = [f for f in won if f.converted_into]
        pushable = [
            f
            for f in won
            if not f.converted_into
            and f.conversion_assessment is not None
            and f.conversion_assessment.status == "push_now"
        ]
        opponents = [t for t in md.team_nums if t != team]
        their_won = [f for f in fights if opponents and f.winner == opponents[0]]
        their_converted = sum(1 for f in their_won if f.converted_into)
        if won and pushable:
            comparison = ""
            if their_won:
                comparison = (
                    f" {names.team(opponents[0])} converted "
                    f"{their_converted}/{len(their_won)} of theirs."
                )
            out.append(
                Finding(
                    kind=MISSING,
                    key="fight_conversion",
                    headline=(
                        f"Won {len(won)} fights, turned {len(converted)} into an objective. "
                        f"{len(pushable)} of the unconverted wins had a lane wave already "
                        f"exposing a structure — the report calls those pushable on the spot."
                        f"{comparison}"
                    ),
                    basis=BASIS_REPLAY,
                    section="Teamfights",
                    weight=100,
                    count=len(won),
                    rate=Rate(
                        key="fight_conversion",
                        label="Won fights converted",
                        numerator=len(converted),
                        denominator=len(won),
                    ),
                    evidence=tuple(clock(f.start_tick) for f in pushable[:MAX_EVIDENCE]),
                    detail=(
                        "Winning a fight and spending it are scored separately on purpose: "
                        "a won fight that buys nothing is the most expensive repeated mistake "
                        "in this game."
                    ),
                )
            )

    # -- how you died
    deaths = [c for c in kills if hero_id is not None and c.victim_hero_id == hero_id]
    pickoffs = [c for c in deaths if c.outnumbered]
    if pickoffs:
        reachable = [
            c for c in pickoffs if c.support_seconds is not None and c.support_seconds <= 5.0
        ]
        detail = (
            f"{len(reachable)} of those had a teammate within 5 seconds of you at full "
            "speed, so the shape was reachable help you did not wait for."
            if reachable
            else "In each of these the nearest teammate was too far to arrive at any speed, "
            "which makes it a team-shape problem before it is a positioning one."
        )
        out.append(
            Finding(
                kind=COST,
                key="pickoffs",
                headline=(
                    f"{len(pickoffs)} of your {len(deaths)} "
                    f"{'death was' if len(deaths) == 1 else 'deaths were'} a pickoff: the "
                    f"attackers outnumbered your nearby living allies by 2 or more."
                ),
                basis=BASIS_REPLAY,
                section="Kill patterns",
                weight=90,
                count=len(pickoffs),
                rate=Rate(
                    key="pickoff_share",
                    label="Deaths that were pickoffs",
                    numerator=len(pickoffs),
                    denominator=len(deaths),
                ),
                evidence=tuple(clock(c.tick) for c in pickoffs[:MAX_EVIDENCE]),
                detail=detail,
            )
        )

    # -- windows the map offered and nobody spent
    mine = _observer_windows(opportunities, hero_id)
    macro = mine["macro"]
    if macro:
        kinds: dict[str, int] = {}
        for window in macro:
            kinds[window.kind] = kinds.get(window.kind, 0) + 1
        breakdown = ", ".join(f"{count} {kind.replace('_', ' ')}" for kind, count in kinds.items())
        out.append(
            Finding(
                kind=MISSING,
                key="macro_windows",
                headline=(
                    f"{len(macro)} map windows passed unused ({breakdown}). These are read "
                    f"from announced information only — the kill feed, fallen structures, "
                    f"Mid Boss and Urn events — so no vision model is involved."
                ),
                basis=BASIS_DETECTOR,
                section="Player-perspective opportunities",
                weight=80,
                count=len(macro),
                evidence=tuple(
                    f"{clock(w.start_tick)} ({w.kind.replace('_', ' ')})" for w in macro[:MAX_EVIDENCE]
                ),
            )
        )

    rotations = mine["rotation"]
    if rotations:
        out.append(
            Finding(
                kind=MISSING,
                key="rotation_windows",
                headline=(
                    f"{len(rotations)} cross-lane fights were reachable in time and you did "
                    f"not go. Each one is a fight a teammate lost while you were free, "
                    f"healthy and out of combat."
                ),
                basis=BASIS_DETECTOR,
                section="Player-perspective opportunities",
                weight=70,
                count=len(rotations),
                evidence=tuple(
                    f"{clock(w.start_tick)} ({w.estimated_travel_seconds:.0f}s away)"
                    for w in rotations[:MAX_EVIDENCE]
                ),
                detail="Travel time is estimated from your own movement in this replay; "
                "route geometry and mobility abilities are not modelled.",
            )
        )

    kill_windows = mine["kill"]
    if kill_windows:
        out.append(
            Finding(
                kind=MISSING,
                key="kill_windows",
                headline=(
                    f"{len(kill_windows)} kill windows went unfinished — an opponent below "
                    f"35% health, inside your range, with the local count in your favour."
                ),
                basis=BASIS_DETECTOR,
                section="Player-perspective opportunities",
                weight=60,
                count=len(kill_windows),
                evidence=tuple(
                    f"{clock(w.start_tick)} vs {names.hero(w.target_hero_id)} "
                    f"({w.target_health_pct:.0%} hp)"
                    for w in kill_windows[:MAX_EVIDENCE]
                ),
                detail="These are independent decision windows, not kills you were owed: "
                "aim, ammo and opponent reactions are not modelled. Do not add them to "
                "your actual kill count.",
            )
        )

    invades = [w for w in mine["jungle"] if w.camp_status != "unknown — scout before committing"]
    if invades:
        out.append(
            Finding(
                kind=MISSING,
                key="invade_windows",
                headline=(
                    f"{len(invades)} enemy camps were open with every known opponent far "
                    f"away, and went untaken."
                ),
                basis=BASIS_DETECTOR,
                section="Player-perspective opportunities",
                weight=40,
                count=len(invades),
                evidence=tuple(clock(w.start_tick) for w in invades[:MAX_EVIDENCE]),
            )
        )

    # -- mechanics last, because they move slowest
    if benchmark is not None and benchmark.available and benchmark.top is not None:
        top_label = benchmark.top.label
        deficits = benchmark.deficits(limit=3)
        # Accuracy is the one mechanical number that is both cleanly measured
        # and strongly rank-correlated, and it is the number players ask about
        # by name. If it is a real deficit it goes in the list whatever else
        # outranked it.
        accuracy = benchmark.by_key("accuracy")
        if accuracy is not None and accuracy.severity and accuracy not in deficits:
            deficits.append(accuracy)
        for metric in deficits:
            out.append(
                Finding(
                    kind=MECHANIC,
                    key=metric.key,
                    headline=(
                        f"{metric.label}: {metric.format(metric.player)} against "
                        f"{metric.format(metric.top)} for {names.hero(benchmark.hero_id)} "
                        f"players at {top_label} "
                        f"({abs(metric.relative_to_top or 0):.0%} behind, "
                        f"{benchmark.top.matches:,} matches)."
                    ),
                    basis=BASIS_POPULATION,
                    section="Mechanics and movement",
                    weight=30 + min(20.0, metric.severity * 40),
                    detail=metric.reads_as.capitalize() + ".",
                )
            )

    out.sort(key=lambda f: -f.weight)
    return out


def _observer_windows(analysis: OpportunityAnalysis, hero_id: int | None) -> dict[str, list]:
    def owned(items: Iterable) -> list:
        return [
            w
            for w in items
            if hero_id is None or getattr(w, "observer_hero_id", None) == hero_id
        ]

    return {
        "kill": owned(analysis.kill_windows),
        "rotation": owned(analysis.rotation_windows),
        "macro": owned(analysis.macro_windows),
        "jungle": owned(analysis.jungle_windows),
    }


# ------------------------------------------------------------------ markdown


def render_summary_section(summary: QuickSummary) -> str:
    """The report's opening section: the answer before the working."""
    who = summary.hero_name
    if summary.player_name:
        who = f"{summary.hero_name} ({summary.player_name})"
    lines = [
        "## Bottom line",
        "",
        "The ranked answer for the focused player. Every number below is repeated "
        "with its working in a later section, named on each line. Nothing here is "
        "new evidence.",
        "",
        f"**{who}** — {summary.shape.text()}",
    ]

    if summary.rates:
        lines += ["", "Rates with a real denominator:", ""]
        lines += [f"- {rate.text()}" for rate in summary.rates]

    benchmark = summary.benchmark
    if benchmark is not None and benchmark.available:
        peer = benchmark.peer
        top = benchmark.top
        peer_head = (
            f"{peer.label} ({peer.matches:,} matches)" if peer else "your rank (no sample)"
        )
        top_head = f"{top.label} ({top.matches:,} matches)" if top else "top rank (no sample)"
        lines += [
            "",
            f"### {benchmark_heading(benchmark)}",
            "",
            "Every row is a ratio, so match length cancels and a short high-rank "
            "match compares cleanly with a long one. Rows are this hero only — "
            "accuracy and damage do not transfer between heroes. `style` marks a "
            "metric with no better direction: it explains what kind of player "
            "someone is, not how good they are.",
            "",
            f"| metric | you | {peer_head} | {top_head} | read |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for metric in benchmark.metrics:
            if metric.player is None:
                continue
            lines.append(
                f"| {metric.label} | {metric.format(metric.player)} "
                f"| {metric.format(metric.peer)} | {metric.format(metric.top)} "
                f"| {metric.verdict} |"
            )
        lines += [
            "",
            "Population rows are the summed post-match counters of every match in "
            "that (hero, rank) bucket, divided through — a weighted mean, not a "
            "median, and not filtered by role or lane.",
        ]

    for kind, group in (
        (COST, summary.costs),
        (MISSING, summary.missing),
        (MECHANIC, summary.mechanics),
    ):
        title = KIND_TITLES[kind]
        if not group:
            continue
        lines += ["", f"### {title}", ""]
        for finding in group:
            rate = f" [{finding.rate.numerator:,.0f}/{finding.rate.denominator:,.0f}]" if finding.rate else ""
            lines.append(f"- **{finding.headline}**{rate}")
            if finding.evidence:
                lines.append(f"  - When: {', '.join(finding.evidence)}")
            if finding.detail:
                lines.append(f"  - {finding.detail}")
            lines.append(f"  - Basis: {finding.basis}; working in *{finding.section}*.")

    if summary.unavailable:
        lines += ["", "Not established by this match:", ""]
        lines += [f"- {note}" for note in summary.unavailable]

    return "\n".join(lines)


def benchmark_heading(benchmark: PlayerBenchmark) -> str:
    if benchmark.top is not None:
        return f"You against {benchmark.top.label} on this hero"
    return "You against this hero's population"


def worst_metric(benchmark: PlayerBenchmark | None) -> MetricComparison | None:
    """The single metric furthest behind the top rank, for a one-line answer."""
    if benchmark is None:
        return None
    deficits = benchmark.deficits(limit=1)
    return deficits[0] if deficits else None
