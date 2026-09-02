"""Where the advantage was actually compounded — and where it could have been.

Every other section of the report answers *what happened*. This one answers the
only question a player can act on next game: **how do we win this one**.

Deadlock is not decided by a single play. It is decided by which team compounds
resources faster, and there are exactly two ways to compound: gather more of
your own, and take away more of theirs. So everything here is modelled as a
*stream* — something that produced souls over time — and every stream is
reported for both teams, because a stream you cannot compare is not a lever.

The levers at the end are the streams with the largest measurable gap, ranked
by the souls at stake, each carrying the arithmetic that produced it. Where the
souls cannot be derived honestly (a neutral camp's bounty is not in the demo)
the lever carries a count and says so rather than inventing a number.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from math import hypot

import polars as pl

from .match import MatchData
from .names import Names
from .opportunities import Camp, neutral_camps, team_bases
from .tactics import (
    EconomySnapshot,
    analyze_fights,
    economy_curve,
    fixed_windows,
    phase_stats,
    player_reports,
)

# Five minutes, not the ten used by `match_arc`. A ten-minute bucket can hide a
# lane going even for five minutes and collapsing for the next five, which is
# precisely the resolution a "where did it slip" question needs.
LEDGER_WINDOW_SECONDS = 300.0

# A camp clear is credited to the team with a living hero closest to it. Beyond
# this range nobody is credited — an unattributed clear is more useful than a
# confident wrong one.
CAMP_CREDIT_RANGE = 1_800.0
CAMP_CREDIT_MAX_GAP_SECONDS = 2.0

# Neutral camps never report zero health in this build (verified on match
# 98811241: 4,325 neutral rows, minimum health 1). A cleared camp instead goes
# quiet on low health and reappears minutes later at a *higher* max health,
# because camps scale with match time. So a clear is "last seen on low health,
# then silent for longer than any regen would take".
CAMP_KILL_HEALTH_SHARE = 0.3
CAMP_RESPAWN_MIN_SECONDS = 120.0
CAMP_RESPAWN_HEALTH_SHARE = 0.9
CAMP_CLEAR_MERGE_SECONDS = 15.0

# An urn run that ends with the carrier dying inside this window was ended by
# the enemy, not fumbled.
URN_DENIAL_SECONDS = 5.0

# The urn resetting to its spawn emits `returned` and then, one tick later, a
# `picked_up` naming the same hero — an artifact of the carrier field being
# re-read, not a player reaching for it again. Counting those as runs inflated
# the pickup totals by roughly a third on match 12345678.
URN_RESET_TICKS = 5

# `killed` on the mid boss carries the boss's own neutral team; the buff `used`
# rows that follow name the team that actually took it.
MID_BOSS_USE_SECONDS = 90.0

# A death with no reappearance inside this window is not measured rather than
# charged the whole rest of the match — the last death of a game never respawns.
MAX_RESPAWN_SECONDS = 240.0

MAX_LEVERS = 6


# ------------------------------------------------------------------ types


@dataclass(frozen=True)
class StreamTotals:
    """One accumulable resource, both teams side by side.

    ``estimated`` is load-bearing: a reader who cannot tell a counted structure
    from an inferred camp clear will over-trust the inferred one.
    """

    key: str
    label: str
    by_team: dict[int, float | None]
    unit: str = ""
    higher_is_better: bool = True
    estimated: bool = False
    source: str = "measured"

    @property
    def leader(self) -> int | None:
        known = {t: v for t, v in self.by_team.items() if v is not None}
        if len(known) < 2:
            return None
        ranked = sorted(known.items(), key=lambda kv: kv[1], reverse=self.higher_is_better)
        if ranked[0][1] == ranked[1][1]:
            return None
        return ranked[0][0]

    @property
    def margin(self) -> float | None:
        known = [v for v in self.by_team.values() if v is not None]
        if len(known) < 2:
            return None
        return abs(max(known) - min(known))


@dataclass(frozen=True)
class CampClear:
    """One neutral camp going from alive to dead, and our best guess at who did it."""

    tick: int
    camp_id: int
    x: float
    y: float
    half_of: int | None  # whose side of the map the camp sits on
    team_num: int | None  # who cleared it (estimated by proximity)
    distance: float | None

    @property
    def is_raid(self) -> bool:
        return (
            self.team_num is not None
            and self.half_of is not None
            and self.team_num != self.half_of
        )


@dataclass(frozen=True)
class NeutralPrize:
    """A contested map reward that resolved for somebody: mid boss, rift, urn."""

    kind: str
    tick: int
    team_num: int | None
    detail: str


@dataclass(frozen=True)
class UrnRun:
    pickup_tick: int
    hero_id: int | None
    team_num: int | None
    end_tick: int | None
    ended_by: str
    denied_by: int | None


@dataclass(frozen=True)
class Territory:
    """How much of the match a team spent standing on the opponent's ground.

    This is the least speculative denial metric in the report: it needs no
    vision model, no attribution and no soul table — only where twelve heroes
    were standing, which is sampled directly. A team farming inside the enemy
    half is taking resources the enemy then cannot take.
    """

    team_num: int
    samples: int
    enemy_half_samples: int
    kills_in_enemy_half: int
    deaths_in_own_half: int

    @property
    def share(self) -> float | None:
        return self.enemy_half_samples / self.samples if self.samples else None


@dataclass(frozen=True)
class Downtime:
    """Time spent dead, priced at the player's own earning rate.

    This is the resource nobody counts. A player who dies eight times has not
    lost eight kills' worth of souls — they have lost eight kills *plus* every
    soul they would have earned during four minutes of lying on the floor.
    """

    hero_id: int
    team_num: int | None
    player_name: str | None
    deaths: int
    measured_deaths: int
    seconds: float
    souls_per_min: float | None
    souls_forgone: int | None


@dataclass(frozen=True)
class WindowLedger:
    """One clock window, scored on every stream at once."""

    label: str
    start_tick: int
    end_tick: int
    minutes: float
    net_worth_by_team: dict[int, int]
    gained_by_team: dict[int, int]
    per_min_by_team: dict[int, float]
    kills_by_team: dict[int, int]
    structures_by_team: dict[int, int]
    camps_by_team: dict[int, int]
    raids_by_team: dict[int, int]
    prizes_by_team: dict[int, int]
    leader: int | None
    margin: int
    driver: str


@dataclass(frozen=True)
class Lever:
    """A ranked, actionable place the advantage was available.

    ``souls_at_stake`` is None when the demo does not carry the soul value of
    the resource (camp bounties, for one). The lever still ships — with a count
    instead of a number that would have been made up.
    """

    key: str
    title: str
    team_num: int | None
    souls_at_stake: int | None
    confidence: str
    evidence: tuple[str, ...]
    action: str


@dataclass(frozen=True)
class AdvantageLedger:
    teams: tuple[int, ...] = ()
    focus_team: int | None = None
    focus_reason: str = ""
    streams: tuple[StreamTotals, ...] = ()
    windows: tuple[WindowLedger, ...] = ()
    territory: tuple[Territory, ...] = ()
    downtime: tuple[Downtime, ...] = ()
    camp_clears: tuple[CampClear, ...] = ()
    prizes: tuple[NeutralPrize, ...] = ()
    urn_runs: tuple[UrnRun, ...] = ()
    levers: tuple[Lever, ...] = ()
    caveats: tuple[str, ...] = ()


# ------------------------------------------------------------ entry point


def analyze_advantage(
    md: MatchData,
    names: Names | None = None,
    *,
    window_seconds: float = LEDGER_WINDOW_SECONDS,
    focus_hero_ids: tuple[int, ...] | None = None,
) -> AdvantageLedger:
    """Decompose the match into resource streams and rank the recoverable gaps."""
    names = names or Names.from_boon()
    teams = tuple(md.team_nums)
    if len(teams) != 2:
        # One-sided or spectator-only data. Everything downstream is a
        # comparison between two teams, so there is nothing honest to say.
        return AdvantageLedger(
            teams=teams,
            caveats=("Advantage ledger needs exactly two playing teams; skipped.",),
        )

    assert md.clock is not None
    curve = economy_curve(md)
    reports = player_reports(md)
    clears = tuple(camp_clears(md))
    prizes = tuple(neutral_prizes(md))
    runs = tuple(urn_runs(md))
    dead_time = tuple(death_downtime(md, curve))
    ground = tuple(territory(md, teams))

    focus_team, focus_reason = _orient(md, teams, focus_hero_ids)

    streams = tuple(
        _streams(md, teams, curve, reports, clears, prizes, runs, dead_time, ground)
    )
    windows = tuple(
        _windows(md, teams, curve, clears, prizes, window_seconds)
    )
    levers = tuple(
        _levers(
            md, names, teams, focus_team, reports, windows, dead_time, clears, prizes, ground
        )
    )
    return AdvantageLedger(
        teams=teams,
        focus_team=focus_team,
        focus_reason=focus_reason,
        streams=streams,
        windows=windows,
        territory=ground,
        downtime=dead_time,
        camp_clears=clears,
        prizes=prizes,
        urn_runs=runs,
        levers=levers,
        caveats=_caveats(md, clears, runs, dead_time),
    )


def _orient(
    md: MatchData,
    teams: tuple[int, ...],
    focus_hero_ids: tuple[int, ...] | None,
) -> tuple[int | None, str]:
    """Whose side the levers are written for.

    A requested player wins; otherwise the team that lost, because that is the
    team for whom "how do we win this" has an answer that changes anything.
    """
    if focus_hero_ids:
        for hero in focus_hero_ids:
            team = md.team_of(hero)
            if team in teams:
                return team, "the requested player's team"
    if md.winning_team_num in teams:
        loser = [t for t in teams if t != md.winning_team_num]
        if loser:
            return loser[0], "the losing team"
    return None, ""


# --------------------------------------------------------------- streams


def _streams(
    md: MatchData,
    teams: tuple[int, ...],
    curve: list[EconomySnapshot],
    reports,
    clears: tuple[CampClear, ...],
    prizes: tuple[NeutralPrize, ...],
    runs: tuple[UrnRun, ...],
    dead_time: tuple[Downtime, ...],
    ground: tuple[Territory, ...],
) -> list[StreamTotals]:
    assert md.clock is not None
    minutes = max(md.clock.seconds(md.end_tick) / 60.0, 1e-9)

    def per_team(fn) -> dict[int, float | None]:
        return {team: fn(team) for team in teams}

    def sum_reports(attr: str):
        def inner(team: int) -> float | None:
            values = [
                getattr(r, attr) for r in reports if r.team_num == team and getattr(r, attr) is not None
            ]
            return float(sum(values)) if values else None

        return inner

    final = curve[-1].net_worth_by_team if curve else {}
    first = curve[0].net_worth_by_team if curve else {}

    kills_by_team = _kills_by_team(md, teams, 0, md.end_tick)
    structures = _structures_by_team(md, teams, 0, md.end_tick)

    camp_data_available = not md.neutrals.is_empty()

    def camps(team: int, raids_only: bool = False) -> float | None:
        if not camp_data_available:
            return None
        return float(
            sum(1 for c in clears if c.team_num == team and (c.is_raid or not raids_only))
        )

    def conceded_camps(team: int) -> float | None:
        """Camps taken out of *your* half by them — resources you were farmed off."""
        if not camp_data_available:
            return None
        return float(sum(1 for c in clears if c.half_of == team and c.is_raid))

    def prize_count(team: int, kind: str) -> float:
        return float(sum(1 for p in prizes if p.team_num == team and p.kind == kind))

    def urn_starts(team: int) -> float:
        return float(sum(1 for r in runs if r.team_num == team))

    def urn_denials(team: int) -> float:
        return float(sum(1 for r in runs if r.denied_by == team))

    def dead_seconds(team: int) -> float | None:
        rows = [d for d in dead_time if d.team_num == team]
        return float(sum(d.seconds for d in rows)) if rows else None

    by_team_ground = {t.team_num: t for t in ground}

    def enemy_ground(team: int) -> float | None:
        row = by_team_ground.get(team)
        return round(row.share * 100, 1) if row and row.share is not None else None

    def enemy_ground_kills(team: int) -> float | None:
        row = by_team_ground.get(team)
        return float(row.kills_in_enemy_half) if row else None

    out = [
        StreamTotals(
            key="net_worth",
            label="Net worth (final)",
            by_team=per_team(lambda t: float(final.get(t)) if t in final else None),
            unit="souls",
        ),
        StreamTotals(
            key="souls_per_min",
            label="Souls per minute (team total)",
            by_team=per_team(
                lambda t: round((final.get(t, 0) - first.get(t, 0)) / minutes, 1)
                if t in final
                else None
            ),
            unit="souls/min",
        ),
        StreamTotals(
            key="last_hits",
            label="Lane last hits",
            by_team=per_team(sum_reports("last_hits")),
        ),
        StreamTotals(
            key="denies",
            label="Denies (souls taken off them in lane)",
            by_team=per_team(sum_reports("denies")),
        ),
        StreamTotals(
            key="camps",
            label="Neutral camps cleared",
            by_team=per_team(lambda t: camps(t)),
            estimated=True,
            source=f"estimated — nearest living hero within {CAMP_CREDIT_RANGE:,.0f} units",
        ),
        StreamTotals(
            key="raids",
            label="— of those, inside the enemy half",
            by_team=per_team(lambda t: camps(t, raids_only=True)),
            estimated=True,
            source="estimated — camp side derived from base positions",
        ),
        StreamTotals(
            key="camps_conceded",
            label="Camps conceded — cleared by the enemy inside own half",
            by_team=per_team(conceded_camps),
            higher_is_better=False,
            estimated=True,
            source="estimated — same attribution as above",
        ),
        StreamTotals(
            key="time_in_enemy_half",
            label="Share of the match spent inside the enemy half",
            by_team=per_team(enemy_ground),
            unit="%",
            source="measured — sampled hero positions against the two base positions",
        ),
        StreamTotals(
            key="kills",
            label="Kills",
            by_team={t: float(kills_by_team.get(t, 0)) for t in teams},
        ),
        StreamTotals(
            key="kills_in_enemy_half",
            label="— kills scored on the enemy's own ground",
            by_team=per_team(enemy_ground_kills),
        ),
        StreamTotals(
            key="structures",
            label="Structures destroyed",
            by_team={t: float(structures.get(t, 0)) for t in teams},
        ),
        StreamTotals(
            key="objective_damage",
            label="Objective damage",
            by_team=per_team(sum_reports("objective_damage")),
        ),
        StreamTotals(
            key="mid_boss",
            label="Mid Boss taken",
            by_team=per_team(lambda t: prize_count(t, "mid_boss")),
        ),
        StreamTotals(
            key="rift",
            label="Rifts captured",
            by_team=per_team(lambda t: prize_count(t, "rift")),
        ),
        StreamTotals(
            key="urn_runs",
            label="Urn runs started",
            by_team=per_team(urn_starts),
            source="measured — this build's urn dataset has no delivery event, so "
            "whether a run scored is unknown",
        ),
        StreamTotals(
            key="urn_denials",
            label="Enemy urn runs ended by killing the carrier",
            by_team=per_team(urn_denials),
        ),
        StreamTotals(
            key="dead_time",
            label="Time spent dead",
            by_team=per_team(dead_seconds),
            unit="s",
            higher_is_better=False,
            estimated=True,
            source="estimated — respawn read from the gap in positional samples",
        ),
    ]
    return [s for s in out if any(v is not None for v in s.by_team.values())]


# --------------------------------------------------------------- windows


def _windows(
    md: MatchData,
    teams: tuple[int, ...],
    curve: list[EconomySnapshot],
    clears: tuple[CampClear, ...],
    prizes: tuple[NeutralPrize, ...],
    window_seconds: float,
) -> list[WindowLedger]:
    assert md.clock is not None
    slots = fixed_windows(md, window_seconds)
    out: list[WindowLedger] = []
    for slot in slots:
        minutes = max(
            (md.clock.seconds(slot.end_tick) - md.clock.seconds(slot.start_tick)) / 60.0, 1e-9
        )
        start_worth = _worth_at(curve, slot.start_tick, teams)
        end_worth = _worth_at(curve, slot.end_tick, teams)
        gained = {t: end_worth[t] - start_worth[t] for t in teams}
        kills = _kills_by_team(md, teams, slot.start_tick, slot.end_tick)
        structures = _structures_by_team(md, teams, slot.start_tick, slot.end_tick)
        camps = {
            t: sum(
                1
                for c in clears
                if c.team_num == t and slot.start_tick < c.tick <= slot.end_tick
            )
            for t in teams
        }
        raids = {
            t: sum(
                1
                for c in clears
                if c.team_num == t and c.is_raid and slot.start_tick < c.tick <= slot.end_tick
            )
            for t in teams
        }
        prize_counts = {
            t: sum(
                1 for p in prizes if p.team_num == t and slot.start_tick < p.tick <= slot.end_tick
            )
            for t in teams
        }

        ranked = sorted(gained.items(), key=lambda kv: kv[1], reverse=True)
        leader = ranked[0][0] if ranked[0][1] > ranked[1][1] else None
        margin = ranked[0][1] - ranked[1][1]
        out.append(
            WindowLedger(
                label=slot.name,
                start_tick=slot.start_tick,
                end_tick=slot.end_tick,
                minutes=minutes,
                net_worth_by_team=end_worth,
                gained_by_team=gained,
                per_min_by_team={t: round(gained[t] / minutes, 1) for t in teams},
                kills_by_team=kills,
                structures_by_team=structures,
                camps_by_team=camps,
                raids_by_team=raids,
                prizes_by_team=prize_counts,
                leader=leader,
                margin=abs(margin),
                driver=_driver(leader, teams, kills, structures, camps, prize_counts),
            )
        )
    return out


def _driver(
    leader: int | None,
    teams: tuple[int, ...],
    kills: dict[int, int],
    structures: dict[int, int],
    camps: dict[int, int],
    prizes: dict[int, int],
) -> str:
    """Name the stream that most plausibly produced the window's gap.

    Deliberately coarse: this points a reader at the right paragraph, it does
    not claim to have attributed souls to a cause.
    """
    if leader is None:
        return "even"
    other = [t for t in teams if t != leader]
    if not other:
        return "even"
    rival = other[0]
    edges = {
        "fights": kills.get(leader, 0) - kills.get(rival, 0),
        "structures": structures.get(leader, 0) - structures.get(rival, 0),
        "farm": camps.get(leader, 0) - camps.get(rival, 0),
        "map prizes": prizes.get(leader, 0) - prizes.get(rival, 0),
    }
    best = max(edges.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else "farm tempo, no event edge"


def _worth_at(
    curve: list[EconomySnapshot], tick: int, teams: tuple[int, ...]
) -> dict[int, int]:
    latest: EconomySnapshot | None = None
    for snapshot in curve:
        if snapshot.tick > tick:
            break
        latest = snapshot
    if latest is None:
        return {t: 0 for t in teams}
    return {t: int(latest.net_worth_by_team.get(t, 0)) for t in teams}


def _kills_by_team(
    md: MatchData, teams: tuple[int, ...], start: int, end: int
) -> dict[int, int]:
    out = {t: 0 for t in teams}
    if md.kills.is_empty():
        return out
    window = md.kills.filter((pl.col("tick") > start) & (pl.col("tick") <= end))
    if start == 0:
        window = md.kills.filter(pl.col("tick") <= end)
    for row in window.iter_rows(named=True):
        victim_team = md.team_of(row.get("victim_hero_id"))
        for team in out:
            if team != victim_team:
                out[team] += 1
    return out


def _structures_by_team(
    md: MatchData, teams: tuple[int, ...], start: int, end: int
) -> dict[int, int]:
    """Structures each team *destroyed*, counted once per entity."""
    out = {t: 0 for t in teams}
    df = md.objectives
    if df.is_empty():
        return out
    dead = df.filter(
        pl.col("health").is_not_null()
        & (pl.col("health") <= 0)
        & pl.col("team_num").is_in(list(teams))
    )
    if dead.is_empty():
        return out
    keys = (
        ["entity_id"]
        if dead["entity_id"].null_count() < dead.height
        else ["objective_type", "team_num", "lane"]
    )
    first = dead.sort("tick").group_by(keys, maintain_order=True).first()
    lower = pl.col("tick") >= start if start == 0 else pl.col("tick") > start
    for row in first.filter(lower & (pl.col("tick") <= end)).iter_rows(named=True):
        owner = row.get("team_num")
        for team in out:
            if team != owner:
                out[team] += 1
    return out


# ----------------------------------------------------------- neutral farm


class _Positions:
    """Sampled hero positions, indexed for nearest-tick lookup."""

    def __init__(self, md: MatchData):
        self.ticks: list[int] = []
        self.rows: dict[int, list[tuple[int, float, float]]] = {}
        samples = md.player_samples
        if samples.is_empty():
            return
        alive = samples.filter(
            pl.col("x").is_not_null()
            & pl.col("y").is_not_null()
            & (pl.col("is_alive").is_null() | pl.col("is_alive"))
        )
        for row in alive.sort("tick").iter_rows(named=True):
            hero, tick = row.get("hero_id"), row.get("tick")
            if hero is None or tick is None:
                continue
            self.rows.setdefault(int(tick), []).append(
                (int(hero), float(row["x"]), float(row["y"]))
            )
        self.ticks = sorted(self.rows)

    def near(self, tick: int, max_gap_ticks: int) -> list[tuple[int, float, float]]:
        if not self.ticks:
            return []
        at = bisect_right(self.ticks, tick)
        candidates = [self.ticks[at - 1]] if at else []
        if at < len(self.ticks):
            candidates.append(self.ticks[at])
        best = min(candidates, key=lambda t: abs(t - tick), default=None)
        if best is None or abs(best - tick) > max_gap_ticks:
            return []
        return self.rows[best]


def camp_clears(md: MatchData) -> list[CampClear]:
    """Every neutral camp that was cleared, credited by proximity.

    Two inferences stack here and both are labelled wherever they surface.
    *That* a camp died is inferred from the health series (see
    ``CAMP_KILL_HEALTH_SHARE``); *who* killed it is not in the demo at all, so
    it is credited to the nearest living hero at that moment.
    """
    camps = neutral_camps(md)
    if not camps:
        return []
    assert md.clock is not None
    series = _neutral_series(md)
    kills = _entity_kill_ticks(md, series)
    if not kills:
        return []
    bases = team_bases(md)
    positions = _Positions(md)
    max_gap = int(CAMP_CREDIT_MAX_GAP_SECONDS * md.tick_rate)
    teams_of = {hero: md.team_of(hero) for hero in md.hero_ids}
    merge = CAMP_CLEAR_MERGE_SECONDS * md.tick_rate

    out: list[CampClear] = []
    for camp in camps:
        ticks = sorted(t for entity in camp.entity_ids for t in kills.get(entity, []))
        # A camp is several entities; killing it produces one low-health row per
        # creep within a few seconds. Collapse those into a single clear.
        for group in _cluster(ticks, merge):
            tick = max(group)
            team, distance = _credit_camp(camp, tick, positions, teams_of, max_gap)
            out.append(
                CampClear(
                    tick=tick,
                    camp_id=camp.camp_id,
                    x=camp.x,
                    y=camp.y,
                    half_of=_half_of(camp.x, camp.y, bases),
                    team_num=team,
                    distance=distance,
                )
            )
    return sorted(out, key=lambda c: c.tick)


def _neutral_series(md: MatchData) -> dict[int, list[tuple[int, int, int]]]:
    """``entity -> [(tick, health, max_health)]``, ascending."""
    out: dict[int, list[tuple[int, int, int]]] = {}
    if md.neutrals.is_empty():
        return out
    for row in md.neutrals.sort("tick").iter_rows(named=True):
        entity, tick = row.get("entity_id"), row.get("tick")
        health, maximum = row.get("health"), row.get("max_health")
        if entity is None or tick is None or health is None or not maximum:
            continue
        out.setdefault(int(entity), []).append((int(tick), int(health), int(maximum)))
    return out


def _entity_kill_ticks(
    md: MatchData, series: dict[int, list[tuple[int, int, int]]]
) -> dict[int, list[int]]:
    assert md.clock is not None
    out: dict[int, list[int]] = {}
    for entity, rows in series.items():
        for index, (tick, health, maximum) in enumerate(rows):
            if health > CAMP_KILL_HEALTH_SHARE * maximum:
                continue
            if index + 1 == len(rows):
                # Last row of the file on low health: cleared, never came back.
                out.setdefault(entity, []).append(tick)
                continue
            next_tick, next_health, next_max = rows[index + 1]
            quiet = md.clock.seconds(next_tick) - md.clock.seconds(tick)
            if (
                quiet >= CAMP_RESPAWN_MIN_SECONDS
                and next_health >= CAMP_RESPAWN_HEALTH_SHARE * next_max
            ):
                out.setdefault(entity, []).append(tick)
    return out


def _cluster(ticks: list[int], window: float) -> list[list[int]]:
    groups: list[list[int]] = []
    for tick in ticks:
        if groups and tick - groups[-1][-1] <= window:
            groups[-1].append(tick)
        else:
            groups.append([tick])
    return groups


def _credit_camp(
    camp: Camp,
    tick: int,
    positions: _Positions,
    teams_of: dict[int, int | None],
    max_gap: int,
) -> tuple[int | None, float | None]:
    nearest: dict[int, float] = {}
    for hero, x, y in positions.near(tick, max_gap):
        team = teams_of.get(hero)
        if team is None:
            continue
        distance = hypot(x - camp.x, y - camp.y)
        if distance < nearest.get(team, float("inf")):
            nearest[team] = distance
    if not nearest:
        return None, None
    team, distance = min(nearest.items(), key=lambda kv: kv[1])
    if distance > CAMP_CREDIT_RANGE:
        return None, distance
    return team, distance


def territory(md: MatchData, teams: tuple[int, ...]) -> list[Territory]:
    """Who was standing on whose ground, and where the kills landed.

    Sampled hero positions are already in the demo, so this needs no vision
    model and no attribution: a hero in the enemy half either was there or was
    not. It is the cleanest available proxy for "we were taking their map".
    """
    bases = team_bases(md)
    if len(bases) != 2 or md.player_samples.is_empty():
        return []
    positions = _Positions(md)
    teams_of = {hero: md.team_of(hero) for hero in md.hero_ids}

    samples = {t: 0 for t in teams}
    invading = {t: 0 for t in teams}
    for rows in positions.rows.values():
        for hero, x, y in rows:
            team = teams_of.get(hero)
            if team not in samples:
                continue
            samples[team] += 1
            half = _half_of(x, y, bases)
            if half is not None and half != team:
                invading[team] += 1

    kills_away = {t: 0 for t in teams}
    deaths_home = {t: 0 for t in teams}
    if not md.kills.is_empty():
        max_gap = int(CAMP_CREDIT_MAX_GAP_SECONDS * md.tick_rate)
        for row in md.kills.iter_rows(named=True):
            victim = row.get("victim_hero_id")
            victim_team = teams_of.get(victim) if victim is not None else None
            if victim_team not in samples:
                continue
            spot = next(
                (
                    (x, y)
                    for hero, x, y in positions.near(int(row["tick"]), max_gap)
                    if hero == victim
                ),
                None,
            )
            if spot is None:
                continue
            half = _half_of(spot[0], spot[1], bases)
            if half is None:
                continue
            if half == victim_team:
                deaths_home[victim_team] += 1
                for team in kills_away:
                    if team != victim_team:
                        kills_away[team] += 1

    return [
        Territory(
            team_num=team,
            samples=samples[team],
            enemy_half_samples=invading[team],
            kills_in_enemy_half=kills_away[team],
            deaths_in_own_half=deaths_home[team],
        )
        for team in teams
    ]


def _half_of(x: float, y: float, bases: dict[int, tuple[float, float]]) -> int | None:
    if len(bases) != 2:
        return None
    ranked = sorted(bases.items(), key=lambda kv: hypot(x - kv[1][0], y - kv[1][1]))
    near, far = ranked[0], ranked[1]
    near_d = hypot(x - near[1][0], y - near[1][1])
    far_d = hypot(x - far[1][0], y - far[1][1])
    # A camp equidistant from both bases belongs to neither half.
    if far_d <= near_d * 1.05:
        return None
    return near[0]


# -------------------------------------------------------- neutral prizes


def neutral_prizes(md: MatchData) -> list[NeutralPrize]:
    """Mid Boss and rift outcomes — public, contested, and worth souls."""
    out: list[NeutralPrize] = []

    if not md.mid_boss.is_empty():
        rows = md.mid_boss.sort("tick").to_dicts()
        window = int(MID_BOSS_USE_SECONDS * md.tick_rate)
        for index, row in enumerate(rows):
            if str(row.get("event") or "") != "killed":
                continue
            tick = int(row["tick"])
            # The kill row carries the boss's neutral team; the buff `used`
            # rows that follow it name whoever actually took it.
            taker = None
            for later in rows[index + 1 :]:
                if int(later["tick"]) - tick > window:
                    break
                if str(later.get("event") or "") == "used" and later.get("team_num") in md.team_nums:
                    taker = int(later["team_num"])
                    break
            out.append(
                NeutralPrize(
                    kind="mid_boss",
                    tick=tick,
                    team_num=taker,
                    detail="taken" if taker is not None else "killed, taker not recorded",
                )
            )

    if not md.rift.is_empty():
        for row in md.rift.sort("rift_num").iter_rows(named=True):
            capture = row.get("capture_tick")
            if capture is None:
                expire = row.get("expire_tick")
                if expire is not None:
                    out.append(
                        NeutralPrize(
                            kind="rift",
                            tick=int(expire),
                            team_num=None,
                            detail=f"rift #{row.get('rift_num')} expired uncaptured",
                        )
                    )
                continue
            winner = row.get("winning_team")
            out.append(
                NeutralPrize(
                    kind="rift",
                    tick=int(capture),
                    team_num=int(winner) if winner in md.team_nums else None,
                    detail=f"rift #{row.get('rift_num')} captured",
                )
            )
    return sorted(out, key=lambda p: p.tick)


def urn_runs(md: MatchData) -> list[UrnRun]:
    """Each urn pickup, and how it ended.

    This build's urn dataset carries ``picked_up`` / ``dropped`` / ``returned``
    but no delivery event, so a run's *payout* is unknowable from the demo. The
    contest is still measurable, and that is the part a coach can act on: who
    reached for it, and who ended somebody else's reach by killing the carrier.
    """
    if md.urn.is_empty():
        return []
    rows = md.urn.sort("tick").to_dicts()
    deaths = _death_ticks(md)
    denial_window = int(URN_DENIAL_SECONDS * md.tick_rate)

    out: list[UrnRun] = []
    open_runs: dict[int, int] = {}  # hero -> pickup tick
    reset_at: dict[int, int] = {}
    for row in rows:
        event = str(row.get("event") or "")
        hero = row.get("hero_id")
        tick = int(row["tick"])
        if not hero:
            continue
        if event == "returned":
            reset_at[int(hero)] = tick
        if event == "picked_up":
            if tick - reset_at.get(int(hero), -URN_RESET_TICKS - 1) <= URN_RESET_TICKS:
                continue
            open_runs[int(hero)] = tick
        elif event in ("dropped", "returned") and int(hero) in open_runs:
            start = open_runs.pop(int(hero))
            killer = _killer_near(md, deaths, int(hero), tick, denial_window)
            out.append(
                UrnRun(
                    pickup_tick=start,
                    hero_id=int(hero),
                    team_num=md.team_of(int(hero)),
                    end_tick=tick,
                    ended_by=event,
                    denied_by=killer,
                )
            )
    for hero, start in open_runs.items():
        out.append(
            UrnRun(
                pickup_tick=start,
                hero_id=hero,
                team_num=md.team_of(hero),
                end_tick=None,
                ended_by="unknown",
                denied_by=None,
            )
        )
    return sorted(out, key=lambda r: r.pickup_tick)


def _death_ticks(md: MatchData) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    if md.kills.is_empty():
        return out
    for row in md.kills.sort("tick").iter_rows(named=True):
        victim = row.get("victim_hero_id")
        if victim is None:
            continue
        out.setdefault(int(victim), []).append(int(row["tick"]))
    return out


def _killer_near(
    md: MatchData,
    deaths: dict[int, list[int]],
    hero: int,
    tick: int,
    window: int,
) -> int | None:
    """The enemy team, if the carrier died within ``window`` ticks of the drop."""
    for death in deaths.get(hero, []):
        if abs(death - tick) <= window:
            team = md.team_of(hero)
            others = [t for t in md.team_nums if t != team]
            return others[0] if len(others) == 1 else None
    return None


# --------------------------------------------------------------- downtime


def death_downtime(md: MatchData, curve: list[EconomySnapshot] | None = None) -> list[Downtime]:
    """Seconds spent dead per player, priced at that player's own earning rate.

    Respawn is not in the demo either. A dead player simply stops emitting
    positional rows, so the respawn is the first sampled tick after the death
    where the hero reappears — an estimate, and reported as one. A death with
    no reappearance (the last one of the match) is excluded rather than charged
    the entire remaining game.
    """
    assert md.clock is not None
    samples = md.player_samples
    if samples.is_empty() or md.kills.is_empty():
        return []

    alive_ticks: dict[int, list[int]] = {}
    live = samples.filter(pl.col("is_alive").is_null() | pl.col("is_alive"))
    for row in live.sort("tick").iter_rows(named=True):
        hero = row.get("hero_id")
        if hero is None:
            continue
        alive_ticks.setdefault(int(hero), []).append(int(row["tick"]))

    deaths = _death_ticks(md)
    reports = {r.hero_id: r for r in player_reports(md)}
    match_minutes = max(md.clock.seconds(md.end_tick) / 60.0, 1e-9)
    cap = MAX_RESPAWN_SECONDS

    out: list[Downtime] = []
    for hero in md.hero_ids:
        hero_deaths = deaths.get(hero, [])
        ticks = alive_ticks.get(hero, [])
        seconds = 0.0
        measured = 0
        for death in hero_deaths:
            at = bisect_right(ticks, death)
            if at >= len(ticks):
                continue
            gap = md.clock.seconds(ticks[at]) - md.clock.seconds(death)
            if gap <= 0 or gap > cap:
                continue
            seconds += gap
            measured += 1

        report = reports.get(hero)
        net_worth = report.final_net_worth if report else None
        alive_minutes = max(match_minutes - seconds / 60.0, 1e-9)
        rate = round(net_worth / alive_minutes, 1) if net_worth is not None else None
        forgone = int(round(rate * seconds / 60.0)) if rate is not None else None
        out.append(
            Downtime(
                hero_id=hero,
                team_num=md.team_of(hero),
                player_name=md.player_name(hero),
                deaths=len(hero_deaths),
                measured_deaths=measured,
                seconds=round(seconds, 1),
                souls_per_min=rate,
                souls_forgone=forgone,
            )
        )
    return sorted(out, key=lambda d: d.souls_forgone or 0, reverse=True)


# ----------------------------------------------------------------- levers


def _levers(
    md: MatchData,
    names: Names,
    teams: tuple[int, ...],
    focus_team: int | None,
    reports,
    windows: tuple[WindowLedger, ...],
    dead_time: tuple[Downtime, ...],
    clears: tuple[CampClear, ...],
    prizes: tuple[NeutralPrize, ...],
    ground: tuple[Territory, ...],
) -> list[Lever]:
    if focus_team is None:
        return []
    rival = next((t for t in teams if t != focus_team), None)
    if rival is None:
        return []
    assert md.clock is not None

    candidates: list[Lever] = []
    candidates += _lever_worst_window(md, names, focus_team, rival, windows)
    candidates += _lever_downtime(md, names, focus_team, dead_time)
    candidates += _lever_net_worth_gap(md, names, focus_team, rival, reports)
    candidates += _lever_territory(md, names, focus_team, rival, ground)
    candidates += _lever_camp_denial(md, names, focus_team, rival, clears)
    candidates += _lever_prizes(md, names, focus_team, rival, prizes)
    candidates += _lever_unconverted(md, names, focus_team)
    candidates += _lever_laning_farm(md, names, focus_team, rival)

    ranked = sorted(
        candidates,
        key=lambda lever: (lever.souls_at_stake is not None, lever.souls_at_stake or 0),
        reverse=True,
    )
    return ranked[:MAX_LEVERS]


def _lever_worst_window(
    md: MatchData,
    names: Names,
    focus: int,
    rival: int,
    windows: tuple[WindowLedger, ...],
) -> list[Lever]:
    losses = [w for w in windows if w.leader == rival]
    if not losses:
        return []
    worst = max(losses, key=lambda w: w.margin)
    per_min = worst.margin / max(worst.minutes, 1e-9)
    return [
        Lever(
            key="worst_window",
            title=f"{worst.label} is where the game was bought",
            team_num=focus,
            souls_at_stake=worst.margin,
            confidence="high",
            evidence=(
                f"{names.team(rival)} out-earned you {worst.gained_by_team[rival]:,} to "
                f"{worst.gained_by_team[focus]:,} souls across {worst.minutes:.1f} minutes "
                f"({per_min:,.0f} souls/min of ground lost).",
                "Inside that window — "
                + "; ".join(
                    f"{label} {names.team(focus)} {mine}, {names.team(rival)} {theirs}"
                    for label, mine, theirs in (
                        ("kills", worst.kills_by_team[focus], worst.kills_by_team[rival]),
                        (
                            "structures",
                            worst.structures_by_team[focus],
                            worst.structures_by_team[rival],
                        ),
                        ("camps", worst.camps_by_team[focus], worst.camps_by_team[rival]),
                        ("map prizes", worst.prizes_by_team[focus], worst.prizes_by_team[rival]),
                    )
                )
                + ".",
                f"Dominant edge in that window: {worst.driver}.",
            ),
            action=(
                f"Start the review here rather than at the first death. Whatever else is on "
                f"this list, the {worst.margin:,} souls conceded between "
                f"{md.clock.mmss(worst.start_tick) if md.clock else '--:--'} and "
                f"{md.clock.mmss(worst.end_tick) if md.clock else '--:--'} is the largest "
                f"single block of the gap."
            ),
        )
    ]


def _lever_downtime(
    md: MatchData, names: Names, focus: int, dead_time: tuple[Downtime, ...]
) -> list[Lever]:
    mine = [d for d in dead_time if d.team_num == focus and d.souls_forgone]
    if not mine:
        return []
    total = sum(d.souls_forgone or 0 for d in mine)
    worst = mine[0]
    return [
        Lever(
            key="downtime",
            title="Souls you never earned because you were dead",
            team_num=focus,
            souls_at_stake=total,
            confidence="medium",
            evidence=(
                f"{names.team(focus)} spent {sum(d.seconds for d in mine) / 60:.1f} minutes dead "
                f"across {sum(d.deaths for d in mine)} deaths.",
                f"Worst: {names.hero(worst.hero_id)}"
                + (f" ({worst.player_name})" if worst.player_name else "")
                + f" — {worst.deaths} deaths, {worst.seconds / 60:.1f} minutes on the floor, "
                f"~{worst.souls_forgone:,} souls forgone at their own "
                f"{worst.souls_per_min:,.0f}/min rate.",
                "Respawn time is inferred from the gap in positional samples, so treat the "
                "totals as an estimate, not a counter.",
            ),
            action=(
                "Downtime compounds twice — the souls you do not earn, and the map the enemy "
                "farms unopposed while you wait. Cut the deaths that bought nothing before "
                "chasing more farm."
            ),
        )
    ]


def _lever_net_worth_gap(
    md: MatchData, names: Names, focus: int, rival: int, reports
) -> list[Lever]:
    def ranked(team: int):
        rows = [r for r in reports if r.team_num == team and r.final_net_worth is not None]
        return sorted(rows, key=lambda r: r.final_net_worth, reverse=True)

    mine, theirs = ranked(focus), ranked(rival)
    if not mine or not theirs:
        return []
    pairs = list(zip(mine, theirs))
    worst = min(pairs, key=lambda pair: pair[0].final_net_worth - pair[1].final_net_worth)
    gap = worst[1].final_net_worth - worst[0].final_net_worth
    if gap <= 0:
        return []
    slot = pairs.index(worst) + 1
    return [
        Lever(
            key="net_worth_slot",
            title=f"Your #{slot} net worth lost their #{slot} by {gap:,}",
            team_num=focus,
            souls_at_stake=gap,
            confidence="high",
            evidence=(
                f"{names.hero(worst[0].hero_id)}"
                + (f" ({worst[0].player_name})" if worst[0].player_name else "")
                + f" finished on {worst[0].final_net_worth:,} souls; their equivalent slot "
                f"{names.hero(worst[1].hero_id)} finished on {worst[1].final_net_worth:,}.",
                f"Last hits {worst[0].last_hits if worst[0].last_hits is not None else 'unknown'} "
                f"vs {worst[1].last_hits if worst[1].last_hits is not None else 'unknown'}; "
                f"denies {worst[0].denies if worst[0].denies is not None else 'unknown'} vs "
                f"{worst[1].denies if worst[1].denies is not None else 'unknown'}.",
            ),
            action=(
                "Slots are matched by final net worth, not by role, so read this as "
                "'the resource share that went missing' rather than a matchup call."
            ),
        )
    ]


def _lever_territory(
    md: MatchData, names: Names, focus: int, rival: int, ground: tuple[Territory, ...]
) -> list[Lever]:
    mine = next((t for t in ground if t.team_num == focus), None)
    theirs = next((t for t in ground if t.team_num == rival), None)
    if mine is None or theirs is None or mine.share is None or theirs.share is None:
        return []
    if theirs.share <= mine.share:
        return []
    return [
        Lever(
            key="territory",
            title="They played the match on your half of the map",
            team_num=focus,
            souls_at_stake=None,
            confidence="high",
            evidence=(
                f"{names.team(rival)} spent {theirs.share:.1%} of their sampled time inside "
                f"your half; {names.team(focus)} spent {mine.share:.1%} inside theirs.",
                f"Kills scored on enemy ground: {names.team(focus)} {mine.kills_in_enemy_half}, "
                f"{names.team(rival)} {theirs.kills_in_enemy_half}.",
                "Position samples only — no vision or intent is inferred, just where twelve "
                "heroes actually stood.",
            ),
            action=(
                "Every minute they spend on your half is a minute of your farm they can take "
                "and you cannot. Ground is the resource that carries all the others; contest "
                "it before contesting any single fight."
            ),
        )
    ]


def _lever_camp_denial(
    md: MatchData, names: Names, focus: int, rival: int, clears: tuple[CampClear, ...]
) -> list[Lever]:
    if not clears:
        return []
    my_raids = sum(1 for c in clears if c.team_num == focus and c.is_raid)
    their_raids = sum(1 for c in clears if c.team_num == rival and c.is_raid)
    mine = sum(1 for c in clears if c.team_num == focus)
    theirs = sum(1 for c in clears if c.team_num == rival)
    if theirs <= mine and their_raids <= my_raids:
        return []
    return [
        Lever(
            key="camp_denial",
            title="The jungle was one-sided",
            team_num=focus,
            souls_at_stake=None,
            confidence="low",
            evidence=(
                f"Camps cleared: {names.team(focus)} {mine}, {names.team(rival)} {theirs}.",
                f"Camps cleared inside the *enemy* half: {names.team(focus)} {my_raids}, "
                f"{names.team(rival)} {their_raids}. Their raids are souls taken off your map, "
                "not just souls they earned.",
                "Camp kills are credited to the nearest living hero, so this is an inference; "
                "the demo does not record who hit a neutral. Camp bounties are not in the demo "
                "either, so no soul figure is given.",
            ),
            action=(
                "A camp taken in your half is worth double — they gain it and you cannot. "
                "Contesting the raids is usually cheaper than out-farming them."
            ),
        )
    ]


def _lever_prizes(
    md: MatchData, names: Names, focus: int, rival: int, prizes: tuple[NeutralPrize, ...]
) -> list[Lever]:
    if not prizes:
        return []
    mine = sum(1 for p in prizes if p.team_num == focus)
    theirs = sum(1 for p in prizes if p.team_num == rival)
    if theirs <= mine:
        return []
    assert md.clock is not None
    lost = [p for p in prizes if p.team_num == rival]
    return [
        Lever(
            key="map_prizes",
            title="Mid Boss and rifts went to them",
            team_num=focus,
            souls_at_stake=None,
            confidence="high",
            evidence=(
                f"Contested map rewards resolved {names.team(focus)} {mine} — "
                f"{names.team(rival)} {theirs}.",
                "Taken by them: "
                + ", ".join(f"{p.kind} at {md.clock.mmss(p.tick)}" for p in lost[:6]),
                "These are the least deniable windows in the report: a spawning Mid Boss and an "
                "active rift are announced to both teams, so neither side needed vision to know.",
            ),
            action=(
                "Public timers are free information. Contesting one is a decision you can make "
                "before the fight starts, unlike everything else in this report."
            ),
        )
    ]


def _lever_unconverted(md: MatchData, names: Names, focus: int) -> list[Lever]:
    fights = analyze_fights(md, names)
    assert md.clock is not None
    missed = [
        f
        for f in fights
        if f.winner == focus
        and not f.converted_into
        and f.conversion_assessment is not None
        and f.conversion_assessment.status == "push_now"
    ]
    if not missed:
        return []
    return [
        Lever(
            key="unconverted",
            title=f"{len(missed)} won fights bought nothing",
            team_num=focus,
            souls_at_stake=None,
            confidence="high",
            evidence=(
                "Fights won with a structure genuinely available afterwards — the wave check "
                "passed, so this is not a report blaming you for a lane with no troopers: "
                + ", ".join(md.clock.mmss(f.end_tick) for f in missed[:6]),
                "A won fight is a timer, not a reward. It only becomes souls when it is spent "
                "on a structure, a camp sweep, or a boss.",
            ),
            action=(
                "After a won fight, the default should be a target, not a recall. The report "
                "already lists which target was reachable in each case."
            ),
        )
    ]


def _lever_laning_farm(md: MatchData, names: Names, focus: int, rival: int) -> list[Lever]:
    stats = [s for s in phase_stats(md) if s.phase == "laning"]
    if not stats:
        return []

    def total(team: int, attr: str) -> int | None:
        values = [
            getattr(s, attr) for s in stats if s.team_num == team and getattr(s, attr) is not None
        ]
        return sum(values) if values else None

    mine, theirs = total(focus, "last_hits"), total(rival, "last_hits")
    my_souls, their_souls = total(focus, "net_worth_gained"), total(rival, "net_worth_gained")
    if mine is None or theirs is None or my_souls is None or their_souls is None:
        return []
    if their_souls - my_souls <= 0:
        return []
    return [
        Lever(
            key="laning_farm",
            title="The lane phase already had a gap in it",
            team_num=focus,
            souls_at_stake=their_souls - my_souls,
            confidence="high",
            evidence=(
                f"Souls gained during laning: {names.team(focus)} {my_souls:,}, "
                f"{names.team(rival)} {their_souls:,}.",
                f"Lane last hits over the same window: {mine:,} vs {theirs:,}.",
                "Laning souls compound for the rest of the match — this gap is the cheapest one "
                "on the list to close, because it needs no rotation, no vision and no fight.",
            ),
            action=(
                "Before looking at any teamfight, check whether the lane was already losing "
                "souls per minute. A fight loss that follows a farm loss is a symptom."
            ),
        )
    ]


def _caveats(
    md: MatchData,
    clears: tuple[CampClear, ...],
    runs: tuple[UrnRun, ...],
    dead_time: tuple[Downtime, ...],
) -> tuple[str, ...]:
    out: list[str] = []
    if clears:
        unattributed = sum(1 for c in clears if c.team_num is None)
        out.append(
            f"Camp clears: {len(clears)} detected, {unattributed} could not be credited to a "
            "team (no sampled hero within range). The demo does not record who damaged a "
            "neutral; credit is proximity-inferred."
        )
    else:
        out.append(
            "No neutral camp clears could be read from this demo, so jungle income is "
            "unknown, not zero."
        )
    if runs:
        out.append(
            "Urn: this build records pickup, drop and return but no delivery event, so whether "
            "a run scored is unknown. Only the contest is reported."
        )
    if dead_time and any(d.measured_deaths < d.deaths for d in dead_time):
        out.append(
            "Some deaths had no measurable respawn (the last death of a match never respawns); "
            "those are excluded from downtime rather than charged the rest of the game."
        )
    return tuple(out)
