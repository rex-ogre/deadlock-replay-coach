"""Derived tactical reads — the layer that turns events into *judgements*.

A model handed a kill feed can tell you what happened. It cannot tell you the
kill was a 3v1 pickoff on a rotating support that converted into a Walker 20
seconds later, because that requires joining positions, timings and objectives.
That join happens here, in Polars, and the result is a handful of short
sentences the model can reason over.

Every threshold is a named constant with a stated rationale, because these are
judgement calls and you will want to tune them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

import polars as pl

from .gamedata import GameConstants, load_constants
from .match import MatchData
from .names import NO_HERO, Names
from .physics import distance_m, travel_seconds

# A teamfight "converts" if the winner takes an objective within this window.
# 45s is roughly a respawn + rotation in Deadlock: long enough to include the
# objective the fight was fought *for*, short enough to exclude the next one.
CONVERSION_WINDOW_SECONDS = 45.0

# A structure is immediately pushable only when the winning team has a real
# creep wave near it and at least one healthy winner can meet that wave. These
# deliberately conservative thresholds prevent "won but took nothing" from
# being presented as a player mistake when the map offered no legal push.
CONVERSION_WAVE_READY_DISTANCE = 3_000.0
CONVERSION_WAVE_SETUP_DISTANCE = 4_500.0
CONVERSION_HERO_READY_DISTANCE = 4_500.0
CONVERSION_HERO_SETUP_DISTANCE = 7_500.0
CONVERSION_MIN_FRONT_TROOPERS = 3
CONVERSION_MIN_HEALTH_PCT = 0.35
CONVERSION_TROOPER_CONTEST_RANGE = 1_800.0
CONVERSION_SETUP_LOOKAHEAD_SECONDS = 10.0
MID_BOSS_FIRST_AVAILABLE_SECONDS = 600.0

# Boon sometimes emits several adjacent damage clusters for one continuous
# engagement.  Treat clusters that are only a few seconds apart and still
# share at least two participants as one fight; otherwise one kill can appear
# as three separate 2v2 victories in the report.
FIGHT_MERGE_GAP_SECONDS = 3.0

# Two heroes further apart than this cannot realistically trade for each other.
# Deadlock's map units are large; ~2500 is a lane's worth of spacing.
ISOLATION_RADIUS = 2500.0

# A fight is decided long before this. Once the closest living teammate needs
# more than eight seconds even under the most generous movement assumptions,
# "he died with nobody nearby" stops being a read on the victim's positioning
# and becomes a read on the team's shape — which is a different conversation
# and often a deliberate, correct split.
UNSUPPORTABLE_SECONDS = 8.0

# How far a sampled tick may be from the tick we asked about before we treat
# the sample as not describing that moment (0.5s at 60 tick).
SAMPLE_TOLERANCE_TICKS = 30

# Fallback phase boundaries, used only when a demo has no objective kills to
# derive real ones from.
FALLBACK_LANING_END_SECONDS = 600.0
FALLBACK_MIDGAME_END_SECONDS = 1500.0


@dataclass(frozen=True)
class Phase:
    name: str
    start_tick: int
    end_tick: int
    trigger: str


@dataclass(frozen=True)
class ConversionAssessment:
    """What the fight winner could plausibly take, given the actual waves."""

    status: str
    target: str | None
    lane: int | None
    sample_tick: int | None
    allied_front_troopers: int | None
    enemy_contesting_troopers: int | None
    wave_distance: float | None
    nearest_winner_distance: float | None
    winner_distances: dict[int, float]
    reason: str
    alternatives: tuple[str, ...] = ()


@dataclass(frozen=True)
class FightSummary:
    fight_id: int
    start_tick: int
    end_tick: int
    duration_seconds: float
    participants_by_team: dict[int, list[int]]
    kills_by_team: dict[int, int]
    winner: int | None
    verdict: str
    engagement: str
    hero_damage: int
    converted_into: str | None
    conversion_assessment: ConversionAssessment | None
    phase: str


@dataclass(frozen=True)
class KillContext:
    tick: int
    victim_hero_id: int | None
    attacker_hero_id: int | None
    attackers: int
    defenders_nearby: int | None
    outnumbered: bool | None
    isolated: bool | None
    phase: str
    # How far help actually was, in the units that decide whether "alone" was a
    # mistake. ``support_seconds`` is a lower bound on the nearest living
    # teammate's travel time (see :mod:`physics`), so a large value is evidence
    # that no positioning choice by the victim could have changed the outcome.
    nearest_ally_m: float | None = None
    support_seconds: float | None = None

    @property
    def support_read(self) -> str | None:
        """Why the victim was alone, phrased as a decision rather than a label."""
        if self.support_seconds is None or self.nearest_ally_m is None:
            return None
        if self.defenders_nearby:
            return None
        if self.support_seconds >= UNSUPPORTABLE_SECONDS:
            return (
                f"nearest ally {self.nearest_ally_m:.0f}m away, "
                f"{self.support_seconds:.0f}s at best — nobody could have helped"
            )
        return (
            f"nearest ally {self.nearest_ally_m:.0f}m away, "
            f"reachable in {self.support_seconds:.0f}s"
        )


@dataclass(frozen=True)
class PlayerReport:
    hero_id: int
    player_name: str | None
    team_num: int | None
    start_lane: int | None
    kills: int
    deaths: int
    assists: int
    kill_participation: float | None
    isolated_deaths: int | None
    deaths_by_phase: dict[str, int]
    final_net_worth: int | None
    final_level: int | None
    last_hits: int | None
    denies: int | None
    hero_damage: int | None
    objective_damage: int | None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PhaseStat:
    """One player's contribution within one phase.

    Whole-match totals hide the shape of a game: a player can farm a large net
    worth in the first twenty minutes and contribute nothing to the fights that
    decided it, or the reverse. Every figure here is a *delta* across the phase,
    not a running total.
    """

    hero_id: int
    team_num: int | None
    phase: str
    start_tick: int
    end_tick: int
    kills: int
    deaths: int
    assists: int
    net_worth_gained: int | None
    hero_damage: int | None
    objective_damage: int | None
    last_hits: int | None
    damage_share: float | None


@dataclass(frozen=True)
class EconomySnapshot:
    tick: int
    net_worth_by_team: dict[int, int]
    lead_team: int | None
    lead: int


@dataclass(frozen=True)
class TeamLeadPeak:
    """The largest positive economy lead one team held, even if it later lost."""

    team_num: int
    tick: int
    lead: int


@dataclass(frozen=True)
class MatchArcWindow:
    """A fixed clock window that forces the report to cover the whole match."""

    label: str
    start_tick: int
    end_tick: int
    net_worth_by_team: dict[int, int]
    kills_by_team: dict[int, int]
    structures_by_team: dict[int, int]
    mid_boss_by_team: dict[int, int]


# ---------------------------------------------------------------- phases


def phases(md: MatchData) -> list[Phase]:
    """Segment the match into laning / mid / late.

    Boundaries are derived from what actually changed how the map was played,
    not from fixed clock times:

    - **Laning ends** at the first cross-lane kill — the first time a hero dies
      to someone who did not start in their lane, i.e. the first rotation. Build
      10854 has no Guardians, so there is no lane tower falling to mark this;
      the first rotation is the real signal and it is in the kill feed.
    - **Mid game ends** at the first Walker destroyed.

    Each step falls back to the clock when the data to derive it is absent.
    """
    end = md.end_tick
    assert md.clock is not None
    clock = md.clock

    laning_end = _first_cross_lane_kill(md)
    mid_end = _first_objective_tick(md, ("walker", "guardian"))

    trigger_laning = "first cross-lane kill (first rotation)"
    trigger_mid = "first Walker destroyed"
    if laning_end is None:
        laning_end = _tick_at(clock, FALLBACK_LANING_END_SECONDS, end)
        trigger_laning = "10:00 (no rotation detected)"
    if mid_end is None or mid_end <= laning_end:
        mid_end = max(laning_end, _tick_at(clock, FALLBACK_MIDGAME_END_SECONDS, end))
        trigger_mid = "25:00 (no Walker data)"

    laning_end = min(laning_end, end) if end else laning_end
    mid_end = min(mid_end, end) if end else mid_end

    out = [Phase("laning", 0, laning_end, trigger_laning)]
    if mid_end > laning_end:
        out.append(Phase("mid game", laning_end, mid_end, trigger_mid))
    if end > mid_end:
        out.append(Phase("late game", mid_end, end, "match end"))
    return out


def _tick_at(clock, seconds: float, end_tick: int) -> int:
    tick = int(seconds * clock.tick_rate)
    return min(tick, end_tick) if end_tick else tick


def _first_objective_tick(md: MatchData, types: tuple[str, ...]) -> int | None:
    df = md.objectives
    if df.is_empty():
        return None
    dead = df.filter(
        pl.col("health").is_not_null()
        & (pl.col("health") <= 0)
        & pl.col("objective_type").str.to_lowercase().is_in(list(types))
        # Neutral objectives (Mid Boss) are not lane structures.
        & pl.col("team_num").is_in(md.team_nums)
    )
    return None if dead.is_empty() else int(dead["tick"].min())


def _first_cross_lane_kill(md: MatchData) -> int | None:
    """First kill where killer and victim started in different lanes.

    A 2v2 inside one lane is still laning. Someone showing up from another lane
    is the moment the map opens up, and it is the earliest reliable marker we
    have that does not depend on a structure falling.
    """
    if md.kills.is_empty() or md.players.is_empty():
        return None
    lanes = {
        row["hero_id"]: row["start_lane"]
        for row in md.players.iter_rows(named=True)
        if row.get("hero_id") is not None
    }
    for row in md.kills.sort("tick").iter_rows(named=True):
        attacker, victim = row.get("attacker_hero_id"), row.get("victim_hero_id")
        if attacker not in lanes or victim not in lanes or attacker == victim:
            continue
        if lanes[attacker] is not None and lanes[attacker] != lanes[victim]:
            return int(row["tick"])
    return None


def phase_at(phase_list: list[Phase], tick: int) -> str:
    for phase in phase_list:
        if phase.start_tick <= tick <= phase.end_tick:
            return phase.name
    return phase_list[-1].name if phase_list else "unknown"


def fixed_windows(md: MatchData, window_seconds: float = 600.0) -> list[Phase]:
    """Split a match into equal clock windows (10 minutes by default).

    Objective-derived phases describe map state, but can be very uneven — in
    match 12345678, ``late game`` spans 21:28–39:46 and hides a strong 21–33
    minute stretch inside the same bucket as the final collapse. Fixed windows
    are a second, deliberately boring view that prevents that compression.
    """
    if window_seconds <= 0:
        raise ValueError(f"window_seconds must be positive, got {window_seconds}")
    end = md.end_tick
    if end <= 0:
        return []
    assert md.clock is not None

    duration = md.clock.seconds(end)
    boundaries = [0]
    target = window_seconds
    while target < duration:
        tick = _tick_for_clock_seconds(md, target)
        if tick > boundaries[-1]:
            boundaries.append(tick)
        target += window_seconds
    if boundaries[-1] != end:
        boundaries.append(end)

    return [
        Phase(
            name=f"{md.clock.mmss(start)}–{md.clock.mmss(stop)}",
            start_tick=start,
            end_tick=stop,
            trigger=f"fixed {window_seconds / 60:g}-minute window",
        )
        for start, stop in zip(boundaries, boundaries[1:])
    ]


def _tick_for_clock_seconds(md: MatchData, seconds: float) -> int:
    """Invert the pause-aware clock with a small monotonic binary search."""
    assert md.clock is not None
    low, high = 0, md.end_tick
    while low < high:
        mid = (low + high) // 2
        if md.clock.seconds(mid) < seconds:
            low = mid + 1
        else:
            high = mid
    return low


# ------------------------------------------------------------- teamfights


def analyze_fights(md: MatchData, names: Names | None = None) -> list[FightSummary]:
    """Score each detected teamfight and check whether it bought anything.

    The verdict is kill difference inside the fight window. "Won the fight but
    took nothing" is the most common losing pattern in Deadlock, so conversion
    into an objective is tracked separately from winning.
    """
    names = names or Names.from_boon()
    if md.teamfights.is_empty():
        return []
    phase_list = phases(md)
    assert md.clock is not None

    out: list[FightSummary] = []
    for row in _merged_teamfight_rows(md):
        start, end = int(row["start_tick"]), int(row["end_tick"])
        participants = [p for p in (row.get("participants") or []) if p is not None]

        by_team: dict[int, list[int]] = {}
        for hero in participants:
            team = md.team_of(hero)
            if team is not None:
                by_team.setdefault(team, []).append(hero)
        for team in by_team:
            by_team[team].sort()

        kills_by_team = _kills_by_team(md, start, end, set(participants))
        winner, verdict = _verdict(md, kills_by_team)
        converted = _conversion(md, names, end, winner)
        conversion_assessment = _conversion_assessment(md, names, end, winner, converted)

        sizes = [len(by_team.get(t, [])) for t in md.team_nums]
        engagement = "v".join(str(s) for s in sizes) if sizes else "unknown"

        out.append(
            FightSummary(
                fight_id=int(row["fight_id"]),
                start_tick=start,
                end_tick=end,
                duration_seconds=float(row.get("duration_seconds") or 0.0),
                participants_by_team=by_team,
                kills_by_team=kills_by_team,
                winner=winner,
                verdict=verdict,
                engagement=engagement,
                hero_damage=int(row.get("hero_damage") or 0),
                converted_into=converted,
                conversion_assessment=conversion_assessment,
                phase=phase_at(phase_list, start),
            )
        )
    return out


def _merged_teamfight_rows(md: MatchData) -> list[dict]:
    """Merge adjacent fragments that still describe the same engagement."""
    if md.teamfights.is_empty():
        return []
    gap = max(1, int(FIGHT_MERGE_GAP_SECONDS * md.tick_rate))
    merged: list[dict] = []

    for raw in md.teamfights.sort("start_tick").iter_rows(named=True):
        row = dict(raw)
        row["participants"] = sorted(
            {int(hero) for hero in (row.get("participants") or []) if hero is not None}
        )
        row["hero_damage"] = int(row.get("hero_damage") or 0)
        row["start_tick"] = int(row["start_tick"])
        row["end_tick"] = int(row["end_tick"])

        if merged:
            previous = merged[-1]
            shared = set(previous["participants"]) & set(row["participants"])
            if row["start_tick"] <= previous["end_tick"] + gap and len(shared) >= 2:
                previous["end_tick"] = max(previous["end_tick"], row["end_tick"])
                previous["participants"] = sorted(
                    set(previous["participants"]) | set(row["participants"])
                )
                previous["hero_damage"] += row["hero_damage"]
                previous["duration_seconds"] = (
                    md.clock.seconds(previous["end_tick"])
                    - md.clock.seconds(previous["start_tick"])
                )
                continue

        row["duration_seconds"] = (
            md.clock.seconds(row["end_tick"]) - md.clock.seconds(row["start_tick"])
        )
        merged.append(row)
    return merged


def _kills_by_team(
    md: MatchData,
    start: int,
    end: int,
    participants: set[int] | None = None,
) -> dict[int, int]:
    counts = {team: 0 for team in md.team_nums}
    if md.kills.is_empty():
        return counts
    window = md.kills.filter((pl.col("tick") >= start) & (pl.col("tick") <= end))
    for row in window.iter_rows(named=True):
        # Several lane fights can overlap in clock time.  A global time-window
        # count credits the same kill to every one of them; the victim must
        # belong to this cluster before it can decide this fight.
        victim = row.get("victim_hero_id")
        if participants is not None and victim not in participants:
            continue
        # Credit the kill to the *victim's opponents*, not the attacker's team:
        # that way environment and suicide deaths still score correctly.
        victim_team = md.team_of(victim)
        for team in counts:
            if team != victim_team:
                counts[team] += 1
    return counts


def _verdict(md: MatchData, kills_by_team: dict[int, int]) -> tuple[int | None, str]:
    if not kills_by_team or all(v == 0 for v in kills_by_team.values()):
        return None, "no kills (poke / standoff)"
    ranked = sorted(kills_by_team.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) < 2 or ranked[0][1] == ranked[1][1]:
        return None, f"even trade ({'-'.join(str(v) for _, v in ranked)})"
    winner, top = ranked[0]
    return winner, f"won {top}-{ranked[1][1]}"


def _conversion(md: MatchData, names: Names, end_tick: int, winner: int | None) -> str | None:
    """Did the fight's winner take an objective off the back of it?"""
    if winner is None or md.objectives.is_empty():
        return None
    assert md.clock is not None
    limit = md.clock.seconds(end_tick) + CONVERSION_WINDOW_SECONDS
    dead = md.objectives.filter(
        pl.col("health").is_not_null()
        & (pl.col("health") <= 0)
        & (pl.col("tick") >= end_tick)
        # Neutral objectives belong to nobody, so taking one is not a conversion
        # of a fight against the enemy's structures.
        & pl.col("team_num").is_in(md.team_nums)
    ).sort("tick")
    for row in dead.iter_rows(named=True):
        if md.clock.seconds(int(row["tick"])) > limit:
            break
        # The objective belongs to the loser, so its owner must not be the winner.
        if row.get("team_num") != winner:
            return names.objective(row.get("objective_type"))
    return None


def _conversion_assessment(
    md: MatchData,
    names: Names,
    end_tick: int,
    winner: int | None,
    converted: str | None,
) -> ConversionAssessment | None:
    """Judge whether a fight win actually exposed a structure.

    Objective deaths answer what happened after the fight. This function
    answers the missing counterfactual: was there a creep wave and a healthy
    winner close enough to push at all? It never calls a structure available
    from hero positions alone, because heroes cannot safely substitute for a
    lane wave.
    """
    if winner is None:
        return None
    if converted is not None:
        return ConversionAssessment(
            status="converted",
            target=converted,
            lane=None,
            sample_tick=end_tick,
            allied_front_troopers=None,
            enemy_contesting_troopers=None,
            wave_distance=None,
            nearest_winner_distance=None,
            winner_distances={},
            reason=f"actually took {converted} within {CONVERSION_WINDOW_SECONDS:g}s",
        )
    if md.trooper_samples.is_empty():
        return ConversionAssessment(
            status="unknown",
            target=None,
            lane=None,
            sample_tick=None,
            allied_front_troopers=None,
            enemy_contesting_troopers=None,
            wave_distance=None,
            nearest_winner_distance=None,
            winner_distances={},
            reason="trooper samples unavailable; push feasibility is unknown",
        )

    structures = _next_enemy_lane_structures(md, winner, end_tick)
    if not structures:
        return ConversionAssessment(
            status="no_structure_window",
            target=None,
            lane=None,
            sample_tick=end_tick,
            allied_front_troopers=0,
            enemy_contesting_troopers=0,
            wave_distance=None,
            nearest_winner_distance=None,
            winner_distances={},
            reason="no surviving enemy lane structure was available",
            alternatives=_conversion_alternatives(md, winner, end_tick),
        )

    samples = md.trooper_samples.filter(
        (pl.col("tick") >= end_tick)
        & (
            pl.col("tick")
            <= end_tick + int(CONVERSION_SETUP_LOOKAHEAD_SECONDS * md.tick_rate)
        )
    )
    if samples.is_empty():
        return ConversionAssessment(
            status="unknown",
            target=None,
            lane=None,
            sample_tick=None,
            allied_front_troopers=None,
            enemy_contesting_troopers=None,
            wave_distance=None,
            nearest_winner_distance=None,
            winner_distances={},
            reason="no trooper sample was close enough to the fight ending",
        )

    winners = _healthy_winner_positions(md, winner, end_tick)
    candidates: list[tuple[int, float, float, ConversionAssessment]] = []
    for structure in structures:
        lane = structure["lane"]
        sx, sy = structure["x"], structure["y"]
        target = f"{names.lane(lane)}-lane {names.objective(structure['objective_type'])}"
        winner_distances = {
            hero: hypot(x - sx, y - sy) for hero, (x, y) in winners.items()
        }
        nearest_winner = min(winner_distances.values()) if winner_distances else None
        for (tick,), frame in samples.sort("tick").group_by("tick", maintain_order=True):
            allied = _live_lane_troopers(frame, winner, lane)
            if not allied:
                continue
            distances = [hypot(x - sx, y - sy) for x, y in allied]
            wave_distance = min(distances)
            front = [point for point, distance in zip(allied, distances) if distance <= CONVERSION_WAVE_SETUP_DISTANCE]
            if not front:
                continue
            enemy_team = structure["team_num"]
            enemies = _live_lane_troopers(frame, enemy_team, lane)
            contesting = sum(
                1
                for ex, ey in enemies
                if any(hypot(ex - ax, ey - ay) <= CONVERSION_TROOPER_CONTEST_RANGE for ax, ay in front)
            )
            enough_wave = len(front) >= CONVERSION_MIN_FRONT_TROOPERS
            unblocked = contesting <= len(front)
            hero_ready = nearest_winner is not None and nearest_winner <= CONVERSION_HERO_READY_DISTANCE
            hero_can_rotate = (
                nearest_winner is not None and nearest_winner <= CONVERSION_HERO_SETUP_DISTANCE
            )
            if (
                enough_wave
                and unblocked
                and wave_distance <= CONVERSION_WAVE_READY_DISTANCE
                and hero_ready
            ):
                status, rank = "push_now", 2
                reason = (
                    f"{len(front)} allied front troopers versus {contesting} contesting; "
                    f"wave {wave_distance:.0f} away and nearest healthy winner "
                    f"{nearest_winner:.0f} away"
                )
            elif enough_wave and unblocked and hero_can_rotate:
                status, rank = "setup_required", 1
                reason = (
                    f"wave needs setup: {len(front)} allied front troopers versus "
                    f"{contesting} contesting; wave {wave_distance:.0f} away and nearest "
                    f"healthy winner {nearest_winner:.0f} away"
                )
            else:
                continue
            assessment = ConversionAssessment(
                status=status,
                target=target,
                lane=lane,
                sample_tick=int(tick),
                allied_front_troopers=len(front),
                enemy_contesting_troopers=contesting,
                wave_distance=wave_distance,
                nearest_winner_distance=nearest_winner,
                winner_distances=winner_distances,
                reason=reason,
                alternatives=_conversion_alternatives(md, winner, end_tick),
            )
            candidates.append((rank, -wave_distance, -nearest_winner, assessment))

    if candidates:
        return max(candidates, key=lambda item: item[:3])[3]

    closest = _closest_wave_evidence(samples, structures, winner)
    reason = "no allied lane wave was close and strong enough to expose a structure"
    if closest is not None:
        lane, distance, count = closest
        reason += f"; closest was {count} {names.lane(lane)} front troopers {distance:.0f} away"
    return ConversionAssessment(
        status="no_structure_window",
        target=None,
        lane=None,
        sample_tick=end_tick,
        allied_front_troopers=None,
        enemy_contesting_troopers=None,
        wave_distance=None if closest is None else closest[1],
        nearest_winner_distance=None,
        winner_distances={},
        reason=reason,
        alternatives=_conversion_alternatives(md, winner, end_tick),
    )


def _next_enemy_lane_structures(md: MatchData, winner: int, tick: int) -> list[dict]:
    """Return the outermost living enemy structure in every lane."""
    if md.objectives.is_empty():
        return []
    priorities = {"guardian": 0, "walker": 1, "barracks": 2, "base_guardian": 3}
    rows = md.objectives.filter(
        pl.col("team_num").is_in([team for team in md.team_nums if team != winner])
        & pl.col("lane").is_not_null()
        & (pl.col("lane") > 0)
        & (pl.col("lane") < 1_000_000)
        & pl.col("objective_type").str.to_lowercase().is_in(list(priorities))
    )
    if rows.is_empty():
        return []
    keys = ["entity_id"] if rows["entity_id"].null_count() < rows.height else [
        "objective_type", "team_num", "lane"
    ]
    living: list[dict] = []
    for _, history in rows.sort("tick").group_by(keys, maintain_order=True):
        known = history.filter(pl.col("tick") <= tick)
        if known.is_empty():
            continue
        last = known.tail(1).row(0, named=True)
        if last.get("health") is not None and int(last["health"]) <= 0:
            continue
        if last.get("x") is None or last.get("y") is None:
            continue
        living.append(
            {
                "objective_type": str(last.get("objective_type") or "objective").lower(),
                "team_num": int(last["team_num"]),
                "lane": int(last["lane"]),
                "x": float(last["x"]),
                "y": float(last["y"]),
            }
        )
    chosen: dict[int, dict] = {}
    for structure in living:
        lane = structure["lane"]
        if lane not in chosen or priorities[structure["objective_type"]] < priorities[
            chosen[lane]["objective_type"]
        ]:
            chosen[lane] = structure
    return list(chosen.values())


def _healthy_winner_positions(
    md: MatchData, winner: int, tick: int
) -> dict[int, tuple[float, float]]:
    frame = _samples_at(md, tick)
    if frame is None:
        return {}
    out: dict[int, tuple[float, float]] = {}
    for row in frame.iter_rows(named=True):
        hero = row.get("hero_id")
        health, maximum = row.get("health"), row.get("max_health")
        if md.team_of(hero) != winner or not row.get("is_alive", True):
            continue
        if row.get("x") is None or row.get("y") is None or not maximum:
            continue
        if float(health or 0) / float(maximum) < CONVERSION_MIN_HEALTH_PCT:
            continue
        out[int(hero)] = (float(row["x"]), float(row["y"]))
    return out


def _live_lane_troopers(frame: pl.DataFrame, team: int, lane: int) -> list[tuple[float, float]]:
    rows = frame.filter(
        (pl.col("team_num") == team)
        & (pl.col("lane") == lane)
        & (pl.col("health").fill_null(0) > 0)
        & (pl.col("trooper_type").fill_null("") != "trooper_boss")
        & pl.col("x").is_not_null()
        & pl.col("y").is_not_null()
    )
    return [(float(x), float(y)) for x, y in rows.select(["x", "y"]).iter_rows()]


def _closest_wave_evidence(
    samples: pl.DataFrame, structures: list[dict], winner: int
) -> tuple[int, float, int] | None:
    best: tuple[int, float, int] | None = None
    for _, frame in samples.sort("tick").group_by("tick", maintain_order=True):
        for structure in structures:
            troops = _live_lane_troopers(frame, winner, structure["lane"])
            if not troops:
                continue
            distances = [
                hypot(x - structure["x"], y - structure["y"]) for x, y in troops
            ]
            distance = min(distances)
            front = sum(d <= CONVERSION_WAVE_SETUP_DISTANCE for d in distances)
            candidate = (structure["lane"], distance, front)
            if best is None or candidate[1] < best[1]:
                best = candidate
    return best


def _conversion_alternatives(md: MatchData, winner: int, tick: int) -> tuple[str, ...]:
    alternatives: list[str] = []
    if _mid_boss_available(md, tick):
        alternatives.append("Mid Boss was available; clear speed and enemy re-entry are not modeled")
    return tuple(alternatives)


def _mid_boss_available(md: MatchData, tick: int) -> bool:
    if md.mid_boss.is_empty() or md.clock is None:
        return False
    if md.clock.seconds(tick) < MID_BOSS_FIRST_AVAILABLE_SECONDS:
        return False
    prior = md.mid_boss.filter(pl.col("tick") <= tick).sort("tick")
    if prior.is_empty():
        return False
    spawns = prior.filter(pl.col("event").str.to_lowercase() == "spawned")
    if spawns.is_empty():
        return False
    last_spawn = int(spawns["tick"].max())
    killed = prior.filter(
        (pl.col("tick") >= last_spawn) & (pl.col("event").str.to_lowercase() == "killed")
    )
    return killed.is_empty()


# ------------------------------------------------------- kills in context


def kill_contexts(md: MatchData, constants: GameConstants | None = None) -> list[KillContext]:
    """Classify every death by the numbers on the ground when it happened.

    ``defenders_nearby`` counts the victim's *living teammates* within
    ISOLATION_RADIUS at the moment of death. When positional samples are not
    available for that tick, the field is ``None`` — unknown, not zero. Silently
    reporting "died alone" for a demo we never sampled would be worse than
    reporting nothing.
    """
    if md.kills.is_empty():
        return []
    constants = constants or load_constants()
    phase_list = phases(md)
    out: list[KillContext] = []

    for row in md.kills.sort("tick").iter_rows(named=True):
        tick = int(row["tick"])
        victim = row.get("victim_hero_id")
        attacker = row.get("attacker_hero_id")
        assisters = [a for a in (row.get("assister_hero_ids") or []) if a not in (None, NO_HERO)]
        # hero_id 0 is a trooper/objective, not a hero on the ground.
        real_attacker = attacker if attacker not in (None, NO_HERO, victim) else None
        attackers = len({*([real_attacker] if real_attacker is not None else []), *assisters})

        nearby = _living_allies_near(md, tick, victim)
        outnumbered = None if nearby is None else attackers > nearby + 1
        isolated = None if nearby is None else nearby == 0
        gap_m, support_s = _nearest_ally_reach(md, tick, victim, constants)

        out.append(
            KillContext(
                tick=tick,
                victim_hero_id=victim,
                attacker_hero_id=real_attacker,
                attackers=attackers,
                defenders_nearby=nearby,
                outnumbered=outnumbered,
                isolated=isolated,
                phase=phase_at(phase_list, tick),
                nearest_ally_m=gap_m,
                support_seconds=support_s,
            )
        )
    return out


def _nearest_ally_reach(
    md: MatchData,
    tick: int,
    victim: int | None,
    constants: GameConstants,
) -> tuple[float | None, float | None]:
    """Distance to the closest living teammate, and their best-case travel time.

    The teammate's own hero constants are used, not the victim's: the roster
    runs three different dash durations since 2026-07-28, and the question is
    how fast *the helper* moves.
    """
    if victim is None:
        return None, None
    frame = _samples_at(md, tick)
    if frame is None or frame["x"].null_count() == frame.height:
        return None, None
    me = frame.filter(pl.col("hero_id") == victim)
    if me.is_empty() or me["x"][0] is None:
        return None, None
    vx, vy = float(me["x"][0]), float(me["y"][0])

    allies = md.teammates_of(victim)
    if not allies:
        return None, None
    living = frame.filter(
        pl.col("hero_id").is_in(allies)
        & pl.col("x").is_not_null()
        & pl.col("is_alive").fill_null(True)
    )
    best: tuple[float, float] | None = None
    for row in living.iter_rows(named=True):
        hero = constants.hero(row["hero_id"])
        if hero is None:
            continue
        gap = distance_m(vx, vy, float(row["x"]), float(row["y"]))
        # A teammate close enough to be in the same fight is presumed to have
        # been shot at, so they do not get the out-of-combat sprint bonus.
        estimate = travel_seconds(gap, hero, sprinting=gap > ISOLATION_RADIUS / 39.37)
        if best is None or estimate.seconds < best[1]:
            best = (gap, estimate.seconds)
    if best is None:
        return None, None
    return round(best[0], 1), round(best[1], 1)


def _samples_at(md: MatchData, tick: int) -> pl.DataFrame | None:
    """Rows from the nearest sampled tick, or None if nothing is close enough."""
    samples = md.player_samples
    if samples.is_empty() or samples["tick"].null_count() == samples.height:
        return None
    exact = samples.filter(pl.col("tick") == tick)
    if not exact.is_empty():
        return exact
    ticks = samples["tick"].drop_nulls().unique().to_list()
    if not ticks:
        return None
    nearest = min(ticks, key=lambda t: abs(t - tick))
    if abs(nearest - tick) > SAMPLE_TOLERANCE_TICKS:
        return None
    return samples.filter(pl.col("tick") == nearest)


def _living_allies_near(md: MatchData, tick: int, victim: int | None) -> int | None:
    if victim is None:
        return None
    frame = _samples_at(md, tick)
    if frame is None or frame["x"].null_count() == frame.height:
        return None
    me = frame.filter(pl.col("hero_id") == victim)
    if me.is_empty() or me["x"][0] is None:
        return None
    vx, vy = float(me["x"][0]), float(me["y"][0])

    allies = md.teammates_of(victim)
    if not allies:
        return None
    near = frame.filter(
        pl.col("hero_id").is_in(allies)
        & pl.col("x").is_not_null()
        & (pl.col("is_alive").fill_null(True))
        & (((pl.col("x") - vx) ** 2 + (pl.col("y") - vy) ** 2).sqrt() <= ISOLATION_RADIUS)
    )
    return int(near.height)


# ---------------------------------------------------------- player reports


def player_reports(md: MatchData) -> list[PlayerReport]:
    """One coaching-oriented row per player.

    K/D/A comes from the kill feed rather than the scoreboard columns, so it
    stays correct on partial demos where the last sampled tick predates the end
    of the match.
    """
    if md.players.is_empty():
        return []
    phase_list = phases(md)
    contexts = {c.tick: c for c in kill_contexts(md)}
    final = _final_samples(md)

    team_kills: dict[int, int] = {t: 0 for t in md.team_nums}
    if not md.kills.is_empty():
        for row in md.kills.iter_rows(named=True):
            victim_team = md.team_of(row.get("victim_hero_id"))
            for team in team_kills:
                if team != victim_team:
                    team_kills[team] += 1

    out: list[PlayerReport] = []
    for prow in md.players.sort(["team_num", "hero_id"]).iter_rows(named=True):
        hero = prow.get("hero_id")
        if hero is None:
            continue
        team = prow.get("team_num")

        kills = deaths = assists = 0
        isolated = 0
        saw_position = False
        by_phase: dict[str, int] = {}

        if not md.kills.is_empty():
            for krow in md.kills.iter_rows(named=True):
                attacker = krow.get("attacker_hero_id")
                victim = krow.get("victim_hero_id")
                helpers = krow.get("assister_hero_ids") or []
                if attacker == hero and victim != hero:
                    kills += 1
                if hero in helpers:
                    assists += 1
                if victim == hero:
                    deaths += 1
                    tick = int(krow["tick"])
                    phase = phase_at(phase_list, tick)
                    by_phase[phase] = by_phase.get(phase, 0) + 1
                    ctx = contexts.get(tick)
                    if ctx is not None and ctx.isolated is not None:
                        saw_position = True
                        isolated += int(ctx.isolated)

        total = team_kills.get(team, 0) if team is not None else 0
        participation = (kills + assists) / total if total else None
        stats = final.get(hero, {})

        report = PlayerReport(
            hero_id=hero,
            player_name=prow.get("player_name"),
            team_num=team,
            start_lane=prow.get("start_lane"),
            kills=kills,
            deaths=deaths,
            assists=assists,
            kill_participation=participation,
            isolated_deaths=isolated if saw_position else None,
            deaths_by_phase=by_phase,
            final_net_worth=stats.get("net_worth"),
            final_level=stats.get("level"),
            last_hits=stats.get("last_hits"),
            denies=stats.get("denies"),
            hero_damage=stats.get("hero_damage"),
            objective_damage=stats.get("objective_damage"),
        )
        out.append(_annotate(report))
    return out


def _final_samples(md: MatchData) -> dict[int, dict]:
    """Last sampled row per hero, with net worth reconstructed."""
    samples = md.player_samples
    if samples.is_empty():
        return {}
    last = samples.sort("tick").group_by("hero_id", maintain_order=True).last()
    out: dict[int, dict] = {}
    for row in last.iter_rows(named=True):
        hero = row.get("hero_id")
        if hero is None:
            continue
        souls, spent = row.get("souls"), row.get("spent_souls")
        # Net worth is current souls plus everything already spent; `souls`
        # alone drops to near zero after a shopping trip and would rank a
        # well-itemised player last.
        net_worth = None if souls is None else int(souls) + int(spent or 0)
        out[hero] = {
            "net_worth": net_worth,
            "level": row.get("level"),
            "last_hits": row.get("last_hits"),
            "denies": row.get("denies"),
            "hero_damage": row.get("hero_damage"),
            "objective_damage": row.get("objective_damage"),
        }
    return out


# Coaching heuristics. These are intentionally conservative: a note only fires
# on a pattern clear enough that a human coach would also call it out.
ISOLATED_DEATH_SHARE = 0.5
LOW_PARTICIPATION = 0.45
HIGH_PARTICIPATION = 0.75
LANING_DEATH_LIMIT = 3


def _annotate(report: PlayerReport) -> PlayerReport:
    notes: list[str] = []
    if report.isolated_deaths is not None and report.deaths >= 3:
        share = report.isolated_deaths / report.deaths
        if share >= ISOLATED_DEATH_SHARE:
            notes.append(
                f"{report.isolated_deaths}/{report.deaths} deaths came with no living "
                f"teammate nearby — repeated positioning error, not bad fights."
            )
    if report.kill_participation is not None:
        if report.kill_participation < LOW_PARTICIPATION:
            notes.append(
                f"Kill participation {report.kill_participation:.0%} — present for "
                f"under half of the team's kills; check rotation timing."
            )
        elif report.kill_participation >= HIGH_PARTICIPATION:
            notes.append(f"Kill participation {report.kill_participation:.0%} — anchored the map.")
    laning_deaths = report.deaths_by_phase.get("laning", 0)
    if laning_deaths >= LANING_DEATH_LIMIT:
        notes.append(f"{laning_deaths} deaths in the laning phase — lane was lost early.")
    return PlayerReport(**{**report.__dict__, "notes": notes})


# ---------------------------------------------------------- phase splits


def phase_stats(md: MatchData) -> list[PhaseStat]:
    """Per-player, per-phase contribution across the whole match.

    Cumulative columns (`hero_damage`, `last_hits`, souls) are differenced at
    the phase boundaries. A hero with no sample at a boundary carries their last
    known value forward — dead players emit no rows at all, and treating that as
    zero would show them losing net worth.
    """
    return _period_stats(md, phases(md))


def fixed_window_stats(md: MatchData, window_seconds: float = 600.0) -> list[PhaseStat]:
    """Per-player deltas in equal clock windows, independent of map phases."""
    return _period_stats(md, fixed_windows(md, window_seconds))


def _period_stats(md: MatchData, periods: list[Phase]) -> list[PhaseStat]:
    if md.players.is_empty() or not periods:
        return []

    samples = md.player_samples
    per_hero = (
        {h: samples.filter(pl.col("hero_id") == h).sort("tick") for h in md.hero_ids}
        if not samples.is_empty()
        else {}
    )

    def value_at(hero: int, tick: int, column: str) -> int | None:
        frame = per_hero.get(hero)
        if frame is None or frame.is_empty() or column not in frame.columns:
            return None
        rows = frame.filter((pl.col("tick") <= tick) & pl.col(column).is_not_null())
        return None if rows.is_empty() else int(rows[column][-1])

    def delta(hero: int, start: int, end: int, column: str) -> int | None:
        before, after = value_at(hero, start, column), value_at(hero, end, column)
        if after is None:
            return None
        # No sample before the phase began means the player started from zero.
        return after - (before or 0)

    out: list[PhaseStat] = []
    for phase in periods:
        # Team hero damage in this phase, for the share column.
        team_damage: dict[int, int] = {}
        per_hero_damage: dict[int, int | None] = {}
        for hero in md.hero_ids:
            dealt = delta(hero, phase.start_tick, phase.end_tick, "hero_damage")
            per_hero_damage[hero] = dealt
            team = md.team_of(hero)
            if team is not None and dealt is not None:
                team_damage[team] = team_damage.get(team, 0) + dealt

        for prow in md.players.sort(["team_num", "hero_id"]).iter_rows(named=True):
            hero = prow.get("hero_id")
            if hero is None:
                continue
            team = prow.get("team_num")
            kills = deaths = assists = 0
            if not md.kills.is_empty():
                window = md.kills.filter(
                    (pl.col("tick") > phase.start_tick) & (pl.col("tick") <= phase.end_tick)
                )
                for krow in window.iter_rows(named=True):
                    attacker, victim = krow.get("attacker_hero_id"), krow.get("victim_hero_id")
                    helpers = krow.get("assister_hero_ids") or []
                    if attacker == hero and victim != hero and attacker != NO_HERO:
                        kills += 1
                    if hero in helpers:
                        assists += 1
                    if victim == hero:
                        deaths += 1

            dealt = per_hero_damage.get(hero)
            pool = team_damage.get(team) if team is not None else None
            share = dealt / pool if dealt is not None and pool else None
            souls = delta(hero, phase.start_tick, phase.end_tick, "souls")
            spent = delta(hero, phase.start_tick, phase.end_tick, "spent_souls")
            net_worth = None if souls is None else souls + (spent or 0)

            out.append(
                PhaseStat(
                    hero_id=hero,
                    team_num=team,
                    phase=phase.name,
                    start_tick=phase.start_tick,
                    end_tick=phase.end_tick,
                    kills=kills,
                    deaths=deaths,
                    assists=assists,
                    net_worth_gained=net_worth,
                    hero_damage=dealt,
                    objective_damage=delta(
                        hero, phase.start_tick, phase.end_tick, "objective_damage"
                    ),
                    last_hits=delta(hero, phase.start_tick, phase.end_tick, "last_hits"),
                    damage_share=share,
                )
            )
    return out


def fight_record_by_phase(md: MatchData, names: Names | None = None) -> dict[str, dict[int, int]]:
    """Decisive fights won per team, per phase — the shape of the whole match."""
    record: dict[str, dict[int, int]] = {p.name: {t: 0 for t in md.team_nums} for p in phases(md)}
    for fight in analyze_fights(md, names):
        if fight.winner is not None and fight.phase in record:
            record[fight.phase][fight.winner] = record[fight.phase].get(fight.winner, 0) + 1
    return record


# -------------------------------------------------------------- economy


def economy_curve(md: MatchData) -> list[EconomySnapshot]:
    """Team net worth at every sampled tick, plus who led and by how much."""
    samples = md.player_samples
    if samples.is_empty() or samples["souls"].null_count() == samples.height:
        return []

    team_map = pl.DataFrame(
        {
            "hero_id": md.hero_ids,
            "team_num": [md.team_of(h) for h in md.hero_ids],
        },
        schema={"hero_id": pl.Int64, "team_num": pl.Int64},
    )

    # A dead player emits no row at all in `player_ticks` — verified on match
    # 12345678, where late-game ticks carry 7 of 12 heroes. Summing the rows
    # present would make a dead player's entire net worth vanish and reappear,
    # which showed up as 150k phantom swings. Carry each hero's last known value
    # forward across the sampled ticks instead.
    ticks = samples.select(pl.col("tick").unique().sort())
    grid = ticks.join(team_map, how="cross")
    per_hero = (
        grid.join(
            samples.select(["tick", "hero_id", "souls", "spent_souls"]),
            on=["tick", "hero_id"],
            how="left",
        )
        .with_columns(
            (pl.col("souls").fill_null(0) + pl.col("spent_souls").fill_null(0)).alias("net_worth")
        )
        .with_columns(
            pl.when(pl.col("souls").is_null())
            .then(None)
            .otherwise(pl.col("net_worth"))
            .alias("net_worth")
        )
        .sort(["hero_id", "tick"])
        .with_columns(pl.col("net_worth").forward_fill().over("hero_id"))
        # Before a hero's first sample they genuinely had nothing.
        .with_columns(pl.col("net_worth").fill_null(0))
    )
    joined = (
        per_hero.group_by(["tick", "team_num"], maintain_order=True)
        .agg(pl.col("net_worth").sum())
        .sort("tick")
    )

    out: list[EconomySnapshot] = []
    for (tick,), group in joined.group_by(["tick"], maintain_order=True):
        by_team = {
            int(r["team_num"]): int(r["net_worth"])
            for r in group.iter_rows(named=True)
            if r["team_num"] is not None
        }
        if len(by_team) < 2:
            continue
        ranked = sorted(by_team.items(), key=lambda kv: kv[1], reverse=True)
        lead = ranked[0][1] - ranked[1][1]
        out.append(
            EconomySnapshot(
                tick=int(tick),
                net_worth_by_team=by_team,
                lead_team=ranked[0][0] if lead > 0 else None,
                lead=lead,
            )
        )
    return out


def peak_leads(curve: list[EconomySnapshot]) -> list[TeamLeadPeak]:
    """Return every team's best lead, not only the match's largest final lead.

    Reporting only ``max(snapshot.lead)`` systematically hides the losing
    team's advantage. A comeback can only be understood when both peaks are
    visible.
    """
    teams = sorted({team for snapshot in curve for team in snapshot.net_worth_by_team})
    out: list[TeamLeadPeak] = []
    for team in teams:
        best: TeamLeadPeak | None = None
        for snapshot in curve:
            own = snapshot.net_worth_by_team.get(team)
            others = [
                value for other, value in snapshot.net_worth_by_team.items() if other != team
            ]
            if own is None or not others:
                continue
            lead = own - max(others)
            if lead > 0 and (best is None or lead > best.lead):
                best = TeamLeadPeak(team_num=team, tick=snapshot.tick, lead=lead)
        if best is not None:
            out.append(best)
    return out


def match_arc(md: MatchData, window_seconds: float = 600.0) -> list[MatchArcWindow]:
    """Summarise economy, kills and map conversion in fixed clock windows."""
    windows = fixed_windows(md, window_seconds)
    if not windows:
        return []
    curve = economy_curve(md)

    # One destruction per entity. Rows after the first zero are duplicates.
    dead = md.objectives.filter(
        pl.col("health").is_not_null()
        & (pl.col("health") <= 0)
        & pl.col("team_num").is_in(md.team_nums)
    )
    if not dead.is_empty():
        keys = ["entity_id"] if dead["entity_id"].null_count() < dead.height else [
            "objective_type",
            "team_num",
            "lane",
        ]
        dead = dead.sort("tick").group_by(keys, maintain_order=True).first()

    def worth_at(tick: int) -> dict[int, int]:
        prior = [snapshot for snapshot in curve if snapshot.tick <= tick]
        if prior:
            return dict(prior[-1].net_worth_by_team)
        return {team: 0 for team in md.team_nums}

    out: list[MatchArcWindow] = []
    for index, window in enumerate(windows):
        # The left edge belongs to the preceding window, except at match start.
        lower = pl.col("tick") >= window.start_tick if index == 0 else pl.col("tick") > window.start_tick
        upper = pl.col("tick") <= window.end_tick

        kills = {team: 0 for team in md.team_nums}
        for row in md.kills.filter(lower & upper).iter_rows(named=True):
            victim_team = md.team_of(row.get("victim_hero_id"))
            for team in kills:
                if team != victim_team:
                    kills[team] += 1

        structures = {team: 0 for team in md.team_nums}
        if not dead.is_empty():
            for row in dead.filter(lower & upper).iter_rows(named=True):
                owners_opponents = [team for team in md.team_nums if team != row.get("team_num")]
                if len(owners_opponents) == 1:
                    structures[owners_opponents[0]] += 1

        boss = {team: 0 for team in md.team_nums}
        if not md.mid_boss.is_empty():
            used = md.mid_boss.filter(
                lower & upper & (pl.col("event") == "used") & pl.col("team_num").is_in(md.team_nums)
            )
            for team in used["team_num"].drop_nulls().unique().to_list():
                boss[int(team)] = 1

        out.append(
            MatchArcWindow(
                label=window.name,
                start_tick=window.start_tick,
                end_tick=window.end_tick,
                net_worth_by_team=worth_at(window.end_tick),
                kills_by_team=kills,
                structures_by_team=structures,
                mid_boss_by_team=boss,
            )
        )
    return out


def biggest_swing(curve: list[EconomySnapshot]) -> tuple[EconomySnapshot, EconomySnapshot] | None:
    """The steepest sustained change in soul lead, as a (from, to) pair.

    Signed against a fixed team so that a lead flipping sides registers as the
    huge swing it is, rather than cancelling out.
    """
    if len(curve) < 2:
        return None
    anchor = min(curve[0].net_worth_by_team)

    def signed(snapshot: EconomySnapshot) -> int:
        values = snapshot.net_worth_by_team
        others = [v for t, v in values.items() if t != anchor]
        return values.get(anchor, 0) - (max(others) if others else 0)

    best: tuple[EconomySnapshot, EconomySnapshot] | None = None
    best_delta = 0
    for before, after in zip(curve, curve[1:]):
        delta = abs(signed(after) - signed(before))
        if delta > best_delta:
            best_delta, best = delta, (before, after)
    return best
