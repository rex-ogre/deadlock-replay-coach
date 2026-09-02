"""Player-perspective counterfactual opportunity detection.

The replay knows every hero's real position. A player did not. This module is
the information firewall between those two facts: opportunity detectors only
receive enemies that the observer's team could plausibly know about, plus
short-lived last-seen memories. Hidden current positions never enter scoring.

This is deliberately an evidence-based heuristic, not a combat simulator. A
window means "worth considering from the information available then", never
"this kill or camp was guaranteed".
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from math import hypot

from .gamedata import GameConstants, HeroConstants, load_constants
from .match import MatchData
from .names import LANE_SENTINEL, NO_HERO, Names
from .physics import HU_PER_METER, IMPLAUSIBLE_SPEED, hammer_units
from .physics import travel_seconds as physics_travel
from .tactics import FightSummary, analyze_fights, phase_at, phases

# Stand-in for a hero the assets API has never heard of -- a brand new release,
# or a replay old enough to predate a rename. Valve's own default kit.
_DEFAULT_HERO = HeroConstants(
    hero_id=0,
    name="unknown",
    max_move_speed=6.7,
    sprint_speed=1.6,
    crouch_speed=4.75,
    move_acceleration=4.0,
    stamina=3.0,
    stamina_regen_per_second=0.2222,
    ground_dash_distance_m=10.0,
    ground_dash_duration=0.68,
    air_dash_distance_m=8.0,
    air_dash_duration=0.47,
)

# Without VPK collision geometry we cannot do exact ray casts. Proximity is
# therefore named a range model and can never, on its own, produce high
# confidence. Actual hero-vs-hero damage is stronger direct evidence.
TEAM_VISION_RANGE = 2_800.0
LAST_SEEN_SECONDS = 8.0
COMBAT_REVEAL_SECONDS = 3.0

KILL_MAX_RANGE = 2_600.0
LOCAL_FIGHT_RANGE = 2_500.0
KILL_FOLLOWUP_SECONDS = 12.0

CAMP_CLUSTER_RADIUS = 750.0
CAMP_VISIBLE_RANGE = 2_800.0
CAMP_ROUTE_RANGE = 6_500.0
CAMP_DANGER_RANGE = 3_500.0
CAMP_ENEMY_AWAY_RANGE = 4_500.0

# A rotation is a decision to leave your own ground. Anything closer than the
# kill range is the same fight, not a rotation; anything past the travel cap
# arrives after the fight it was meant to join.
ROTATION_MIN_TRAVEL = 2_600.0
ROTATION_MAX_TRAVEL = 9_000.0
ROTATION_MIN_HEALTH = 0.60
ROTATION_TARGET_HEALTH = 0.70
ROTATION_ARRIVAL_RANGE = 2_500.0
ROTATION_FOLLOWUP_SECONDS = 25.0
ROTATION_REACTION_SECONDS = 0.75
ROTATION_SPEED_PERCENTILE = 0.85
# The clamp band used to be three flat hammer-unit numbers picked by eye. It is
# now derived per hero from the game's own constants (see :mod:`physics`),
# because the floor and ceiling both mean something specific: nobody rotates
# slower than they can sprint, and nobody sustains more than an outer transit
# line without the boost, which is not up for most of a match. Guessing high
# here is not a neutral choice -- it manufactures rotations the player is then
# blamed for skipping.
ROTATION_TELEPORT_SPEED = hammer_units(IMPLAUSIBLE_SPEED)
# Every window this detector considers is at least ROTATION_MIN_TRAVEL away --
# 2,600 hammer units, or 66 metres, which is cross-lane. At that range the map
# is built so a transit line is on the path, and assuming the player walks the
# whole way understates a real rotation badly enough to hide misses. Half the
# route on a line reproduces the flat 450 hu/s the detector was hand-tuned to
# before this model existed, but now it is a claim about the map rather than a
# number someone picked.
ROTATION_ZIPLINE_FRACTION = 0.5
ALLY_ENGAGE_RANGE = 3_000.0
LANE_MATCH_RANGE = 3_500.0

# Macro windows run on public information only: the kill feed, structures that
# have already fallen, and the Mid Boss / Urn announcements. Nothing here needs
# a vision model, which is why these are the most trustworthy windows in the
# report.
MACRO_MIN_ADVANTAGE = 2
MACRO_MIN_WINDOW_SECONDS = 10.0
MACRO_STRONG_WINDOW_SECONDS = 18.0
# A team that lands less than this on structures during a man advantage did not
# convert it. One Walker hit by one hero for a few seconds clears ~700.
MACRO_OBJECTIVE_DAMAGE_USED = 750
MACRO_FREE_HEALTH = 0.55
MACRO_GRACE_SECONDS = 20.0

# The demo announces the Mid Boss entity spawning ~30 seconds in, which is when
# it is created in the world, not necessarily when it can be attacked. On match
# 98811241 that event lands at 00:30 and the boss is not touched until 33:51.
#
# What "attackable" means has *moved*. The 2026-05-22 gameplay update made the
# Mid Boss start the match already spawned; before that it was gated to 10:00.
# A single hardcoded gate therefore mislabels one era or the other, and demos
# carry no patch stamp.
#
# Match IDs do carry time, though: they are issued in order. Two points from a
# real account's history bracket the era —
#     88,338,851 -> 2026-06-12   100,107,054 -> 2026-08-18
# which is ~176k matches a day, putting 2026-05-22 near 84.7M. That is an
# extrapolation across a three-week gap, not a measurement, so it is stated as
# such wherever a Mid Boss window is reported and can be overridden per run.
MID_BOSS_AVAILABLE_SECONDS = 600.0
MID_BOSS_UNGATED_FROM_MATCH_ID = 84_700_000


def mid_boss_available_from(md: MatchData, override: float | None = None) -> float:
    """Match seconds after which the Mid Boss can be contested.

    Returns 0.0 for replays recorded after the boss was ungated. When the demo
    has no match ID we keep the old 10:00 gate, because under-reporting a macro
    window is a smaller error than inventing thirty minutes of them.
    """
    if override is not None:
        return override
    match_id = getattr(md, "match_id", None)
    if match_id and int(match_id) >= MID_BOSS_UNGATED_FROM_MATCH_ID:
        return 0.0
    return MID_BOSS_AVAILABLE_SECONDS

# Trooper chip damage on a Walker is continuous, so only a fast, deep drop is
# evidence of heroes actually sieging it.
DEFEND_LOSS_FRACTION = 0.25
DEFEND_WINDOW_SECONDS = 45.0
# A structure bursted down inside a couple of seconds is not a rotation anyone
# could have answered. Only a siege you had time to walk to is a missed call.
DEFEND_MIN_SECONDS = 8.0
DEFEND_AWAY_RANGE = 7_000.0


# Caveats that are true of *every* window a detector produces. They live here,
# once, instead of on each window: repeated on several hundred rows they stop
# being read at all and crowd out the row-specific facts that actually differ.
# Anything that varies from window to window stays in that window's own
# ``limitations``.
STANDING_CAVEATS: dict[str, tuple[str, ...]] = {
    "kill": (
        "aim, opponent reactions, ammo, and full damage simulation are not modeled",
    ),
    "rotation": (
        "travel time is estimated from that player's replay movement; route geometry, "
        "zip lines, mobility abilities, and what the original lane would lose are not modeled",
    ),
    "macro": (
        "respawn timing is read from the replay; in game you would read it from the "
        "scoreboard",
        "structure health, clear speed, and whether the survivors could defend are "
        "not modeled",
    ),
    "defend": (
        "trooper damage and hero damage to structures are not separated",
        "whether defending was survivable against the attackers present is not modeled",
    ),
    "jungle": (
        "lane-wave loss, exact clear time, and escape geometry are not modeled",
    ),
}


@dataclass(frozen=True)
class ObservedEnemy:
    hero_id: int
    x: float
    y: float
    health: int | None
    max_health: int | None
    is_alive: bool
    source: str
    age_seconds: float


@dataclass(frozen=True)
class ObservationFrame:
    """Everything an opportunity scorer may know about opponents at one tick."""

    tick: int
    observer_hero_id: int
    enemies: tuple[ObservedEnemy, ...]
    unknown_enemy_ids: tuple[int, ...]


@dataclass(frozen=True)
class KillOpportunity:
    observer_hero_id: int
    target_hero_id: int
    start_tick: int
    end_tick: int
    phase: str
    confidence: str
    score: int
    observation_basis: str
    target_health_pct: float
    observer_health_pct: float
    distance: float
    recent_damage_by_observer: int
    known_allies_nearby: int
    known_enemies_nearby: int
    unknown_enemy_count: int
    evidence: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class JungleOpportunity:
    observer_hero_id: int
    camp_id: int
    lane: int | None
    camp_x: float
    camp_y: float
    enemy_team_num: int
    start_tick: int
    end_tick: int
    confidence: str
    camp_status: str
    distance: float
    enemies_last_seen_away: int
    # Distance from the camp to the closest opponent the team could place.
    # None when no opponent had a known position at all.
    nearest_known_enemy: float | None
    unknown_enemy_count: int
    evidence: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class RotationOpportunity:
    """A moment when leaving your own ground to join a fight elsewhere was on.

    Scored only when a teammate was already engaged there and arriving would
    have changed the local count — a lone low enemy across the map is a chase,
    not a rotation.
    """

    observer_hero_id: int
    target_hero_id: int
    start_tick: int
    end_tick: int
    phase: str
    confidence: str
    score: int
    observation_basis: str
    from_lane: int | None
    to_lane: int | None
    travel_distance: float
    target_health_pct: float
    observer_health_pct: float
    allies_engaged: int
    known_enemies_there: int
    unknown_enemy_count: int
    estimated_travel_seconds: float
    seconds_before_first_ally_death: float
    actual_winner_team_num: int
    actual_kill_margin: int
    evidence: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class MacroOpportunity:
    """A map-state window: what the board offered while nobody took it.

    ``kind`` is what the window was for — ``siege``, ``mid_boss``, ``urn`` or
    ``defend``. Every field here comes from information both teams could see.
    """

    observer_hero_id: int
    kind: str
    action: str
    start_tick: int
    end_tick: int
    phase: str
    confidence: str
    score: int
    target_kind: str
    target_lane: int | None
    target_team_num: int | None
    # Other objectives that were on at the same moment. Ranking them against the
    # chosen one needs judgement the demo cannot supply, so they are named, not
    # scored.
    alternatives: tuple[str, ...]
    enemies_known_dead: int
    window_seconds: float
    team_objective_damage: int
    observer_objective_damage: int
    # None where the target has no coordinates in the demo (the Mid Boss).
    distance: float | None
    evidence: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class OpportunityAnalysis:
    kill_windows: tuple[KillOpportunity, ...]
    jungle_windows: tuple[JungleOpportunity, ...]
    rotation_windows: tuple[RotationOpportunity, ...]
    macro_windows: tuple[MacroOpportunity, ...]
    vision_model: str
    # Always-true caveats per window kind; see STANDING_CAVEATS.
    standing_caveats: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class _PlayerState:
    hero_id: int
    x: float
    y: float
    health: int | None
    max_health: int | None
    is_alive: bool


@dataclass(frozen=True)
class _KillPoint:
    tick: int
    observer: int
    target: int
    phase: str
    confidence: str
    score: int
    basis: str
    target_pct: float
    observer_pct: float
    distance: float
    recent_damage: int
    allies: int
    enemies: int
    unknown: int
    evidence: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class Camp:
    camp_id: int
    x: float
    y: float
    entity_ids: tuple[int, ...]
    first_tick: int


@dataclass(frozen=True)
class _JunglePoint:
    tick: int
    observer: int
    camp_id: int
    lane: int | None
    camp_x: float
    camp_y: float
    enemy_team: int
    confidence: str
    status: str
    distance: float
    away: int
    nearest: float | None
    unknown: int
    evidence: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class _RotationPoint:
    tick: int
    observer: int
    target: int
    phase: str
    confidence: str
    score: int
    basis: str
    from_lane: int | None
    to_lane: int | None
    travel: float
    target_pct: float
    observer_pct: float
    allies: int
    enemies: int
    unknown: int
    travel_seconds: float
    seconds_before_loss: float
    winner_team: int
    kill_margin: int
    evidence: tuple[str, ...]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class _Structure:
    entity_id: int
    objective_type: str
    team_num: int
    lane: int | None
    x: float
    y: float
    death_tick: int | None
    max_health: int | None


class _LaneMap:
    """Which lane a point on the map belongs to.

    Lanes are located from the structures that sit on them, both teams' ends
    included, so a point halfway down a lane still resolves to that lane. A
    point too far from every structure is off-lane (jungle, mid) and returns
    ``None`` rather than being forced into the nearest lane.
    """

    def __init__(self, md: MatchData) -> None:
        anchors: dict[int, list[tuple[float, float]]] = {}
        seen: set[int] = set()
        for row in md.objectives.iter_rows(named=True):
            lane, entity = row.get("lane"), row.get("entity_id")
            x, y = row.get("x"), row.get("y")
            if lane in (None, 0, LANE_SENTINEL) or x is None or y is None:
                continue
            if entity is not None:
                if int(entity) in seen:
                    continue
                seen.add(int(entity))
            anchors.setdefault(int(lane), []).append((float(x), float(y)))
        self._anchors = anchors

    def __bool__(self) -> bool:
        return bool(self._anchors)

    def lane_of(self, x: float, y: float) -> int | None:
        best: int | None = None
        best_distance = LANE_MATCH_RANGE
        for lane, points in self._anchors.items():
            distance = min(_distance(x, y, px, py) for px, py in points)
            if distance <= best_distance:
                best, best_distance = lane, distance
        return best


class _CounterSeries:
    """Step lookup over a cumulative per-hero counter in the samples."""

    def __init__(self, md: MatchData, column: str) -> None:
        series: dict[int, tuple[list[int], list[int]]] = {}
        if column in md.player_samples.columns and not md.player_samples.is_empty():
            rows: dict[int, list[tuple[int, int]]] = {}
            for row in md.player_samples.sort("tick").iter_rows(named=True):
                hero, tick, value = row.get("hero_id"), row.get("tick"), row.get(column)
                if hero is None or tick is None or value is None:
                    continue
                rows.setdefault(int(hero), []).append((int(tick), int(value)))
            series = {
                hero: ([r[0] for r in history], [r[1] for r in history])
                for hero, history in rows.items()
            }
        self._series = series

    def at(self, hero: int, tick: int) -> int:
        ticks, values = self._series.get(hero, ([], []))
        at = bisect_right(ticks, tick)
        return values[at - 1] if at else 0

    def delta(self, heroes: list[int], start: int, end: int) -> int:
        return sum(max(0, self.at(h, end) - self.at(h, start)) for h in heroes)


class _DamageIndex:
    def __init__(self, md: MatchData) -> None:
        pairs: dict[tuple[int, int], list[tuple[int, int]]] = {}
        involving: dict[int, list[int]] = {}
        if not md.damage.is_empty():
            for row in md.damage.sort("tick").iter_rows(named=True):
                tick = row.get("tick")
                attacker = row.get("attacker_hero_id")
                victim = row.get("victim_hero_id")
                if tick is None or attacker in (None, NO_HERO) or victim in (None, NO_HERO):
                    continue
                tick = int(tick)
                attacker = int(attacker)
                victim = int(victim)
                pairs.setdefault((attacker, victim), []).append(
                    (tick, max(0, int(row.get("damage") or 0)))
                )
                involving.setdefault(attacker, []).append(tick)
                involving.setdefault(victim, []).append(tick)

        self._pair_ticks: dict[tuple[int, int], list[int]] = {}
        self._pair_prefix: dict[tuple[int, int], list[int]] = {}
        for key, events in pairs.items():
            ticks: list[int] = []
            prefix = [0]
            for tick, amount in events:
                ticks.append(tick)
                prefix.append(prefix[-1] + amount)
            self._pair_ticks[key] = ticks
            self._pair_prefix[key] = prefix
        self._involving = involving

    def dealt(self, attacker: int, victim: int, tick: int, window_ticks: int) -> int:
        key = (attacker, victim)
        ticks = self._pair_ticks.get(key, [])
        if not ticks:
            return 0
        left = bisect_left(ticks, tick - window_ticks)
        right = bisect_right(ticks, tick)
        prefix = self._pair_prefix[key]
        return prefix[right] - prefix[left]

    def recently_in_combat(self, hero: int, tick: int, window_ticks: int) -> bool:
        ticks = self._involving.get(hero, [])
        if not ticks:
            return False
        at = bisect_right(ticks, tick)
        return at > 0 and ticks[at - 1] >= tick - window_ticks


def analyze_opportunities(
    md: MatchData,
    *,
    max_per_player: int | None = None,
    constants: GameConstants | None = None,
) -> OpportunityAnalysis:
    """Return conservative kill and jungle decision windows for every player."""
    constants = constants or load_constants()
    observations = {
        hero: observation_frames(md, hero)
        for hero in md.hero_ids
    }
    kills = _kill_opportunities(md, observations, max_per_player=max_per_player)
    jungle = _jungle_opportunities(md, observations, max_per_player=max_per_player)
    rotations = _rotation_opportunities(
        md, observations, max_per_player=max_per_player, constants=constants
    )
    macro = _macro_opportunities(md, max_per_player=max_per_player)
    return OpportunityAnalysis(
        kill_windows=tuple(kills),
        jungle_windows=tuple(jungle),
        rotation_windows=tuple(rotations),
        macro_windows=tuple(macro),
        vision_model=(
            "team-shared information model: direct combat plus a range-based visibility "
            "estimate and 8-second last-seen memory; map occlusion is not reconstructed"
        ),
        standing_caveats=STANDING_CAVEATS,
    )


def observation_frames(md: MatchData, observer_hero_id: int) -> list[ObservationFrame]:
    """Build the observer's knowledge timeline without exposing hidden live state.

    Current enemy state is copied into the knowledge cache only after direct
    combat with the observer's team or proximity to a living teammate. Once an
    enemy disappears, the cache freezes; later frames receive that stale copy,
    never the replay's hidden current coordinates or health.
    """
    team = md.team_of(observer_hero_id)
    if team is None:
        return []
    samples = _states_by_tick(md)
    if not samples:
        return []
    teammates = {h for h in md.hero_ids if md.team_of(h) == team}
    enemies = {h for h in md.hero_ids if md.team_of(h) not in (None, team)}
    reveal_ticks = _team_combat_reveals(md, team)
    reveal_window = max(1, int(COMBAT_REVEAL_SECONDS * md.tick_rate))
    memory_ticks = max(1, int(LAST_SEEN_SECONDS * md.tick_rate))
    memory: dict[int, tuple[int, _PlayerState, str]] = {}
    out: list[ObservationFrame] = []

    for tick, states in sorted(samples.items()):
        allied_states = [
            states[h] for h in teammates
            if h in states and states[h].is_alive
        ]
        known: list[ObservedEnemy] = []
        unknown: list[int] = []
        for enemy in sorted(enemies):
            state = states.get(enemy)
            source: str | None = None
            if state is not None and state.is_alive:
                last_combat = reveal_ticks.get(enemy)
                # reveal_ticks stores all events; select the latest no later than this sample.
                if last_combat:
                    at = bisect_right(last_combat, tick)
                    if at and last_combat[at - 1] >= tick - reveal_window:
                        source = "direct combat"
                if source is None and any(
                    _distance(ally.x, ally.y, state.x, state.y) <= TEAM_VISION_RANGE
                    for ally in allied_states
                ):
                    source = "range model"

            if source is not None and state is not None:
                memory[enemy] = (tick, state, source)
                known.append(_observed(enemy, state, source, 0.0))
                continue

            previous = memory.get(enemy)
            if previous is not None and tick - previous[0] <= memory_ticks:
                seen_tick, seen_state, seen_source = previous
                age = (tick - seen_tick) / md.tick_rate
                known.append(_observed(enemy, seen_state, f"last seen ({seen_source})", age))
            else:
                unknown.append(enemy)
        out.append(
            ObservationFrame(
                tick=tick,
                observer_hero_id=observer_hero_id,
                enemies=tuple(known),
                unknown_enemy_ids=tuple(unknown),
            )
        )
    return out


def _observed(hero: int, state: _PlayerState, source: str, age: float) -> ObservedEnemy:
    return ObservedEnemy(
        hero_id=hero,
        x=state.x,
        y=state.y,
        health=state.health,
        max_health=state.max_health,
        is_alive=state.is_alive,
        source=source,
        age_seconds=age,
    )


def _states_by_tick(md: MatchData) -> dict[int, dict[int, _PlayerState]]:
    out: dict[int, dict[int, _PlayerState]] = {}
    if md.player_samples.is_empty():
        return out
    for row in md.player_samples.sort(["tick", "hero_id"]).iter_rows(named=True):
        tick, hero = row.get("tick"), row.get("hero_id")
        x, y = row.get("x"), row.get("y")
        if tick is None or hero is None or x is None or y is None:
            continue
        out.setdefault(int(tick), {})[int(hero)] = _PlayerState(
            hero_id=int(hero),
            x=float(x),
            y=float(y),
            health=_int_or_none(row.get("health")),
            max_health=_int_or_none(row.get("max_health")),
            is_alive=bool(row.get("is_alive") if row.get("is_alive") is not None else True),
        )
    return out


def _team_combat_reveals(md: MatchData, observer_team: int) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    if md.damage.is_empty():
        return out
    for row in md.damage.sort("tick").iter_rows(named=True):
        tick = row.get("tick")
        attacker = row.get("attacker_hero_id")
        victim = row.get("victim_hero_id")
        if tick is None or attacker in (None, NO_HERO) or victim in (None, NO_HERO):
            continue
        attacker_team = md.team_of(int(attacker))
        victim_team = md.team_of(int(victim))
        if attacker_team == observer_team and victim_team not in (None, observer_team):
            out.setdefault(int(victim), []).append(int(tick))
        if victim_team == observer_team and attacker_team not in (None, observer_team):
            out.setdefault(int(attacker), []).append(int(tick))
    return out


def _kill_opportunities(
    md: MatchData,
    observations: dict[int, list[ObservationFrame]],
    *,
    max_per_player: int,
) -> list[KillOpportunity]:
    states = _states_by_tick(md)
    damage = _DamageIndex(md)
    damage_window = max(1, int(4.0 * md.tick_rate))
    phase_list = phases(md)
    lane_by_hero = {
        int(r["hero_id"]): r.get("start_lane")
        for r in md.players.iter_rows(named=True)
        if r.get("hero_id") is not None
    }
    points: list[_KillPoint] = []

    for observer, frames in observations.items():
        team = md.team_of(observer)
        if team is None:
            continue
        for frame in frames:
            tick_states = states.get(frame.tick, {})
            own = tick_states.get(observer)
            if own is None or not own.is_alive:
                continue
            own_pct = _health_pct(own.health, own.max_health)
            if own_pct is None or own_pct < 0.45:
                continue

            for target in frame.enemies:
                # Last-seen memory may guide rotations, but it may not supply a
                # target's current HP for a kill claim.
                if target.age_seconds > 0 or not target.is_alive:
                    continue
                target_pct = _health_pct(target.health, target.max_health)
                if target_pct is None or target_pct <= 0 or target_pct > 0.35:
                    continue
                distance = _distance(own.x, own.y, target.x, target.y)
                if distance > KILL_MAX_RANGE:
                    continue

                recent = damage.dealt(observer, target.hero_id, frame.tick, damage_window)
                current_phase = phase_at(phase_list, frame.tick)
                # Without a recent interaction by this player, declaring a
                # personal kill window needs a full hero/aim simulator we do
                # not have. The single exception is a laning target visibly at
                # low health within one step of the observer: that read is
                # positional rather than mechanical, and missing it is exactly
                # the "I could have gone in" mistake laners ask about. It can
                # never reach high confidence — the `high` test below requires
                # committed damage.
                uncommitted = recent <= 0
                if uncommitted and not (
                    current_phase == "laning"
                    and target_pct <= 0.25
                    and own_pct >= 0.60
                    and distance <= 1_500
                ):
                    continue
                known_enemies = sum(
                    1
                    for enemy in frame.enemies
                    if enemy.age_seconds <= 4.0
                    and enemy.is_alive
                    and _distance(enemy.x, enemy.y, target.x, target.y) <= LOCAL_FIGHT_RANGE
                )
                allies = sum(
                    1
                    for hero, state in tick_states.items()
                    if md.team_of(hero) == team
                    and state.is_alive
                    and _distance(state.x, state.y, target.x, target.y) <= LOCAL_FIGHT_RANGE
                )
                if allies < known_enemies:
                    continue

                score = 4 if target_pct <= 0.20 else 3 if target_pct <= 0.35 else 2
                score += 2 if own_pct >= 0.65 else 1 if own_pct >= 0.45 else 0
                score += 2 if distance <= 1_200 else 1 if distance <= 2_000 else 0
                score += 2 if recent > 0 else 0
                score += 1 if allies >= known_enemies else -2
                if (
                    current_phase == "laning"
                    and lane_by_hero.get(observer) is not None
                    and lane_by_hero.get(observer) == lane_by_hero.get(target.hero_id)
                ):
                    score += 1
                if target.source == "direct combat":
                    score += 1

                if current_phase != "laning" and len(frame.unknown_enemy_ids) > 3:
                    continue
                minimum = 8
                if score < minimum:
                    continue
                high = (
                    recent > 0
                    and target.health is not None
                    and recent >= max(1, target.health) * 0.5
                    and target_pct <= 0.25
                    and own_pct >= 0.60
                    and allies >= known_enemies
                    and score >= 10
                    and len(frame.unknown_enemy_ids) <= (2 if current_phase == "laning" else 1)
                )
                confidence = "high" if high else "medium"
                evidence = [
                    f"target at {target_pct:.0%} health",
                    f"observer at {own_pct:.0%} health",
                    f"known local numbers {allies}v{known_enemies}",
                ]
                if recent:
                    evidence.append(f"observer dealt {recent} damage in the previous 4s")
                else:
                    evidence.append(
                        f"no committed trade — target was {distance:.0f} units away in the open"
                    )
                limitations = [f"{len(frame.unknown_enemy_ids)} enemies were not currently known"]
                if uncommitted:
                    limitations.append(
                        "you had not engaged this target; whether your kit could close the gap "
                        "in time is not modeled"
                    )
                if target.source == "range model":
                    limitations.append("visibility inferred by range; walls and occlusion are unavailable")
                points.append(
                    _KillPoint(
                        tick=frame.tick,
                        observer=observer,
                        target=target.hero_id,
                        phase=current_phase,
                        confidence=confidence,
                        score=score,
                        basis=target.source,
                        target_pct=target_pct,
                        observer_pct=own_pct,
                        distance=distance,
                        recent_damage=recent,
                        allies=allies,
                        enemies=known_enemies,
                        unknown=len(frame.unknown_enemy_ids),
                        evidence=tuple(evidence),
                        limitations=tuple(limitations),
                    )
                )

    # Treat pressure within the same ~15 second fight as one branch, even if
    # the best target changes halfway through it.
    windows = _dedupe_by_observer(md, _group_kill_points(md, points), 15.0)
    return _limit_by_player(windows, max_per_player)


def _group_kill_points(md: MatchData, points: list[_KillPoint]) -> list[KillOpportunity]:
    gap = max(1, int(2.5 * md.tick_rate))
    followup = max(1, int(KILL_FOLLOWUP_SECONDS * md.tick_rate))
    deaths: dict[int, list[int]] = {}
    if not md.kills.is_empty():
        for row in md.kills.iter_rows(named=True):
            if row.get("tick") is not None and row.get("victim_hero_id") is not None:
                deaths.setdefault(int(row["victim_hero_id"]), []).append(int(row["tick"]))
    for ticks in deaths.values():
        ticks.sort()

    out: list[KillOpportunity] = []
    grouped: list[_KillPoint] = []

    def flush() -> None:
        if not grouped:
            return
        start, end = grouped[0].tick, grouped[-1].tick
        target = grouped[0].target
        target_deaths = deaths.get(target, [])
        at = bisect_left(target_deaths, start)
        if at < len(target_deaths) and target_deaths[at] <= end + followup:
            return  # This pressure became an actual kill, not an additional missed window.
        best = max(grouped, key=lambda p: (_confidence_rank(p.confidence), p.score))
        display_radius = max(1, int(2.0 * md.tick_rate))
        display_start = max(start, best.tick - display_radius)
        display_end = min(end, best.tick + display_radius)
        out.append(
            KillOpportunity(
                observer_hero_id=best.observer,
                target_hero_id=best.target,
                start_tick=display_start,
                end_tick=display_end,
                phase=best.phase,
                confidence=best.confidence,
                score=best.score,
                observation_basis=best.basis,
                target_health_pct=best.target_pct,
                observer_health_pct=best.observer_pct,
                distance=best.distance,
                recent_damage_by_observer=best.recent_damage,
                known_allies_nearby=best.allies,
                known_enemies_nearby=best.enemies,
                unknown_enemy_count=best.unknown,
                evidence=best.evidence,
                limitations=best.limitations,
            )
        )

    for point in sorted(points, key=lambda p: (p.observer, p.target, p.tick)):
        if (
            grouped
            and (point.observer, point.target) == (grouped[-1].observer, grouped[-1].target)
            and point.tick - grouped[-1].tick <= gap
        ):
            grouped.append(point)
        else:
            flush()
            grouped = [point]
    flush()
    return out


def _jungle_opportunities(
    md: MatchData,
    observations: dict[int, list[ObservationFrame]],
    *,
    max_per_player: int,
) -> list[JungleOpportunity]:
    camps = neutral_camps(md)
    bases = team_bases(md)
    if not camps or len(bases) < 2:
        return []
    # Lane lookup scans objective entities; do it once per static camp, not for
    # every player/sample candidate.
    camp_lanes = {camp.camp_id: _nearest_lane(md, camp.x, camp.y) for camp in camps}
    states = _states_by_tick(md)
    damage = _DamageIndex(md)
    neutral_history = neutral_health_history(md)
    points: list[_JunglePoint] = []
    combat_window = max(1, int(3.0 * md.tick_rate))

    for observer, frames in observations.items():
        own_team = md.team_of(observer)
        if own_team not in bases:
            continue
        enemy_teams = [team for team in md.team_nums if team != own_team and team in bases]
        if not enemy_teams:
            continue
        enemy_team = enemy_teams[0]
        own_base, enemy_base = bases[own_team], bases[enemy_team]

        for frame in frames:
            tick_states = states.get(frame.tick, {})
            own = tick_states.get(observer)
            own_pct = None if own is None else _health_pct(own.health, own.max_health)
            if own is None or not own.is_alive or own_pct is None or own_pct < 0.60:
                continue
            if damage.recently_in_combat(observer, frame.tick, combat_window):
                continue
            allied_states = [
                state for hero, state in tick_states.items()
                if md.team_of(hero) == own_team and state.is_alive
            ]

            for camp in camps:
                # Static camp position is public map knowledge. Only sites
                # materially closer to the opposing base count as enemy jungle.
                if _distance(camp.x, camp.y, *enemy_base) + 500 >= _distance(
                    camp.x, camp.y, *own_base
                ):
                    continue
                if frame.tick < camp.first_tick:
                    continue
                route = _distance(own.x, own.y, camp.x, camp.y)
                if route > CAMP_ROUTE_RANGE:
                    continue

                recent_enemies = [e for e in frame.enemies if e.age_seconds <= LAST_SEEN_SECONDS]
                near = [
                    e for e in recent_enemies
                    if _distance(e.x, e.y, camp.x, camp.y) < CAMP_DANGER_RANGE
                ]
                if near:
                    continue
                camp_distances = sorted(
                    _distance(e.x, e.y, camp.x, camp.y) for e in recent_enemies
                )
                away = sum(d >= CAMP_ENEMY_AWAY_RANGE for d in camp_distances)
                if away < 2:
                    continue
                # Counting only the opponents that were comfortably far away
                # reports the reassuring half of the picture: on match 98811241
                # a camp was called clear with "3 opponents at least 4500 units
                # away" while the other three sat at 3979, 4147 and 4440. The
                # nearest known opponent is what actually decides whether the
                # camp is contestable, so it is measured and named.
                nearest = camp_distances[0] if camp_distances else None

                camp_visible = any(
                    _distance(ally.x, ally.y, camp.x, camp.y) <= CAMP_VISIBLE_RANGE
                    for ally in allied_states
                )
                if camp_visible:
                    if not camp_alive(camp, frame.tick, neutral_history):
                        continue
                    status = "visible and available"
                else:
                    status = "unknown — scout before committing"

                all_clear = nearest is not None and nearest >= CAMP_ENEMY_AWAY_RANGE
                score = away
                score += 2 if route <= 3_500 else 1
                score += 1 if own_pct >= 0.80 else 0
                score += 2 if camp_visible else 0
                score += 1 if all_clear else 0
                # `medium` now means what a reader assumes it means: nobody the
                # team could see was within reach of the camp. One opponent
                # loitering at 4000 units makes this a scout call, not a clear.
                confidence = (
                    "medium"
                    if camp_visible
                    and all_clear
                    and score >= 6
                    and away >= 3
                    and len(frame.unknown_enemy_ids) <= 2
                    else "low"
                )
                evidence = (
                    f"{len(recent_enemies)} of {len(recent_enemies) + len(frame.unknown_enemy_ids)} "
                    f"opponents had a known position; the nearest was "
                    f"{nearest:.0f} units from the camp"
                    if nearest is not None
                    else "no opponent had a known position",
                    f"{away} of them were at least {CAMP_ENEMY_AWAY_RANGE:.0f} units away",
                    f"observer at {own_pct:.0%} health and {route:.0f} units from camp",
                    f"camp status: {status}",
                )
                limitations = [f"{len(frame.unknown_enemy_ids)} enemy positions were unknown"]
                if not camp_visible:
                    limitations.append("the replay's hidden camp state was not used")
                points.append(
                    _JunglePoint(
                        tick=frame.tick,
                        observer=observer,
                        camp_id=camp.camp_id,
                        lane=camp_lanes[camp.camp_id],
                        camp_x=camp.x,
                        camp_y=camp.y,
                        enemy_team=enemy_team,
                        confidence=confidence,
                        status=status,
                        distance=route,
                        away=away,
                        nearest=nearest,
                        unknown=len(frame.unknown_enemy_ids),
                        evidence=evidence,
                        limitations=tuple(limitations),
                    )
                )

    windows = _dedupe_jungle_windows(md, _group_jungle_points(md, points))
    return _limit_by_player(windows, max_per_player)


def neutral_camps(md: MatchData) -> list[Camp]:
    if md.neutrals.is_empty():
        return []
    entities: dict[int, tuple[float, float, int]] = {}
    for row in md.neutrals.sort("tick").iter_rows(named=True):
        entity, tick = row.get("entity_id"), row.get("tick")
        x, y = row.get("x"), row.get("y")
        if entity is None or tick is None or x is None or y is None or int(entity) in entities:
            continue
        entities[int(entity)] = (float(x), float(y), int(tick))

    clusters: list[list[tuple[int, float, float, int]]] = []
    for entity, (x, y, tick) in sorted(entities.items(), key=lambda item: item[0]):
        for cluster in clusters:
            cx = sum(r[1] for r in cluster) / len(cluster)
            cy = sum(r[2] for r in cluster) / len(cluster)
            if _distance(x, y, cx, cy) <= CAMP_CLUSTER_RADIUS:
                cluster.append((entity, x, y, tick))
                break
        else:
            clusters.append([(entity, x, y, tick)])

    ordered = sorted(
        clusters,
        key=lambda c: (sum(r[1] for r in c) / len(c), sum(r[2] for r in c) / len(c)),
    )
    return [
        Camp(
            camp_id=i,
            x=sum(r[1] for r in cluster) / len(cluster),
            y=sum(r[2] for r in cluster) / len(cluster),
            entity_ids=tuple(sorted(r[0] for r in cluster)),
            first_tick=min(r[3] for r in cluster),
        )
        for i, cluster in enumerate(ordered, start=1)
    ]


def team_bases(md: MatchData) -> dict[int, tuple[float, float]]:
    if md.objectives.is_empty():
        return {}
    rows = [
        row for row in md.objectives.iter_rows(named=True)
        if row.get("team_num") in md.team_nums
        and row.get("x") is not None
        and row.get("y") is not None
        and str(row.get("objective_type") or "").lower() in ("patron", "shrine")
    ]
    if not rows:
        rows = [
            row for row in md.objectives.iter_rows(named=True)
            if row.get("team_num") in md.team_nums
            and row.get("x") is not None
            and row.get("y") is not None
        ]
    out: dict[int, tuple[float, float]] = {}
    for team in md.team_nums:
        # Collapse repeated health-change rows by entity before averaging.
        by_entity: dict[int, tuple[float, float]] = {}
        for row in rows:
            if row.get("team_num") == team and row.get("entity_id") is not None:
                by_entity[int(row["entity_id"])] = (float(row["x"]), float(row["y"]))
        if by_entity:
            out[team] = (
                sum(p[0] for p in by_entity.values()) / len(by_entity),
                sum(p[1] for p in by_entity.values()) / len(by_entity),
            )
    return out


def neutral_health_history(md: MatchData) -> dict[int, tuple[list[int], list[int]]]:
    rows: dict[int, list[tuple[int, int]]] = {}
    for row in md.neutrals.sort("tick").iter_rows(named=True):
        entity, tick = row.get("entity_id"), row.get("tick")
        if entity is None or tick is None:
            continue
        rows.setdefault(int(entity), []).append((int(tick), int(row.get("health") or 0)))
    return {
        entity: ([r[0] for r in history], [r[1] for r in history])
        for entity, history in rows.items()
    }


def camp_alive(
    camp: Camp,
    tick: int,
    history: dict[int, tuple[list[int], list[int]]],
) -> bool:
    for entity in camp.entity_ids:
        ticks, health = history.get(entity, ([], []))
        at = bisect_right(ticks, tick)
        if at and health[at - 1] > 0:
            return True
    return False


def _group_jungle_points(md: MatchData, points: list[_JunglePoint]) -> list[JungleOpportunity]:
    gap = max(1, int(7.0 * md.tick_rate))
    grouped: list[_JunglePoint] = []
    out: list[JungleOpportunity] = []

    def flush() -> None:
        if not grouped:
            return
        best = max(
            grouped,
            key=lambda p: (_confidence_rank(p.confidence), p.away, -p.distance),
        )
        out.append(
            JungleOpportunity(
                observer_hero_id=best.observer,
                camp_id=best.camp_id,
                lane=best.lane,
                camp_x=best.camp_x,
                camp_y=best.camp_y,
                enemy_team_num=best.enemy_team,
                start_tick=grouped[0].tick,
                end_tick=grouped[-1].tick,
                confidence=best.confidence,
                camp_status=best.status,
                distance=best.distance,
                enemies_last_seen_away=best.away,
                nearest_known_enemy=best.nearest,
                unknown_enemy_count=best.unknown,
                evidence=best.evidence,
                limitations=best.limitations,
            )
        )

    for point in sorted(points, key=lambda p: (p.observer, p.camp_id, p.tick)):
        if (
            grouped
            and (point.observer, point.camp_id) == (grouped[-1].observer, grouped[-1].camp_id)
            and point.tick - grouped[-1].tick <= gap
        ):
            grouped.append(point)
        else:
            flush()
            grouped = [point]
    flush()
    return out


def _dedupe_jungle_windows(
    md: MatchData, windows: list[JungleOpportunity]
) -> list[JungleOpportunity]:
    """Collapse nearby camp choices into one route decision."""
    # Invade routing is a macro decision. Nearby camp suggestions within a
    # minute are alternatives in the same route, not dozens of opportunities.
    margin = max(1, int(45.0 * md.tick_rate))
    chosen: list[JungleOpportunity] = []
    for window in sorted(
        windows,
        key=lambda w: (
            -_confidence_rank(w.confidence),
            -w.enemies_last_seen_away,
            w.distance,
            w.start_tick,
        ),
    ):
        conflicts = any(
            other.observer_hero_id == window.observer_hero_id
            and not (
                window.end_tick + margin < other.start_tick
                or window.start_tick > other.end_tick + margin
            )
            for other in chosen
        )
        if not conflicts:
            chosen.append(window)
    return sorted(chosen, key=lambda w: (w.observer_hero_id, w.start_tick))


# ------------------------------------------------------- rotations (ganks)


def _rotation_opportunities(
    md: MatchData,
    observations: dict[int, list[ObservationFrame]],
    *,
    max_per_player: int | None,
    constants: GameConstants,
) -> list[RotationOpportunity]:
    """Find moments the observer was free and a fight elsewhere was winnable.

    Four things must hold at once, or it is not a rotation you can be faulted
    for skipping: you were healthy and out of combat, a teammate was already
    engaged somewhere else, arriving would have made the local count favourable,
    and you did not in fact go.
    """
    lanes = _LaneMap(md)
    if not lanes:
        return []
    states = _states_by_tick(md)
    if not states:
        return []
    sample_ticks = sorted(states)
    damage = _DamageIndex(md)
    fights = analyze_fights(md, Names())
    if not fights:
        return []
    rotation_speeds = _rotation_speeds(md, states, sample_ticks, constants)
    phase_list = phases(md)
    combat_window = max(1, int(3.0 * md.tick_rate))
    followup = max(1, int(ROTATION_FOLLOWUP_SECONDS * md.tick_rate))
    points: list[_RotationPoint] = []

    for observer, frames in observations.items():
        team = md.team_of(observer)
        if team is None:
            continue
        for frame in frames:
            tick_states = states.get(frame.tick, {})
            own = tick_states.get(observer)
            if own is None or not own.is_alive:
                continue
            own_pct = _health_pct(own.health, own.max_health)
            if own_pct is None or own_pct < ROTATION_MIN_HEALTH:
                continue
            # Someone who is mid-trade is not free to leave, whatever the rest
            # of the map is offering.
            if damage.recently_in_combat(observer, frame.tick, combat_window):
                continue
            fresh = [e for e in frame.enemies if e.is_alive and e.age_seconds <= 4.0]
            if any(_distance(e.x, e.y, own.x, own.y) <= LOCAL_FIGHT_RANGE for e in fresh):
                continue
            from_lane = lanes.lane_of(own.x, own.y)

            for target in fresh:
                travel = _distance(own.x, own.y, target.x, target.y)
                if not ROTATION_MIN_TRAVEL <= travel <= ROTATION_MAX_TRAVEL:
                    continue
                to_lane = lanes.lane_of(target.x, target.y)
                if to_lane is None or (from_lane is not None and to_lane == from_lane):
                    continue
                target_pct = _health_pct(target.health, target.max_health)
                if target_pct is None or target_pct > ROTATION_TARGET_HEALTH:
                    continue

                allies_there = [
                    hero
                    for hero, state in tick_states.items()
                    if hero != observer
                    and md.team_of(hero) == team
                    and state.is_alive
                    and _distance(state.x, state.y, target.x, target.y) <= ALLY_ENGAGE_RANGE
                ]
                # No teammate there means this is a solo dive across the map,
                # which needs a fight simulator to judge. Not our claim to make.
                if not allies_there:
                    continue
                enemies_there = sum(
                    1
                    for enemy in fresh
                    if _distance(enemy.x, enemy.y, target.x, target.y) <= ALLY_ENGAGE_RANGE
                )
                # Your teammates already outnumber them: your arrival adds
                # nothing the report can call a missed opportunity.
                if len(allies_there) > enemies_there:
                    continue
                # Calling a still-outnumbered arrival "winnable" was too loose:
                # it faulted players for joining 4v6 last stands.  Showing up
                # must at least make the *known* local count even.
                if len(allies_there) + 1 < enemies_there:
                    continue
                if _went(states, sample_ticks, observer, target.x, target.y, frame.tick, followup):
                    continue

                fight = _swingable_fight(md, fights, observer, target.hero_id, team, frame.tick)
                if fight is None:
                    continue
                summary, first_ally_death, actual_allies, actual_enemies, kill_margin = fight
                # The replay's full participant set is used only for the
                # after-the-fact verdict.  If arrival still leaves the team
                # outnumbered, this was not a fair "you should have gone" call.
                if actual_allies + 1 < actual_enemies:
                    continue

                # Two readings of how fast this player crosses 66+ metres:
                # what the map allows, and what their own track shows they
                # actually sustain. The faster one is used, so a player is only
                # faulted when arriving in time was within reach on both.
                hero_profile = constants.hero(observer) or _DEFAULT_HERO
                modelled = physics_travel(
                    travel / HU_PER_METER,
                    hero_profile,
                    zipline=constants.zipline,
                    zipline_fraction=ROTATION_ZIPLINE_FRACTION,
                    reaction=ROTATION_REACTION_SECONDS,
                ).seconds
                observed_speed = rotation_speeds.get(observer)
                observed = (
                    travel / observed_speed + ROTATION_REACTION_SECONDS
                    if observed_speed
                    else modelled
                )
                travel_seconds = min(modelled, observed)
                seconds_before_loss = max(0.0, (first_ally_death - frame.tick) / md.tick_rate)
                if travel_seconds > seconds_before_loss:
                    continue

                score = 3 if target_pct <= 0.35 else 2 if target_pct <= 0.50 else 1
                score += 2 if travel <= 4_500 else 1 if travel <= 6_500 else 0
                score += 1 if own_pct >= 0.80 else 0
                score += 2 if len(allies_there) == enemies_there else 1
                score += 1 if target.source == "direct combat" else 0
                score += 1 if len(frame.unknown_enemy_ids) <= 2 else 0
                score += 2 if travel_seconds <= seconds_before_loss * 0.70 else 1
                if score < 7:
                    continue
                confidence = (
                    "high"
                    if (
                        target.source == "direct combat"
                        and target_pct <= 0.40
                        and travel <= 5_000
                        and own_pct >= 0.75
                        and len(frame.unknown_enemy_ids) <= 1
                        and travel_seconds <= seconds_before_loss * 0.80
                        and score >= 9
                    )
                    else "medium"
                    if score >= 8
                    else "low"
                )
                evidence = [
                    f"{len(allies_there)} teammate(s) already engaged {enemies_there} known "
                    f"opponent(s) there",
                    f"target at {target_pct:.0%} health, {travel:.0f} units away",
                    f"observer at {own_pct:.0%} health and out of combat for 3s",
                    f"estimated arrival {travel_seconds:.1f}s; first allied death came "
                    f"{seconds_before_loss:.1f}s later",
                    f"without the observer, the actual fight was lost by {kill_margin} kill",
                ]
                limitations = [f"{len(frame.unknown_enemy_ids)} enemy positions were unknown"]
                if target.source == "range model":
                    limitations.append(
                        "visibility inferred by range; walls and occlusion are unavailable"
                    )
                points.append(
                    _RotationPoint(
                        tick=frame.tick,
                        observer=observer,
                        target=target.hero_id,
                        phase=phase_at(phase_list, frame.tick),
                        confidence=confidence,
                        score=score,
                        basis=target.source,
                        from_lane=from_lane,
                        to_lane=to_lane,
                        travel=travel,
                        target_pct=target_pct,
                        observer_pct=own_pct,
                        allies=len(allies_there),
                        enemies=enemies_there,
                        unknown=len(frame.unknown_enemy_ids),
                        travel_seconds=travel_seconds,
                        seconds_before_loss=seconds_before_loss,
                        winner_team=int(summary.winner),
                        kill_margin=kill_margin,
                        evidence=tuple(evidence),
                        limitations=tuple(limitations),
                    )
                )

    windows = _group_rotation_points(md, points)
    return _limit_by_player(_dedupe_by_observer(md, windows, 20.0), max_per_player)


def _rotation_speeds(
    md: MatchData,
    states: dict[int, dict[int, _PlayerState]],
    sample_ticks: list[int],
    constants: GameConstants,
) -> dict[int, float]:
    """Estimate committed travel speed from each player's own replay track.

    The 85th percentile represents purposeful movement without pretending every
    route has a zip line. Large discontinuities are treated as teleports and
    excluded, then the estimate is clamped to a band derived from that hero's
    own constants: no slower than their sprint, no faster than an unboosted
    outer transit line.
    """
    speeds: dict[int, list[float]] = {}
    for before_tick, after_tick in zip(sample_ticks, sample_ticks[1:]):
        elapsed = (after_tick - before_tick) / md.tick_rate
        if elapsed <= 0 or elapsed > 2.5:
            continue
        before, after = states[before_tick], states[after_tick]
        for hero in set(before) & set(after):
            first, second = before[hero], after[hero]
            if not first.is_alive or not second.is_alive:
                continue
            speed = _distance(first.x, first.y, second.x, second.y) / elapsed
            if 0 < speed <= ROTATION_TELEPORT_SPEED:
                speeds.setdefault(hero, []).append(speed)

    ceiling = hammer_units(constants.zipline.speed_outer)
    out: dict[int, float] = {}
    for hero in md.hero_ids:
        profile = constants.hero(hero)
        floor = hammer_units(profile.sprint_total_speed) if profile else 327.0
        values = sorted(speeds.get(hero, []))
        if values:
            index = round((len(values) - 1) * ROTATION_SPEED_PERCENTILE)
            estimate = values[index]
        else:
            # No track to learn from: assume they sprint, and nothing more.
            estimate = floor
        out[hero] = min(ceiling, max(floor, estimate))
    return out


def _swingable_fight(
    md: MatchData,
    fights: list[FightSummary],
    observer: int,
    target: int,
    observer_team: int,
    tick: int,
) -> tuple[FightSummary, int, int, int, int] | None:
    """The actual fight was a one-kill loss the observer could plausibly swing."""
    padding = max(1, int(2.0 * md.tick_rate))
    candidates: list[tuple[FightSummary, int, int, int, int]] = []
    for fight in fights:
        participants = {
            hero for heroes in fight.participants_by_team.values() for hero in heroes
        }
        if observer in participants or target not in participants:
            continue
        if not (fight.start_tick - padding <= tick <= fight.end_tick):
            continue
        if fight.winner in (None, observer_team):
            continue
        own_kills = fight.kills_by_team.get(observer_team, 0)
        enemy_kills = fight.kills_by_team.get(int(fight.winner), 0)
        margin = enemy_kills - own_kills
        if margin != 1:
            continue
        allied_deaths = [
            int(row["tick"])
            for row in md.kills.iter_rows(named=True)
            if row.get("tick") is not None
            and fight.start_tick <= int(row["tick"]) <= fight.end_tick
            and md.team_of(row.get("victim_hero_id")) == observer_team
        ]
        if not allied_deaths:
            continue
        allies = len(fight.participants_by_team.get(observer_team, []))
        enemies = sum(
            len(heroes)
            for team, heroes in fight.participants_by_team.items()
            if team != observer_team
        )
        candidates.append((fight, min(allied_deaths), allies, enemies, margin))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0].end_tick, item[0].start_tick))


def _went(
    states: dict[int, dict[int, _PlayerState]],
    sample_ticks: list[int],
    observer: int,
    x: float,
    y: float,
    tick: int,
    window: int,
) -> bool:
    """Did the observer actually show up there in time?"""
    start = bisect_right(sample_ticks, tick)
    for later in sample_ticks[start:]:
        if later > tick + window:
            return False
        state = states.get(later, {}).get(observer)
        if state is not None and _distance(state.x, state.y, x, y) <= ROTATION_ARRIVAL_RANGE:
            return True
    return False


def _group_rotation_points(
    md: MatchData, points: list[_RotationPoint]
) -> list[RotationOpportunity]:
    gap = max(1, int(4.0 * md.tick_rate))
    grouped: list[_RotationPoint] = []
    out: list[RotationOpportunity] = []

    def flush() -> None:
        if not grouped:
            return
        best = max(grouped, key=lambda p: (_confidence_rank(p.confidence), p.score))
        out.append(
            RotationOpportunity(
                observer_hero_id=best.observer,
                target_hero_id=best.target,
                start_tick=grouped[0].tick,
                end_tick=grouped[-1].tick,
                phase=best.phase,
                confidence=best.confidence,
                score=best.score,
                observation_basis=best.basis,
                from_lane=best.from_lane,
                to_lane=best.to_lane,
                travel_distance=best.travel,
                target_health_pct=best.target_pct,
                observer_health_pct=best.observer_pct,
                allies_engaged=best.allies,
                known_enemies_there=best.enemies,
                unknown_enemy_count=best.unknown,
                estimated_travel_seconds=best.travel_seconds,
                seconds_before_first_ally_death=best.seconds_before_loss,
                actual_winner_team_num=best.winner_team,
                actual_kill_margin=best.kill_margin,
                evidence=best.evidence,
                limitations=best.limitations,
            )
        )

    for point in sorted(points, key=lambda p: (p.observer, p.target, p.tick)):
        if (
            grouped
            and (point.observer, point.target) == (grouped[-1].observer, grouped[-1].target)
            and point.tick - grouped[-1].tick <= gap
        ):
            grouped.append(point)
        else:
            flush()
            grouped = [point]
    flush()
    return out


def _dedupe_by_observer(md: MatchData, windows: list, margin_seconds: float) -> list:
    """One player takes one decision at a time; overlapping branches are one window.

    Keep the best-supported one and count the moment once, rather than inflating
    the opportunity count with every low-health enemy in the same fight or every
    reachable camp on the same route.
    """
    margin = max(1, int(margin_seconds * md.tick_rate))
    chosen: list = []
    for window in sorted(
        windows,
        key=lambda w: (-_confidence_rank(w.confidence), -getattr(w, "score", 0), w.start_tick),
    ):
        conflicts = any(
            other.observer_hero_id == window.observer_hero_id
            and not (
                window.end_tick + margin < other.start_tick
                or window.start_tick > other.end_tick + margin
            )
            for other in chosen
        )
        if not conflicts:
            chosen.append(window)
    return sorted(chosen, key=lambda w: (w.observer_hero_id, w.start_tick))


# ------------------------------------------------------------ macro windows


def _macro_opportunities(
    md: MatchData, *, max_per_player: int | None
) -> list[MacroOpportunity]:
    """What the map was offering while nobody cashed it in.

    Unlike the kill and jungle detectors, this one needs no vision model: every
    input is announced to both teams. The kill feed says who is dead, the
    structures that have fallen are on everyone's map, and the Mid Boss and Urn
    are broadcast events. That makes these the least speculative windows in the
    report — the judgement is not "could you see it" but "you saw it and the
    board stayed still".
    """
    states = _states_by_tick(md)
    if not states or not md.team_nums:
        return []
    ticks = sorted(states)
    phase_list = phases(md)
    structures = _structures(md)
    deaths = _death_windows(md, states, ticks)
    objective_damage = _CounterSeries(md, "objective_damage")
    damage = _DamageIndex(md)
    boss = _MidBossState(md)
    urn = _UrnState(md)

    out: list[MacroOpportunity] = []
    for team in md.team_nums:
        out.extend(
            _advantage_windows(
                md, team, states, ticks, phase_list, structures,
                deaths, objective_damage, damage, boss, urn,
            )
        )
    out.extend(_defence_windows(md, states, ticks, phase_list, structures, damage))
    return _limit_by_player(_dedupe_by_observer(md, out, 30.0), max_per_player)


def _structures(md: MatchData) -> list[_Structure]:
    """Every team structure with the tick it fell, if it fell."""
    if md.objectives.is_empty():
        return []
    first: dict[int, dict] = {}
    death: dict[int, int] = {}
    for row in md.objectives.sort("tick").iter_rows(named=True):
        entity, team = row.get("entity_id"), row.get("team_num")
        if entity is None or team not in md.team_nums:
            continue
        if row.get("x") is None or row.get("y") is None:
            continue
        entity = int(entity)
        first.setdefault(entity, row)
        health = row.get("health")
        if health is not None and int(health) <= 0 and entity not in death:
            death[entity] = int(row["tick"])
    out = []
    for entity, row in first.items():
        lane = row.get("lane")
        out.append(
            _Structure(
                entity_id=entity,
                objective_type=str(row.get("objective_type") or "objective").lower(),
                team_num=int(row["team_num"]),
                lane=None if lane in (None, 0, LANE_SENTINEL) else int(lane),
                x=float(row["x"]),
                y=float(row["y"]),
                death_tick=death.get(entity),
                max_health=_int_or_none(row.get("max_health")),
            )
        )
    return sorted(out, key=lambda s: s.entity_id)


def _death_windows(
    md: MatchData,
    states: dict[int, dict[int, _PlayerState]],
    ticks: list[int],
) -> dict[int, list[tuple[int, int]]]:
    """When each hero was dead, from the kill feed — public information.

    The death itself comes from the feed both teams read. The end of the window
    is read from the samples when they carry deaths at all; a sample stream that
    never reports a dead hero falls back to an estimate, which is flagged in the
    limitations of every window built on it.
    """
    out: dict[int, list[tuple[int, int]]] = {}
    if md.kills.is_empty():
        return out
    alive_ticks: dict[int, list[int]] = {}
    samples_carry_deaths = False
    for tick in ticks:
        for hero, state in states[tick].items():
            if state.is_alive:
                alive_ticks.setdefault(hero, []).append(tick)
            else:
                samples_carry_deaths = True

    for row in md.kills.sort("tick").iter_rows(named=True):
        victim, tick = row.get("victim_hero_id"), row.get("tick")
        if victim in (None, NO_HERO) or tick is None:
            continue
        victim, tick = int(victim), int(tick)
        if samples_carry_deaths:
            alive = alive_ticks.get(victim, [])
            at = bisect_right(alive, tick)
            revive = alive[at] if at < len(alive) else md.end_tick
        else:
            revive = tick + _estimated_respawn_ticks(md, tick)
        if revive > tick:
            out.setdefault(victim, []).append((tick, revive))
    return out


def _estimated_respawn_ticks(md: MatchData, tick: int) -> int:
    """A rough respawn length for demos whose samples never report a dead hero.

    Deliberately coarse: it only has to be good enough to tell a two-man window
    from a one-second blip, and every window built on it says so.
    """
    assert md.clock is not None
    minutes = md.clock.seconds(tick) / 60.0
    seconds = min(60.0, 10.0 + 1.3 * minutes)
    return max(1, int(seconds * md.tick_rate))


def _is_dead_at(deaths: dict[int, list[tuple[int, int]]], hero: int, tick: int) -> bool:
    return any(start <= tick < end for start, end in deaths.get(hero, ()))


class _MidBossState:
    """Mid Boss availability, position, and who took it.

    The demo credits the ``killed`` event to the neutral team, so the taker is
    read from the ``used`` event that follows it — the buff can only be used by
    the team that earned it. Availability starts at
    :data:`MID_BOSS_AVAILABLE_SECONDS`, not at the spawn event; see the note
    there for why the spawn tick cannot be trusted for this.
    """

    def __init__(self, md: MatchData) -> None:
        self.position = _mid_boss_position(md)
        spawns: list[int] = []
        kills: list[tuple[int, int | None]] = []
        uses: list[tuple[int, int]] = []
        if not md.mid_boss.is_empty():
            for row in md.mid_boss.sort("tick").iter_rows(named=True):
                tick, event = row.get("tick"), str(row.get("event") or "").lower()
                if tick is None:
                    continue
                tick = int(tick)
                team = row.get("team_num")
                if event == "spawned":
                    spawns.append(tick)
                elif event == "killed":
                    kills.append((tick, None))
                elif event == "used" and team in md.team_nums:
                    uses.append((tick, int(team)))

        window = max(1, int(120.0 * md.tick_rate))
        resolved: list[tuple[int, int | None]] = []
        for tick, _ in kills:
            taker = next((t for at, t in uses if tick <= at <= tick + window), None)
            resolved.append((tick, taker))
        self._kills = resolved
        self._intervals: list[tuple[int, int]] = []
        kill_ticks = [t for t, _ in resolved]
        self.available_from = mid_boss_available_from(md)
        earliest = int(self.available_from * md.tick_rate)
        for spawn in spawns:
            at = bisect_left(kill_ticks, spawn)
            end = kill_ticks[at] if at < len(kill_ticks) else md.end_tick
            start = max(spawn, earliest)
            if end > start:
                self._intervals.append((start, end))

    def available(self, tick: int) -> bool:
        return any(start <= tick < end for start, end in self._intervals)

    def taken_by(self, team: int, start: int, end: int) -> bool:
        return any(start <= tick <= end and taker == team for tick, taker in self._kills)


def _mid_boss_position(md: MatchData) -> tuple[float, float] | None:
    """The Mid Boss pit, from the objectives frame.

    The ``mid_boss`` event frame carries no coordinates, but the objectives
    frame lists the boss as an entity with a position, so travel distance is
    knowable after all.
    """
    if md.objectives.is_empty():
        return None
    for row in md.objectives.iter_rows(named=True):
        if str(row.get("objective_type") or "").lower() != "mid_boss":
            continue
        x, y = row.get("x"), row.get("y")
        if x is not None and y is not None:
            return (float(x), float(y))
    return None


class _UrnState:
    """Where the Urn was sitting unclaimed, from the announced events."""

    def __init__(self, md: MatchData) -> None:
        self._events: list[tuple[int, str, float | None, float | None]] = []
        if md.urn.is_empty():
            return
        for row in md.urn.sort("tick").iter_rows(named=True):
            tick, event = row.get("tick"), str(row.get("event") or "").lower()
            if tick is None or event not in ("dropped", "returned", "picked_up"):
                continue
            x, y = row.get("x"), row.get("y")
            self._events.append(
                (int(tick), event, None if x is None else float(x), None if y is None else float(y))
            )
        self._events.sort(key=lambda e: e[0])

    def available_at(self, tick: int) -> tuple[float, float] | None:
        latest = None
        for at, event, x, y in self._events:
            if at > tick:
                break
            latest = (event, x, y)
        if latest is None or latest[0] == "picked_up":
            return None
        return None if latest[1] is None or latest[2] is None else (latest[1], latest[2])


def _advantage_windows(
    md: MatchData,
    team: int,
    states: dict[int, dict[int, _PlayerState]],
    ticks: list[int],
    phase_list,
    structures: list[_Structure],
    deaths: dict[int, list[tuple[int, int]]],
    objective_damage: _CounterSeries,
    damage: _DamageIndex,
    boss: _MidBossState,
    urn: _UrnState,
) -> list[MacroOpportunity]:
    """Stretches where ``team`` was up two or more bodies and spent it on nothing."""
    assert md.clock is not None
    opponents = [h for h in md.hero_ids if md.team_of(h) not in (None, team)]
    mates = [h for h in md.hero_ids if md.team_of(h) == team]
    if not opponents or not mates:
        return []

    counts = [sum(1 for h in opponents if _is_dead_at(deaths, h, t)) for t in ticks]
    grace = max(1, int(MACRO_GRACE_SECONDS * md.tick_rate))
    combat_window = max(1, int(3.0 * md.tick_rate))
    out: list[MacroOpportunity] = []

    index = 0
    while index < len(ticks):
        if counts[index] < MACRO_MIN_ADVANTAGE:
            index += 1
            continue
        stop = index
        while stop + 1 < len(ticks) and counts[stop + 1] >= MACRO_MIN_ADVANTAGE:
            stop += 1
        start, end = ticks[index], ticks[stop]
        peak = max(counts[index : stop + 1])
        index = stop + 1

        seconds = md.clock.seconds(end) - md.clock.seconds(start)
        if seconds < MACRO_MIN_WINDOW_SECONDS:
            continue

        team_damage = objective_damage.delta(mates, start, end + grace)
        structure_fell = any(
            s.death_tick is not None
            and s.team_num != team
            and start <= s.death_tick <= end + grace
            for s in structures
        )
        # The advantage was spent. Nothing to coach here.
        if (
            team_damage >= MACRO_OBJECTIVE_DAMAGE_USED
            or structure_fell
            or boss.taken_by(team, start, end + grace)
        ):
            continue

        living = [
            states[start][h]
            for h in mates
            if h in states.get(start, {}) and states[start][h].is_alive
        ]
        if not living:
            continue
        centre_x = sum(s.x for s in living) / len(living)
        centre_y = sum(s.y for s in living) / len(living)
        target = _nearest_standing(structures, team, start, centre_x, centre_y)

        phase = phase_at(phase_list, start)
        urn_at = urn.available_at(start)
        # Whether the Mid Boss beats a siege depends on team gold, cooldowns and
        # whether you win the fight it starts — none of which we model. A fixed
        # ranking would therefore be an opinion dressed as analysis, and in a
        # long match the Mid Boss is nominally up for twenty minutes, so putting
        # it first silently deletes every other call. Rank by the one thing we
        # actually measured instead — how far the team had to walk — and name
        # the rest as alternatives that were on at the same moment.
        candidates: list[tuple] = []
        if boss.available(start) and phase != "laning":
            boss_x, boss_y = boss.position or (None, None)
            candidates.append(
                ("mid_boss", "take the Mid Boss", "mid_boss", None, None, boss_x, boss_y)
            )
        if urn_at is not None:
            candidates.append(("urn", "run the Urn", "urn", None, None, urn_at[0], urn_at[1]))
        if target is not None:
            candidates.append(
                (
                    "siege", "push the objective", target.objective_type,
                    target.lane, target.team_num, target.x, target.y,
                )
            )
        if not candidates:
            continue

        def _reach(candidate: tuple, cx: float = centre_x, cy: float = centre_y) -> float:
            # An option we cannot locate cannot win on distance: sort it last
            # rather than treating "unknown" as "nearby".
            if candidate[5] is None or candidate[6] is None:
                return float("inf")
            return _distance(cx, cy, candidate[5], candidate[6])

        candidates.sort(key=_reach)
        kind, action, target_kind, target_lane, target_team, target_x, target_y = candidates[0]
        alternatives = tuple(c[2] for c in candidates[1:])

        for hero in mates:
            state = states.get(start, {}).get(hero)
            last = states.get(end, {}).get(hero)
            if state is None or not state.is_alive or last is None or not last.is_alive:
                continue
            own_pct = _health_pct(state.health, state.max_health)
            if own_pct is None or own_pct < MACRO_FREE_HEALTH:
                continue
            if damage.recently_in_combat(hero, start, combat_window):
                continue
            mine = objective_damage.delta([hero], start, end + grace)
            if mine >= MACRO_OBJECTIVE_DAMAGE_USED:
                continue
            distance = (
                None if target_x is None or target_y is None
                else _distance(state.x, state.y, target_x, target_y)
            )

            score = 2 * peak
            score += 2 if seconds >= MACRO_STRONG_WINDOW_SECONDS else 1
            score += 1 if own_pct >= 0.80 else 0
            score += 2 if kind == "mid_boss" else 0
            score += 1 if distance is not None and distance <= 6_000 else 0
            confidence = (
                "high"
                if peak >= 3 and seconds >= MACRO_STRONG_WINDOW_SECONDS
                else "medium"
                if seconds >= MACRO_STRONG_WINDOW_SECONDS or peak >= 3
                else "low"
            )
            evidence = [
                f"{peak} opponents were dead at once for {seconds:.0f}s (kill feed)",
                f"your team dealt {team_damage} objective damage in that window and the "
                f"{MACRO_GRACE_SECONDS:.0f}s after it",
                f"you were alive at {own_pct:.0%} health and out of combat when it opened",
            ]
            if distance is not None:
                evidence.append(f"you were {distance:.0f} units from that objective")
            limitations: list[str] = []
            if kind == "mid_boss":
                gate = getattr(index, "available_from", MID_BOSS_AVAILABLE_SECONDS)
                limitations.append(
                    "the demo does not record when the Mid Boss became attackable; it is "
                    + (
                        "assumed up from the start of the match, per the 2026-05-22 update"
                        if gate <= 0
                        else f"assumed available from {gate / 60:.0f}:00 onward"
                    )
                )
                if distance is None:
                    limitations.append(
                        "the Mid Boss has no position in this demo, so travel distance is unknown"
                    )
            out.append(
                MacroOpportunity(
                    observer_hero_id=hero,
                    kind=kind,
                    action=action,
                    start_tick=start,
                    end_tick=end,
                    phase=phase,
                    confidence=confidence,
                    score=score,
                    target_kind=target_kind,
                    target_lane=target_lane,
                    target_team_num=target_team,
                    alternatives=alternatives,
                    enemies_known_dead=peak,
                    window_seconds=seconds,
                    team_objective_damage=team_damage,
                    observer_objective_damage=mine,
                    distance=distance,
                    evidence=tuple(evidence),
                    limitations=tuple(limitations),
                )
            )
    return out


def _nearest_standing(
    structures: list[_Structure], team: int, tick: int, x: float, y: float
) -> _Structure | None:
    """Closest enemy structure still up. The Patron is a last resort: it is
    always standing, so preferring it would drown out the lane targets."""
    candidates = [
        s
        for s in structures
        if s.team_num != team and (s.death_tick is None or s.death_tick > tick)
    ]
    lane_targets = [s for s in candidates if s.objective_type not in ("patron", "shrine")]
    pool = lane_targets or candidates
    return min(pool, key=lambda s: _distance(x, y, s.x, s.y)) if pool else None


def _defence_windows(
    md: MatchData,
    states: dict[int, dict[int, _PlayerState]],
    ticks: list[int],
    phase_list,
    structures: list[_Structure],
    damage: _DamageIndex,
) -> list[MacroOpportunity]:
    """Your own structure bleeding out while you were somewhere else.

    Troopers chip at structures all game, so only a deep, fast drop counts as
    heroes actually sieging — anything slower is the lane doing its job.
    """
    assert md.clock is not None
    if md.objectives.is_empty():
        return []
    by_entity: dict[int, list[tuple[int, int, int | None]]] = {}
    finished: set[int] = set()
    for row in md.objectives.sort("tick").iter_rows(named=True):
        entity, tick, health = row.get("entity_id"), row.get("tick"), row.get("health")
        if entity is None or tick is None or health is None or int(entity) in finished:
            continue
        entity, health = int(entity), int(health)
        # Carry max_health per row: a Walker's ceiling grows through the match,
        # so measuring a late-game loss against its opening ceiling inflates
        # every percentage and fires the detector on ordinary chip damage.
        by_entity.setdefault(entity, []).append(
            (int(tick), health, _int_or_none(row.get("max_health")))
        )
        # Structures report zero health repeatedly after they fall. Keeping the
        # first one preserves the kill; keeping the rest would re-detect the
        # same loss on every duplicate row.
        if health <= 0:
            finished.add(entity)

    lookup = {s.entity_id: s for s in structures}
    span = max(1, int(DEFEND_WINDOW_SECONDS * md.tick_rate))
    combat_window = max(1, int(3.0 * md.tick_rate))
    out: list[MacroOpportunity] = []

    for entity, history in by_entity.items():
        structure = lookup.get(entity)
        if structure is None:
            continue
        left = 0
        for right in range(len(history)):
            while history[right][0] - history[left][0] > span:
                left += 1
            ceiling = history[right][2] or structure.max_health
            if not ceiling:
                continue
            lost = history[left][1] - history[right][1]
            if lost < ceiling * DEFEND_LOSS_FRACTION:
                continue
            start, end = history[left][0], history[right][0]
            if md.clock.seconds(end) - md.clock.seconds(start) < DEFEND_MIN_SECONDS:
                continue
            defenders = [h for h in md.hero_ids if md.team_of(h) == structure.team_num]
            for hero in defenders:
                state = states.get(_nearest_tick(ticks, start), {}).get(hero)
                if state is None or not state.is_alive:
                    continue
                own_pct = _health_pct(state.health, state.max_health)
                if own_pct is None or own_pct < MACRO_FREE_HEALTH:
                    continue
                distance = _distance(state.x, state.y, structure.x, structure.y)
                if distance < DEFEND_AWAY_RANGE:
                    continue
                if damage.recently_in_combat(hero, start, combat_window):
                    continue
                share = lost / ceiling
                score = 4 + (2 if share >= 0.5 else 0)
                score += 2 if structure.death_tick is not None and structure.death_tick <= end + span else 0
                out.append(
                    MacroOpportunity(
                        observer_hero_id=hero,
                        kind="defend",
                        action="rotate back to defend",
                        start_tick=start,
                        end_tick=end,
                        phase=phase_at(phase_list, start),
                        confidence="medium" if share >= 0.4 else "low",
                        score=score,
                        target_kind=structure.objective_type,
                        target_lane=structure.lane,
                        target_team_num=structure.team_num,
                        alternatives=(),
                        enemies_known_dead=0,
                        window_seconds=md.clock.seconds(end) - md.clock.seconds(start),
                        team_objective_damage=0,
                        observer_objective_damage=0,
                        distance=distance,
                        evidence=(
                            f"your structure lost {share:.0%} of its health in "
                            f"{md.clock.seconds(end) - md.clock.seconds(start):.0f}s",
                            f"you were {distance:.0f} units away, alive at {own_pct:.0%} health "
                            "and out of combat",
                        ),
                        limitations=(),
                    )
                )
            left = right + 1
    return out


def _nearest_tick(ticks: list[int], tick: int) -> int:
    """The sample at or before ``tick`` — objective rows land between samples."""
    at = bisect_right(ticks, tick)
    return ticks[at - 1] if at else (ticks[0] if ticks else tick)


def _limit_by_player(items: list, limit: int | None) -> list:
    by_player: dict[int, list] = {}
    for item in items:
        by_player.setdefault(item.observer_hero_id, []).append(item)
    out: list = []
    for hero, mine in sorted(by_player.items()):
        mine.sort(
            key=lambda item: (
                -_confidence_rank(item.confidence),
                -getattr(item, "score", 0),
                item.start_tick,
            )
        )
        out.extend(mine if limit is None else mine[:limit])
    return out


def _nearest_lane(md: MatchData, x: float, y: float) -> int | None:
    candidates: dict[int, tuple[float, float]] = {}
    if md.objectives.is_empty():
        return None
    for row in md.objectives.iter_rows(named=True):
        lane = row.get("lane")
        ox, oy = row.get("x"), row.get("y")
        if lane in (None, 0, LANE_SENTINEL) or ox is None or oy is None:
            continue
        candidates.setdefault(int(lane), (float(ox), float(oy)))
    if not candidates:
        return None
    return min(candidates, key=lambda lane: _distance(x, y, *candidates[lane]))


def _confidence_rank(value: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}.get(value, 0)


def _health_pct(health: int | None, maximum: int | None) -> float | None:
    if health is None or maximum is None or maximum <= 0:
        return None
    return max(0.0, min(1.0, health / maximum))


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return hypot(x1 - x2, y1 - y2)


def _int_or_none(value) -> int | None:
    return None if value is None else int(value)
