"""Movement maths: how fast a hero can cross a gap, and how fast one did.

Deadlock is not a game where "the ally was 60 metres away" means anything on
its own. Sixty metres is seven seconds of walking, four of dashing, or three
off a zip line, and which of those it was decides whether a death was a
positioning error or a teammate who was never going to arrive.

Two directions are modelled here, and they answer different questions:

*Forwards* — :func:`travel_seconds` turns a distance into a time under stated
assumptions. Every assumption is generous (straight line, no walls, no verticality,
full stamina, perfect routing), which makes the result a **lower bound on travel
time**. That asymmetry is the point: when the lower bound already exceeds the
time available, "he could not have got there" is a fact rather than a guess.
The converse is never claimed.

*Backwards* — :func:`classify_speed` reads an observed speed back to the
mechanic that must have produced it. A hero sustaining 20 m/s was on a zip
line; there is no other way to move that fast. This is how a replay reveals
whether a player actually uses the map's movement, without the demo ever
recording a "used zip line" event.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from .gamedata import HU_PER_METER, HeroConstants, ZiplineConstants

# Sprint engages only after the hero has been out of combat for five seconds,
# so a rotation that starts under fire spends its opening at walking speed.
SPRINT_ENGAGE_SECONDS = 5.0

# Deciding to move, turning, and getting hands off whatever you were doing.
# Kept small and explicit rather than folded into the speed.
DEFAULT_REACTION_SECONDS = 0.75

# Crouching below this is a walk, not a slide. Engine value: 350 hu/s.
SLIDE_THRESHOLD = 350.0 / HU_PER_METER

# Observed speeds above this are teleports, respawns, or a parser hiccup, not
# travel. The fastest legitimate sustained movement is an outer zip line under
# the +80% boost, at ~37 m/s.
IMPLAUSIBLE_SPEED = 45.0


def meters(hu: float) -> float:
    """Hammer units to metres."""
    return hu / HU_PER_METER


def hammer_units(m: float) -> float:
    """Metres to hammer units."""
    return m * HU_PER_METER


def distance_m(ax: float, ay: float, bx: float, by: float) -> float:
    """Horizontal separation of two replay positions, in metres.

    Deliberately 2D. Deadlock's map is stacked, but z in the sample frames
    swings with jumps and rooftops, and including it makes two heroes on the
    same walkway look further apart than they are.
    """
    return meters(hypot(ax - bx, ay - by))


@dataclass(frozen=True)
class TravelEstimate:
    """A lower bound on how long a hero needs to cover a distance."""

    seconds: float
    distance_m: float
    mode: str
    dashes_used: int
    assumption: str

    def beats(self, budget_seconds: float) -> bool:
        return self.seconds <= budget_seconds


def travel_seconds(
    distance: float,
    hero: HeroConstants,
    *,
    zipline: ZiplineConstants | None = None,
    stamina: float | None = None,
    sprinting: bool = True,
    zipline_fraction: float = 0.0,
    reaction: float = DEFAULT_REACTION_SECONDS,
) -> TravelEstimate:
    """Fastest plausible crossing of ``distance`` metres.

    Args:
        distance: straight-line separation in metres.
        hero: the mover's per-hero constants, which differ — the 2026-07-28
            patch gave the roster three different dash durations.
        zipline: transit-line constants, required if ``zipline_fraction`` > 0.
        stamina: bars available for dashing. Defaults to the hero's full bar.
        sprinting: whether the out-of-combat sprint bonus applies. A hero who
            just took damage does not get it.
        zipline_fraction: share of the route covered on a transit line. Left at
            zero unless something in the replay says a line was actually on the
            path, because assuming one is how you manufacture blame.
        reaction: decision and turn time before the hero starts moving.
    """
    if distance <= 0:
        return TravelEstimate(0.0, 0.0, "none", 0, "already there")

    remaining = distance
    elapsed = reaction
    mode = "run"

    if zipline_fraction > 0 and zipline is not None:
        leg = distance * min(1.0, zipline_fraction)
        elapsed += leg / zipline.speed_outer
        remaining -= leg
        mode = "zip line"

    bars = hero.stamina if stamina is None else stamina
    dashes = 0
    dash_speed = hero.ground_dash_speed
    cruise = hero.sprint_total_speed if sprinting else hero.max_move_speed

    # Dashing only helps while it is faster than running, which it always is on
    # the current roster, but the comparison keeps the model honest if a patch
    # changes that.
    if dash_speed > cruise:
        while bars >= 1 and remaining > 0:
            leg = min(remaining, hero.ground_dash_distance_m)
            elapsed += leg / dash_speed
            remaining -= leg
            bars -= 1
            dashes += 1
        if dashes and mode == "run":
            mode = "dash"

    if remaining > 0:
        elapsed += remaining / cruise
        # One acceleration ramp from standing. Ignoring it flatters short
        # rotations, where it is a meaningful share of the total.
        elapsed += cruise / (2 * hero.move_acceleration) if hero.move_acceleration > 0 else 0.0

    parts = [f"straight line, {'sprint' if sprinting else 'walk'}"]
    if dashes:
        parts.append(f"{dashes} dash{'es' if dashes > 1 else ''}")
    if zipline_fraction > 0 and zipline is not None:
        parts.append(f"{zipline_fraction:.0%} on a zip line")
    parts.append("no walls or verticality")

    return TravelEstimate(
        seconds=elapsed,
        distance_m=distance,
        mode=mode,
        dashes_used=dashes,
        assumption="; ".join(parts),
    )


# ------------------------------------------------- reading speed back to tech

# Bands are named for the mechanic that is the *cheapest* explanation of a
# sustained speed, with the boundaries sitting between mechanics rather than on
# them so that sampling jitter does not flip a label.
@dataclass(frozen=True)
class SpeedBand:
    name: str
    floor: float
    note: str


def speed_bands(hero: HeroConstants, zipline: ZiplineConstants) -> list[SpeedBand]:
    """Speed thresholds for one hero, fastest first."""
    walk = hero.max_move_speed
    sprint = hero.sprint_total_speed
    return [
        SpeedBand("zip line + boost", zipline.speed_inner * 1.4, "transit line under the speed boost"),
        SpeedBand("zip line", zipline.speed_inner * 0.85, "riding a transit line"),
        SpeedBand("dash or momentum", sprint * 1.25, "dashing, or carrying speed off a line"),
        SpeedBand("sprint", walk * 1.05, "out-of-combat sprint"),
        SpeedBand("walk", walk * 0.45, "walking, or fighting"),
        SpeedBand("stationary", 0.0, "holding position, shopping, or dead"),
    ]


def classify_speed(speed: float, hero: HeroConstants, zipline: ZiplineConstants) -> str:
    """Name the mechanic that best explains an observed speed, in m/s."""
    if speed > IMPLAUSIBLE_SPEED:
        return "teleport"
    for band in speed_bands(hero, zipline):
        if speed >= band.floor:
            return band.name
    return "stationary"


def reference_speeds(hero: HeroConstants, zipline: ZiplineConstants) -> dict[str, float]:
    """The speed ladder for one hero, in m/s, for the report's physics note."""
    return {
        "walk": hero.max_move_speed,
        "sprint": hero.sprint_total_speed,
        "ground dash": hero.ground_dash_speed,
        "air dash": hero.air_dash_speed,
        "zip line (inner)": zipline.speed_inner,
        "zip line (outer)": zipline.speed_outer,
        "zip line dismount carry": zipline.dismount_carry_speed,
        "zip line + boost (outer)": zipline.speed_outer * zipline.boost_multiplier,
    }


def could_have_arrived(
    distance: float,
    budget_seconds: float,
    hero: HeroConstants,
    *,
    zipline: ZiplineConstants | None = None,
    sprinting: bool = True,
) -> tuple[bool, TravelEstimate]:
    """Was ``distance`` metres crossable inside ``budget_seconds``?

    Answers with the most generous assumptions available, so a ``False`` is
    load-bearing and a ``True`` only means "not ruled out by physics".
    """
    estimate = travel_seconds(distance, hero, zipline=zipline, sprinting=sprinting)
    return estimate.beats(budget_seconds), estimate


def falloff_note(hero: HeroConstants, range_m: float) -> str | None:
    """Where a shot at this range sits on the hero's damage falloff curve.

    A low accuracy number at long range is a positioning read, not an aim read,
    and the two get different advice.
    """
    start, end = hero.damage_falloff_start_m, hero.damage_falloff_end_m
    if start is None or end is None or end <= start:
        return None
    if range_m <= start:
        return "inside full-damage range"
    if range_m >= end:
        return f"past {end:.0f}m, minimum damage"
    share = (range_m - start) / (end - start)
    return f"{share:.0%} into falloff (full damage ends at {start:.0f}m)"
