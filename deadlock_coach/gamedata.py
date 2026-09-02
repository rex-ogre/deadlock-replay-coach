"""Live game constants, pulled from the Deadlock assets API.

Everything in :mod:`physics` needs numbers that Valve changes on a roughly
weekly cadence, and several of them are *per hero*: the 2026-07-28 patch split
the roster into three dash-duration buckets (0.62 / 0.68 / 0.72s over the same
10m), so a single hardcoded dash speed is wrong for two thirds of the cast.

The numbers therefore come from ``api.deadlock-api.com``, which rebuilds itself
from the Steam depot on every patch. Because a coaching run must not fail
because a laptop is offline or the API is down, lookups degrade in three steps:

1. a cached fetch on disk, if it is younger than :data:`CACHE_TTL_SECONDS`
2. a live fetch, which refreshes that cache
3. the snapshot bundled in ``data/game_constants.json``

Whichever wins is recorded on :attr:`GameConstants.source` and surfaced in the
report, because a reader needs to know whether the physics they are being shown
matches the patch the replay was recorded on.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ASSETS_BASE = "https://api.deadlock-api.com/v1/assets"
ANALYTICS_BASE = "https://api.deadlock-api.com/v1/analytics"

BUNDLED_SNAPSHOT = Path(__file__).parent / "data" / "game_constants.json"

# A patch lands every week or two; a day-old copy is close enough for coaching
# and keeps a batch of replays from hammering the API.
CACHE_TTL_SECONDS = 24 * 60 * 60
HTTP_TIMEOUT_SECONDS = 20.0

# Source 2 measures the world in hammer units; Deadlock's own stat sheet is in
# metres. One hu is one inch. Verified against replay tracks: differencing
# player positions at 1Hz puts the median hero at 6.15 m/s (base move speed is
# 6.7), the 95th percentile at 20.3 (outer zip line is 20.6) and the 99.9th at
# 38.6 (outer zip line under the +80% boost is 37.1).
HU_PER_METER = 39.37007874015748


def cache_dir() -> Path:
    override = os.environ.get("DEADLOCK_COACH_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "deadlock-coach"


@dataclass(frozen=True)
class HeroConstants:
    """The movement and weapon numbers that vary between heroes."""

    hero_id: int
    name: str
    max_move_speed: float          # m/s, walking
    sprint_speed: float            # m/s, *added* to max_move_speed out of combat
    crouch_speed: float            # m/s
    move_acceleration: float       # m/s^2
    stamina: float                 # bars
    stamina_regen_per_second: float
    ground_dash_distance_m: float
    ground_dash_duration: float
    air_dash_distance_m: float
    air_dash_duration: float
    # Weapon ballistics. Kept here because "was that even a shot he could hit"
    # is a range question before it is an aim question.
    bullet_speed_hu: float | None = None
    damage_falloff_start_m: float | None = None
    damage_falloff_end_m: float | None = None
    crit_bonus_start_m: float | None = None
    crit_bonus_end_m: float | None = None
    shots_per_second: float | None = None

    @property
    def sprint_total_speed(self) -> float:
        """Top sustained ground speed, once the out-of-combat timer has run."""
        return self.max_move_speed + self.sprint_speed

    @property
    def ground_dash_speed(self) -> float:
        if self.ground_dash_duration <= 0:
            return self.max_move_speed
        return self.ground_dash_distance_m / self.ground_dash_duration

    @property
    def air_dash_speed(self) -> float:
        if self.air_dash_duration <= 0:
            return self.max_move_speed
        return self.air_dash_distance_m / self.air_dash_duration


@dataclass(frozen=True)
class ZiplineConstants:
    """Transit-line numbers, in the units the engine stores them in.

    Speeds are hammer units per second; ``dismount_*_speed_percent`` are the
    share of line speed you keep when you drop off with crouch, which is what
    the community calls zip line momentum conservation.
    """

    speed_inner_hu: float = 693.0
    speed_outer_hu: float = 810.0
    latch_speed_hu: float = 1500.0
    latch_initial_speed_hu: float = 600.0
    latch_end_speed_hu: float = 750.0
    latch_max_time: float = 0.5
    max_mount_distance_m: float = 15.0
    dismount_horizontal_min_percent: float = 40.0
    dismount_horizontal_max_percent: float = 85.0
    dismount_vertical_speed_hu: float = 300.0
    boost_multiplier: float = 1.8
    stun_duration: float = 2.5
    slow_duration_on_hit: float = 2.0

    @property
    def speed_inner(self) -> float:
        return self.speed_inner_hu / HU_PER_METER

    @property
    def speed_outer(self) -> float:
        return self.speed_outer_hu / HU_PER_METER

    @property
    def latch_speed(self) -> float:
        return self.latch_speed_hu / HU_PER_METER

    @property
    def dismount_carry_speed(self) -> float:
        """Best-case speed carried off the outer line, in m/s.

        This is the number behind every "he crossed the map in four seconds"
        clip: 85% of 810 hu/s is 688 hu/s, or 2.6x a hero's walking speed.
        """
        return self.speed_outer * self.dismount_horizontal_max_percent / 100.0


#: Rank tier names, indexed by ``badge // 10``.  Fetched with the rest of the
#: snapshot; this list is the fallback when the assets API is unreachable and
#: exists so a report can say "Ascendant 1" instead of "badge 101".
RANK_NAMES: dict[int, str] = {
    0: "Obscurus",
    1: "Initiate",
    2: "Seeker",
    3: "Acolyte",
    4: "Sentinel",
    5: "Mystic",
    6: "Ritualist",
    7: "Emissary",
    8: "Oracle",
    9: "Phantom",
    10: "Ascendant",
    11: "Eternus",
}

# Below this a population row is noise, and quoting it would invent precision.
MIN_BASELINE_MATCHES = 200

#: Order the compact snapshot rows are written in. Keeping the counters as a
#: list rather than 14 repeated keys per row is what keeps the bundled snapshot
#: to a few hundred KB across ~2,300 (hero, rank) rows.
BASELINE_FIELDS = (
    "matches",
    "wins",
    "shots_hit",
    "shots_missed",
    "kills",
    "deaths",
    "assists",
    "net_worth",
    "last_hits",
    "denies",
    "player_damage",
    "player_damage_taken",
    "boss_damage",
    "creep_damage",
    "neutral_damage",
)


@dataclass(frozen=True)
class HeroBaseline:
    """What one hero's population did, in one rank bucket.

    Every field is a *total* over ``matches`` matches, in exactly the field
    names Valve's own post-match record uses for a single player. That is the
    point: the same arithmetic turns a player's final stat line and this row
    into the same ratios, so "you versus the population" is never comparing two
    differently-defined numbers.
    """

    hero_id: int
    badge: int  # tier * 10 + subrank; 0 is the all-ranks aggregate
    matches: int
    wins: int = 0
    shots_hit: int = 0
    shots_missed: int = 0
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    net_worth: int = 0
    last_hits: int = 0
    denies: int = 0
    player_damage: int = 0
    player_damage_taken: int = 0
    boss_damage: int = 0
    creep_damage: int = 0
    neutral_damage: int = 0

    @property
    def accuracy(self) -> float | None:
        shots = self.shots_hit + self.shots_missed
        return self.shots_hit / shots if shots else None

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.matches if self.matches else None

    @property
    def label(self) -> str:
        return badge_name(self.badge)


def badge_name(badge: int | None) -> str:
    """``101`` -> ``"Ascendant 1"``. Unknown tiers keep their number."""
    if badge is None:
        return "unknown rank"
    badge = int(badge)
    if badge <= 0:
        return "all ranks"
    tier, sub = divmod(badge, 10)
    name = RANK_NAMES.get(tier, f"tier {tier}")
    return f"{name} {sub}" if sub else name


@dataclass(frozen=True)
class GameConstants:
    """A patch-stamped bundle of everything :mod:`physics` needs."""

    heroes: dict[int, HeroConstants]
    zipline: ZiplineConstants
    accuracy_baseline: dict[int, dict[int, tuple[float, int]]] = field(default_factory=dict)
    hero_baseline: dict[int, dict[int, HeroBaseline]] = field(default_factory=dict)
    ranks: dict[int, str] = field(default_factory=dict)
    fetched_at: str = ""
    source: str = "bundled"

    def hero(self, hero_id: int | None) -> HeroConstants | None:
        if hero_id is None:
            return None
        return self.heroes.get(int(hero_id))

    def baseline_accuracy(self, hero_id: int, badge: int | None) -> tuple[float, int] | None:
        """Population accuracy for this hero at this rank badge.

        Falls back to the neighbouring badge and then to the hero's all-rank
        aggregate, because a rare hero at a rare rank can have too few matches
        to be worth quoting.
        """
        table = self.accuracy_baseline.get(int(hero_id))
        if not table:
            return None
        if badge is not None:
            for candidate in (int(badge), int(badge) - 1, int(badge) + 1):
                hit = table.get(candidate)
                if hit and hit[1] >= MIN_BASELINE_MATCHES:
                    return hit
        return table.get(0)

    def peer_baseline(self, hero_id: int, badge: int | None) -> HeroBaseline | None:
        """This hero's population row at the player's own rank.

        Same widening as :meth:`baseline_accuracy` -- own badge, then either
        neighbour, then the all-ranks aggregate -- because "compared with
        players like you" is worth more than an exact bucket with 40 matches
        in it.
        """
        table = self.hero_baseline.get(int(hero_id))
        if not table:
            return None
        if badge is not None:
            for candidate in (int(badge), int(badge) - 1, int(badge) + 1):
                row = table.get(int(candidate))
                if row and row.matches >= MIN_BASELINE_MATCHES:
                    return row
        return table.get(0)

    def top_baseline(self, hero_id: int) -> HeroBaseline | None:
        """The highest rank bucket for this hero that has enough matches.

        This is the "what do the best players do" row. It is deliberately the
        highest *populated* bucket rather than a fixed rank: an off-meta hero
        may have no Eternus sample at all, and inventing one would be worse
        than comparing against Ascendant.
        """
        table = self.hero_baseline.get(int(hero_id))
        if not table:
            return None
        rows = [
            row
            for badge, row in table.items()
            if badge > 0 and row.matches >= MIN_BASELINE_MATCHES
        ]
        return max(rows, key=lambda row: row.badge) if rows else None

    def rank_name(self, badge: int | None) -> str:
        if badge is None:
            return "unknown rank"
        tier, sub = divmod(int(badge), 10)
        name = self.ranks.get(tier) or RANK_NAMES.get(tier)
        if not name:
            return badge_name(badge)
        return f"{name} {sub}" if sub else name

    @property
    def is_stale(self) -> bool:
        """True when these numbers did not come from a recent live fetch.

        Worth saying out loud in a report: between 2026-03 and 2026-08 Valve
        shipped movement, stamina, Mid Boss and Urn changes in separate patches.
        """
        return self.source == "bundled"


# ------------------------------------------------------------------ fetching


def _get_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "deadlock-replay-coach"})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _stat(stats: dict, key: str, default: float = 0.0) -> float:
    entry = stats.get(key)
    if isinstance(entry, dict):
        entry = entry.get("value")
    try:
        return float(entry)
    except (TypeError, ValueError):
        return default


def _weapon_of(hero: dict, weapons: dict[str, dict]) -> dict:
    """Resolve a hero's primary weapon, whose stats live in a separate asset."""
    class_name = (hero.get("items") or {}).get("weapon_primary")
    if not class_name:
        return {}
    return (weapons.get(class_name) or {}).get("weapon_info") or {}


def _hero_from_asset(hero: dict, weapons: dict[str, dict]) -> HeroConstants | None:
    stats = hero.get("starting_stats") or {}
    if not stats:
        return None
    weapon = _weapon_of(hero, weapons)

    def weapon_m(key: str) -> float | None:
        raw = weapon.get(key)
        if raw is None:
            return None
        try:
            return float(raw) / HU_PER_METER
        except (TypeError, ValueError):
            return None

    return HeroConstants(
        hero_id=int(hero["id"]),
        name=str(hero.get("name") or hero.get("class_name") or hero["id"]),
        max_move_speed=_stat(stats, "max_move_speed", 6.7),
        sprint_speed=_stat(stats, "sprint_speed", 1.6),
        crouch_speed=_stat(stats, "crouch_speed", 4.75),
        move_acceleration=_stat(stats, "move_acceleration", 4.0),
        stamina=_stat(stats, "stamina", 3.0),
        stamina_regen_per_second=_stat(stats, "stamina_regen_per_second", 0.2222),
        ground_dash_distance_m=_stat(stats, "ground_dash_distance_in_meters", 10.0),
        ground_dash_duration=_stat(stats, "ground_dash_duration", 0.68),
        air_dash_distance_m=_stat(stats, "air_dash_distance_in_meters", 8.0),
        air_dash_duration=_stat(stats, "air_dash_duration", 0.47),
        bullet_speed_hu=(float(weapon["bullet_speed"]) if weapon.get("bullet_speed") else None),
        damage_falloff_start_m=weapon_m("damage_falloff_start_range"),
        damage_falloff_end_m=weapon_m("damage_falloff_end_range"),
        crit_bonus_start_m=weapon_m("crit_bonus_start_range"),
        crit_bonus_end_m=weapon_m("crit_bonus_end_range"),
        shots_per_second=(
            float(weapon["shots_per_second"]) if weapon.get("shots_per_second") else None
        ),
    )


def _zipline_from_asset(ability: dict) -> ZiplineConstants:
    props = ability.get("properties") or {}

    def prop(key: str, default: float) -> float:
        entry = props.get(key)
        if isinstance(entry, dict):
            entry = entry.get("value")
        if entry is None:
            return default
        try:
            # Some ranges arrive as "15m" rather than a bare number.
            return float(str(entry).rstrip("ms").strip() or default)
        except (TypeError, ValueError):
            return default

    return ZiplineConstants(
        speed_inner_hu=prop("ZipSpeedInner", 693.0),
        speed_outer_hu=prop("ZipSpeedOuter", 810.0),
        latch_speed_hu=prop("LatchSpeed", 1500.0),
        latch_initial_speed_hu=prop("LatchInitialSpeed", 600.0),
        latch_end_speed_hu=prop("LatchEndSpeed", 750.0),
        latch_max_time=prop("LatchMaxTime", 0.5),
        max_mount_distance_m=prop("MaxMountDistance2D", 15.0),
        dismount_horizontal_min_percent=prop("DismountHorizontalMinSpeedPercent", 40.0),
        dismount_horizontal_max_percent=prop("DismountHorizontalMaxSpeedPercent", 85.0),
        dismount_vertical_speed_hu=prop("DismountVerticalSpeed", 300.0),
        stun_duration=prop("StunDuration", 2.5),
        slow_duration_on_hit=prop("ZiplineProtectionSlowDurationOnHit", 2.0),
    )


def _fetch_hero_baseline() -> dict[int, dict[int, HeroBaseline]]:
    """Population totals per hero per rank badge.

    One request carries every counter Valve's post-match record keeps for a
    single player, summed over every match in that (hero, rank) bucket. Divided
    through, it answers the only question a solo player cannot answer from
    their own replay: is this number normal for my hero at my rank, and what
    does the same number look like at the top.

    Nothing here is comparable across heroes -- a shotgun and a bow do not
    share an accuracy scale, and a farming hero and a brawler do not share a
    damage-per-soul scale -- so the table is always keyed by (hero, badge) and
    a player is only ever compared against their own hero's rows.
    """
    rows = _get_json(f"{ANALYTICS_BASE}/hero-stats?bucket=avg_badge")
    if not isinstance(rows, list):
        return {}
    table: dict[int, dict[int, HeroBaseline]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            hero_id = int(row["hero_id"])
            matches = int(row.get("matches") or 0)
        except (KeyError, TypeError, ValueError):
            continue
        if matches <= 0:
            continue
        counters = {
            field_name: int(row.get(f"total_{field_name}") or 0)
            for field_name in BASELINE_FIELDS
            if field_name not in ("matches", "wins")
        }
        badge = int(row.get("bucket") or 0)
        table.setdefault(hero_id, {})[badge] = HeroBaseline(
            hero_id=hero_id,
            badge=badge,
            matches=matches,
            wins=int(row.get("wins") or 0),
            **counters,
        )
    return table


def _fetch_rank_names() -> dict[int, str]:
    rows = _get_json(f"{ASSETS_BASE}/ranks")
    if not isinstance(rows, list):
        return {}
    names: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            names[int(row["tier"])] = str(row["name"])
        except (KeyError, TypeError, ValueError):
            continue
    return names


def _baseline_rows(table: dict[int, dict[int, HeroBaseline]]) -> dict[str, dict]:
    """Serialise the baseline compactly: one list of counters per rank bucket.

    Buckets below :data:`MIN_BASELINE_MATCHES` are dropped on the way out. They
    can never be quoted, and keeping them would roughly double a snapshot that
    ships inside the wheel.
    """
    out: dict[str, dict] = {}
    for hero_id, buckets in table.items():
        rows = {
            str(badge): [getattr(row, name) for name in BASELINE_FIELDS]
            for badge, row in buckets.items()
            if badge == 0 or row.matches >= MIN_BASELINE_MATCHES
        }
        if rows:
            out[str(hero_id)] = rows
    return out


def _baseline_from_rows(payload: dict) -> dict[int, dict[int, HeroBaseline]]:
    table: dict[int, dict[int, HeroBaseline]] = {}
    for raw_id, buckets in (payload.get("hero_baseline") or {}).items():
        for raw_badge, values in (buckets or {}).items():
            if not isinstance(values, list) or len(values) != len(BASELINE_FIELDS):
                continue
            try:
                counters = {
                    name: int(value) for name, value in zip(BASELINE_FIELDS, values)
                }
                hero_id, badge = int(raw_id), int(raw_badge)
            except (TypeError, ValueError):
                continue
            table.setdefault(hero_id, {})[badge] = HeroBaseline(
                hero_id=hero_id, badge=badge, **counters
            )
    return table


def fetch_snapshot(*, with_baseline: bool = True) -> dict:
    """Build a fresh, serialisable snapshot straight from the assets API."""
    heroes = _get_json(f"{ASSETS_BASE}/heroes?only_active=true")
    weapon_assets = _get_json(f"{ASSETS_BASE}/items/by-type/weapon")
    weapons = {w.get("class_name"): w for w in weapon_assets if isinstance(w, dict)}
    zipline = _get_json(f"{ASSETS_BASE}/items/citadel_ability_zip_line")

    hero_rows: dict[str, dict] = {}
    for hero in heroes:
        parsed = _hero_from_asset(hero, weapons)
        if parsed is None:
            continue
        row = parsed.__dict__.copy()
        row.pop("hero_id")
        hero_rows[str(parsed.hero_id)] = row

    baseline: dict[str, dict] = {}
    ranks: dict[int, str] = {}
    if with_baseline:
        try:
            baseline = _baseline_rows(_fetch_hero_baseline())
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            # The baseline is a nicety; the movement constants are not. A dead
            # analytics endpoint costs the comparison columns, nothing else.
            log.warning("hero baseline unavailable (%s: %s)", type(exc).__name__, exc)
        try:
            ranks = _fetch_rank_names()
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            log.info("rank names unavailable (%s: %s)", type(exc).__name__, exc)

    return {
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_url": ASSETS_BASE,
        "hu_per_meter": HU_PER_METER,
        "zipline": _zipline_from_asset(zipline).__dict__,
        "heroes": hero_rows,
        "hero_baseline": baseline,
        "ranks": {str(tier): name for tier, name in ranks.items()},
    }


def _from_snapshot(payload: dict, source: str) -> GameConstants:
    heroes: dict[int, HeroConstants] = {}
    for raw_id, row in (payload.get("heroes") or {}).items():
        try:
            heroes[int(raw_id)] = HeroConstants(hero_id=int(raw_id), **row)
        except (TypeError, ValueError) as exc:
            log.warning("skipping hero %s in snapshot (%s)", raw_id, exc)

    zip_row = payload.get("zipline") or {}
    known = ZiplineConstants.__dataclass_fields__
    zipline = ZiplineConstants(**{k: v for k, v in zip_row.items() if k in known})

    # Two snapshot generations are readable here. The current one carries every
    # population counter and derives accuracy from it; an older cache on disk
    # carries only the pre-computed accuracy pair, which still feeds the
    # mechanics table even though the wider comparison goes unavailable.
    hero_baseline = _baseline_from_rows(payload)
    baseline: dict[int, dict[int, tuple[float, int]]] = {
        hero_id: {
            badge: (round(row.accuracy, 4), row.matches)
            for badge, row in buckets.items()
            if row.accuracy is not None
        }
        for hero_id, buckets in hero_baseline.items()
    }
    for raw_id, buckets in (payload.get("accuracy_baseline") or {}).items():
        rows = baseline.setdefault(int(raw_id), {})
        for badge, value in buckets.items():
            if len(value) == 2:
                rows.setdefault(int(badge), (float(value[0]), int(value[1])))

    ranks: dict[int, str] = {}
    for raw_tier, name in (payload.get("ranks") or {}).items():
        try:
            ranks[int(raw_tier)] = str(name)
        except (TypeError, ValueError):
            continue

    return GameConstants(
        heroes=heroes,
        zipline=zipline,
        accuracy_baseline=baseline,
        hero_baseline=hero_baseline,
        ranks=ranks or dict(RANK_NAMES),
        fetched_at=str(payload.get("fetched_at") or ""),
        source=source,
    )


# ------------------------------------------------------------------- loading

_MEMO: dict[str, GameConstants] = {}


def _cache_path() -> Path:
    return cache_dir() / "game_constants.json"


def _read_cache(ttl: float) -> dict | None:
    path = _cache_path()
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return None
    if ttl >= 0 and age > ttl:
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("ignoring unreadable constants cache (%s: %s)", type(exc).__name__, exc)
        return None


def _write_cache(payload: dict) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    except OSError as exc:
        log.warning("could not cache constants (%s: %s)", type(exc).__name__, exc)


def load_constants(*, refresh: bool = False, offline: bool = False) -> GameConstants:
    """Return the current game constants, cheapest acceptable source first.

    Args:
        refresh: ignore the disk cache and force a live fetch.
        offline: never touch the network. Used by tests and by the ``--offline``
            CLI flag so a run is reproducible.
    """
    memo_key = f"refresh={refresh}&offline={offline}"
    cached = _MEMO.get(memo_key)
    if cached is not None:
        return cached

    result: GameConstants | None = None

    if not refresh and not offline:
        payload = _read_cache(CACHE_TTL_SECONDS)
        if payload:
            result = _from_snapshot(payload, "cache")

    if result is None and not offline:
        try:
            payload = fetch_snapshot()
            _write_cache(payload)
            result = _from_snapshot(payload, "live")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
            log.warning("assets API unavailable, falling back (%s: %s)", type(exc).__name__, exc)

    if result is None:
        # A stale cache still beats a snapshot that shipped with the wheel.
        payload = _read_cache(ttl=-1)
        if payload:
            result = _from_snapshot(payload, "cache")

    if result is None:
        try:
            result = _from_snapshot(json.loads(BUNDLED_SNAPSHOT.read_text()), "bundled")
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("bundled snapshot unusable (%s: %s)", type(exc).__name__, exc)
            result = GameConstants(heroes={}, zipline=ZiplineConstants(), source="defaults")

    _MEMO[memo_key] = result
    return result


def reset_cache() -> None:
    """Drop the in-process memo. Tests use this; nothing else needs it."""
    _MEMO.clear()
