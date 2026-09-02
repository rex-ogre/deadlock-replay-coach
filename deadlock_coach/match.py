"""The in-memory match model everything downstream operates on.

Deliberately decoupled from ``boon``: the analysis and rendering layers take a
:class:`MatchData` of plain Polars frames, so they can be tested against
hand-built fixtures without a 200MB replay on disk. ``source.py`` is the only
module that knows boon exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import polars as pl

# Column contracts for the frames we consume. Kept explicit so that (a) empty
# frames still have the right dtypes for filters/joins and (b) a boon schema
# change surfaces here as one failing test rather than as ten confusing ones.
SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "players": {
        "player_name": pl.String,
        "steam_id": pl.Int64,
        "hero_id": pl.Int64,
        "team_num": pl.Int64,
        "start_lane": pl.Int64,
        "rank": pl.Int64,
    },
    "kills": {
        "tick": pl.Int64,
        "victim_hero_id": pl.Int64,
        "attacker_hero_id": pl.Int64,
        "assister_hero_ids": pl.List(pl.Int64),
    },
    "damage": {
        "tick": pl.Int64,
        "damage": pl.Int64,
        "pre_damage": pl.Float64,
        "victim_hero_id": pl.Int64,
        "attacker_hero_id": pl.Int64,
        "victim_health_new": pl.Int64,
        "hitgroup_id": pl.Int64,
        "crit_damage": pl.Float64,
        "attacker_class": pl.Int64,
        "victim_class": pl.Int64,
    },
    "objectives": {
        "tick": pl.Int64,
        "objective_type": pl.String,
        "team_num": pl.Int64,
        "lane": pl.Int64,
        "health": pl.Int64,
        "max_health": pl.Int64,
        "phase": pl.Int64,
        "x": pl.Float64,
        "y": pl.Float64,
        "z": pl.Float64,
        "entity_id": pl.Int64,
    },
    "mid_boss": {"tick": pl.Int64, "team_num": pl.Int64, "event": pl.String},
    "urn": {
        "tick": pl.Int64,
        "event": pl.String,
        "hero_id": pl.Int64,
        "team_num": pl.Int64,
        "x": pl.Float64,
        "y": pl.Float64,
        "z": pl.Float64,
    },
    "rift": {
        "rift_num": pl.Int64,
        "announce_tick": pl.Int64,
        "active_tick": pl.Int64,
        "capture_tick": pl.Int64,
        "expire_tick": pl.Int64,
        "winning_team": pl.Int64,
        "lane": pl.Int64,
        "x": pl.Float64,
        "y": pl.Float64,
        "z": pl.Float64,
    },
    "teamfights": {
        "fight_id": pl.Int64,
        "start_tick": pl.Int64,
        "end_tick": pl.Int64,
        "start_seconds": pl.Float64,
        "end_seconds": pl.Float64,
        "duration_seconds": pl.Float64,
        "center_x": pl.Float64,
        "center_y": pl.Float64,
        "participants": pl.List(pl.Int64),
        "num_participants": pl.Int64,
        "hero_damage": pl.Int64,
        "kills": pl.Int64,
    },
    # Sampled only around teamfight endings. The raw trooper dataset is several
    # million rows per match; conversion analysis needs the wave at the decision
    # point, not every creep position at every tick.
    "trooper_samples": {
        "tick": pl.Int64,
        "trooper_type": pl.String,
        "team_num": pl.Int64,
        "lane": pl.Int64,
        "health": pl.Int64,
        "max_health": pl.Int64,
        "x": pl.Float64,
        "y": pl.Float64,
        "z": pl.Float64,
        "entity_id": pl.Int64,
    },
    # A *sampled* slice of player_ticks — never the full per-tick frame, which
    # is millions of rows and useless to a language model.
    "player_samples": {
        "tick": pl.Int64,
        "hero_id": pl.Int64,
        "x": pl.Float64,
        "y": pl.Float64,
        "z": pl.Float64,
        "pitch": pl.Float64,
        "yaw": pl.Float64,
        "roll": pl.Float64,
        "is_alive": pl.Boolean,
        "health": pl.Int64,
        "max_health": pl.Int64,
        "in_combat_end_time": pl.Float64,
        "in_combat_last_damage_time": pl.Float64,
        "time_revealed_by_npc": pl.Float64,
        "has_ultimate_trained": pl.Boolean,
        "ultimate_cooldown_start": pl.Float64,
        "ultimate_cooldown_end": pl.Float64,
        "souls": pl.Int64,
        "spent_souls": pl.Int64,
        "level": pl.Int64,
        "kills": pl.Int64,
        "deaths": pl.Int64,
        "assists": pl.Int64,
        "last_hits": pl.Int64,
        "denies": pl.Int64,
        "hero_damage": pl.Int64,
        "objective_damage": pl.Int64,
    },
    "item_purchases": {
        "tick": pl.Int64,
        "hero_id": pl.Int64,
        "ability_id": pl.Int64,
        "change": pl.String,
    },
    "ability_upgrades": {
        "tick": pl.Int64,
        "hero_id": pl.Int64,
        "ability_id": pl.Int64,
        "tier": pl.Int64,
    },
    # Change-only cooldown/charge state from boon.  Unlike player samples this
    # stays compact even for a long match and lets the browser reconstruct the
    # exact skill bar at any replay tick.
    "ability_ticks": {
        "tick": pl.Int64,
        "hero_id": pl.Int64,
        "ability_id": pl.Int64,
        "slot": pl.Int64,
        "cooldown_start": pl.Float64,
        "cooldown_end": pl.Float64,
        "remaining_charges": pl.Int64,
        "charge_recharge_start": pl.Float64,
        "charge_recharge_end": pl.Float64,
    },
    "ability_uses": {
        "tick": pl.Int64,
        "hero_id": pl.Int64,
        # boon currently emits the resolved class name here.  Keeping this as
        # a string also gives newer game builds a graceful fallback.
        "ability": pl.String,
    },
    "neutrals": {
        "tick": pl.Int64,
        "team_num": pl.Int64,
        "health": pl.Int64,
        "max_health": pl.Int64,
        "x": pl.Float64,
        "y": pl.Float64,
        "z": pl.Float64,
        "entity_id": pl.Int64,
    },
}


def empty(name: str) -> pl.DataFrame:
    """An empty frame with the declared schema for ``name``."""
    return pl.DataFrame(schema=SCHEMAS[name])


def conform(name: str, df: pl.DataFrame | None) -> pl.DataFrame:
    """Add any missing declared columns as nulls, so downstream code can index
    them unconditionally. Extra columns are preserved."""
    if df is None:
        return empty(name)
    missing = [
        pl.lit(None, dtype=dtype).alias(col)
        for col, dtype in SCHEMAS[name].items()
        if col not in df.columns
    ]
    return df.with_columns(missing) if missing else df


class Clock:
    """Tick <-> wall clock, excluding paused time.

    The default is a plain linear conversion; ``source.py`` swaps in boon's
    pause-aware version, which is what you want on a real demo where a 90
    second pause would otherwise shift every timestamp after it.
    """

    def __init__(self, tick_rate: int, seconds_fn: Callable[[int], float] | None = None):
        if tick_rate <= 0:
            raise ValueError(f"tick_rate must be positive, got {tick_rate}")
        self.tick_rate = tick_rate
        self._seconds_fn = seconds_fn

    def seconds(self, tick: int | None) -> float:
        if tick is None:
            return 0.0
        if self._seconds_fn is not None:
            return float(self._seconds_fn(int(tick)))
        return int(tick) / self.tick_rate

    def mmss(self, tick: int | None) -> str:
        if tick is None:
            return "--:--"
        total = max(0, int(round(self.seconds(tick))))
        return f"{total // 60:02d}:{total % 60:02d}"


@dataclass
class MatchData:
    """Everything the analysis layer needs, and nothing it doesn't."""

    map_name: str = "unknown"
    match_id: int | None = None
    build: int | None = None
    game_mode: int | None = None
    tick_rate: int = 60
    total_ticks: int = 0
    game_over_tick: int | None = None
    winning_team_num: int | None = None

    players: pl.DataFrame = field(default_factory=lambda: empty("players"))
    kills: pl.DataFrame = field(default_factory=lambda: empty("kills"))
    damage: pl.DataFrame = field(default_factory=lambda: empty("damage"))
    objectives: pl.DataFrame = field(default_factory=lambda: empty("objectives"))
    mid_boss: pl.DataFrame = field(default_factory=lambda: empty("mid_boss"))
    urn: pl.DataFrame = field(default_factory=lambda: empty("urn"))
    rift: pl.DataFrame = field(default_factory=lambda: empty("rift"))
    teamfights: pl.DataFrame = field(default_factory=lambda: empty("teamfights"))
    trooper_samples: pl.DataFrame = field(default_factory=lambda: empty("trooper_samples"))
    player_samples: pl.DataFrame = field(default_factory=lambda: empty("player_samples"))
    item_purchases: pl.DataFrame = field(default_factory=lambda: empty("item_purchases"))
    ability_upgrades: pl.DataFrame = field(default_factory=lambda: empty("ability_upgrades"))
    ability_ticks: pl.DataFrame = field(default_factory=lambda: empty("ability_ticks"))
    ability_uses: pl.DataFrame = field(default_factory=lambda: empty("ability_uses"))
    neutrals: pl.DataFrame = field(default_factory=lambda: empty("neutrals"))

    clock: Clock | None = None

    def __post_init__(self) -> None:
        if self.clock is None:
            self.clock = Clock(self.tick_rate)
        for name in SCHEMAS:
            setattr(self, name, conform(name, getattr(self, name)))

    # -- roster helpers -------------------------------------------------

    @property
    def hero_ids(self) -> list[int]:
        if self.players.is_empty():
            return []
        return sorted(self.players["hero_id"].drop_nulls().to_list())

    @property
    def team_nums(self) -> list[int]:
        """Playing teams only. Team 1 is spectators and never fields a hero."""
        if self.players.is_empty():
            return []
        return sorted({t for t in self.players["team_num"].drop_nulls().to_list() if t != 1})

    def team_of(self, hero_id: int | None) -> int | None:
        if hero_id is None or self.players.is_empty():
            return None
        row = self.players.filter(pl.col("hero_id") == hero_id)
        return None if row.is_empty() else int(row["team_num"][0])

    def player_name(self, hero_id: int | None) -> str | None:
        if hero_id is None or self.players.is_empty():
            return None
        row = self.players.filter(pl.col("hero_id") == hero_id)
        return None if row.is_empty() else str(row["player_name"][0])

    def teammates_of(self, hero_id: int) -> list[int]:
        team = self.team_of(hero_id)
        if team is None:
            return []
        return [h for h in self.hero_ids if h != hero_id and self.team_of(h) == team]

    def with_frames(self, **frames: pl.DataFrame) -> MatchData:
        return replace(self, **frames)

    @property
    def end_tick(self) -> int:
        """Last tick worth reporting on: game over if known, else end of file."""
        return int(self.game_over_tick or self.total_ticks or 0)
