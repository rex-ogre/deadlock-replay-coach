"""Human-readable names for the numeric IDs Deadlock demos are full of.

Everything an LLM reads should say "Infernus", never "hero_id 1". The lookups
come from ``boon`` when it is importable, but this module never *requires* it —
tests and offline use inject their own tables.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# From the CMsgLaneColor proto enum, as documented by boon's `players.start_lane`.
LANE_COLORS: dict[int, str] = {
    0: "none",
    1: "yellow",
    3: "green",
    4: "blue",
    6: "purple",
}

# Lane id used by objectives that do not belong to a lane (Patron, Shrines) and
# by neutral objectives, which report an all-ones sentinel.
LANE_NONE = 0
LANE_SENTINEL = 16777215

# hero_id 0 is not a hero. It is what the demo reports when the killer was a
# trooper, an objective, or the environment — the name table calls it "Base",
# which is a trap: crediting it would invent a 13th player.
NO_HERO = 0

# `objectives.objective_type` is a raw string from the demo. Verified against
# build 10854 (match 12345678), which has walker / barracks / shrine / patron /
# mid_boss and no Guardian at all. Anything unknown passes through untouched
# rather than being dropped, so a game update can't silently blank the report.
OBJECTIVE_LABELS: dict[str, str] = {
    "walker": "Walker",
    "barracks": "Barracks",
    "shrine": "Shrine",
    "patron": "Patron",
    "mid_boss": "Mid Boss",
    # Pre-build-10854 names, kept so older demos still render.
    "guardian": "Guardian",
    "base_guardian": "Base Guardian",
    "titan": "Patron",
    "core": "Patron",
}

PATRON_PHASES: dict[int, str] = {0: "normal", 1: "final", 2: "transforming"}


def _boon_table(fn_name: str) -> dict[int, str]:
    try:
        import boon
    except Exception:  # pragma: no cover - exercised only without boon installed
        return {}
    fn = getattr(boon, fn_name, None)
    if fn is None:  # pragma: no cover - defensive against boon API drift
        return {}
    try:
        return dict(fn())
    except Exception:  # pragma: no cover - defensive
        return {}


@dataclass
class Names:
    """Id -> name resolution with graceful fallbacks.

    Unknown ids render as ``Hero#42`` rather than raising, because a demo
    recorded on a newer patch will contain heroes and items this table has
    never heard of, and a report that renders is worth more than one that
    crashes.
    """

    heroes: dict[int, str] = field(default_factory=dict)
    teams: dict[int, str] = field(default_factory=dict)
    abilities: dict[int, str] = field(default_factory=dict)

    @classmethod
    def from_boon(cls) -> Names:
        return cls(
            heroes=_boon_table("hero_names"),
            teams=_boon_table("team_names"),
            abilities=_boon_table("ability_names"),
        )

    def hero(self, hero_id: int | None) -> str:
        if hero_id is None:
            return "unknown"
        return self.heroes.get(hero_id) or f"Hero#{hero_id}"

    def team(self, team_num: int | None) -> str:
        if team_num is None:
            return "unknown"
        return self.teams.get(team_num) or f"Team#{team_num}"

    def ability(self, ability_id: int | None) -> str:
        if ability_id is None:
            return "unknown"
        return self.abilities.get(ability_id) or f"Item#{ability_id}"

    def lane(self, lane_color: int | None) -> str:
        if lane_color is None or lane_color == LANE_SENTINEL:
            return "unknown"
        return LANE_COLORS.get(lane_color) or f"lane#{lane_color}"

    def objective(self, objective_type: str | None) -> str:
        if not objective_type:
            return "Objective"
        key = objective_type.strip().lower()
        return OBJECTIVE_LABELS.get(key, objective_type)
