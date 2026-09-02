"""Turn boon's per-dataset frames into one clock-stamped narrative event stream.

This is the layer that makes a replay legible to a language model. Raw frames
answer "what was every objective's health on tick 41300"; this answers "at
21:47 Archmother destroyed the yellow Walker", which is the form a model can
actually reason about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import polars as pl

from .match import MatchData
from .names import LANE_NONE, LANE_SENTINEL, NO_HERO, PATRON_PHASES, Names

# Ordering for events landing on the same tick. Objectives and kills are the
# spine of the story; economy noise sorts last.
KIND_ORDER: dict[str, int] = {
    "objective": 0,
    "patron_phase": 1,
    "kill": 2,
    "mid_boss": 3,
    "rift": 4,
    "urn": 5,
    "purchase": 6,
}


@dataclass(frozen=True)
class Event:
    tick: int
    kind: str
    text: str
    team: int | None = None
    actors: tuple[int, ...] = ()
    subject: int | None = None
    detail: dict = field(default_factory=dict)

    def sort_key(self) -> tuple[int, int, str]:
        return (self.tick, KIND_ORDER.get(self.kind, 99), self.text)


def _hero_label(md: MatchData, names: Names, hero_id: int | None) -> str:
    """``Infernus (player_one)`` — hero first, because tactics are about heroes."""
    if hero_id == NO_HERO:
        return "a non-hero source"
    hero = names.hero(hero_id)
    player = md.player_name(hero_id)
    return f"{hero} ({player})" if player else hero


def kill_events(md: MatchData, names: Names) -> list[Event]:
    if md.kills.is_empty():
        return []
    out: list[Event] = []
    for row in md.kills.sort("tick").iter_rows(named=True):
        victim = row.get("victim_hero_id")
        attacker = row.get("attacker_hero_id")
        assisters = [a for a in (row.get("assister_hero_ids") or []) if a not in (None, NO_HERO)]
        team = md.team_of(attacker)

        # hero_id 0 means a trooper, an objective or the environment landed the
        # killing blow. The name table renders it as "Base"; crediting it would
        # invent a player who does not exist. Suicides (attacker == victim) get
        # the same treatment — a coach reads both very differently from a kill.
        by_npc = attacker == NO_HERO
        if attacker is None or by_npc or attacker == victim:
            noun = "non-hero damage (troopers or objectives)" if by_npc else "no killer credited"
            text = f"{_hero_label(md, names, victim)} died to {noun}"
            if assisters:
                text += f" (enemy nearby: {', '.join(names.hero(a) for a in assisters)})"
            # Leave the pseudo-attacker out of `actors` so downstream counts of
            # "how many heroes were on this kill" stay honest. Assisters stay:
            # a hero who chipped the victim before troopers finished them did
            # cause the death, and their team should be credited with it.
            attacker = None
            team = md.team_of(assisters[0]) if assisters else None
        else:
            text = f"{_hero_label(md, names, attacker)} killed {_hero_label(md, names, victim)}"
            if assisters:
                helpers = ", ".join(names.hero(a) for a in assisters)
                text += f" (assist: {helpers})"

        actors = tuple(a for a in ([attacker] if attacker is not None else []) + assisters)
        out.append(
            Event(
                tick=int(row["tick"]),
                kind="kill",
                text=text,
                team=team,
                actors=actors,
                subject=victim,
                detail={
                    "attacker_hero_id": attacker,
                    "victim_hero_id": victim,
                    "assister_hero_ids": assisters,
                    "attackers": len(actors),
                },
            )
        )
    return out


def objective_events(md: MatchData, names: Names) -> list[Event]:
    """Destruction events, derived from the health-per-tick frame.

    boon emits a row whenever an objective's health changes; there is no
    "destroyed" message. So destruction is the *first* tick at which a given
    entity's health reaches zero — first, because some objectives report
    further zero-health rows afterwards and we want one event, not five.
    """
    df = md.objectives
    if df.is_empty():
        return []

    # entity_id is the reliable key; fall back to the type/team/lane triple when
    # a demo does not carry it.
    keys = ["entity_id"] if df["entity_id"].null_count() < df.height else [
        "objective_type",
        "team_num",
        "lane",
    ]
    dead = df.filter(pl.col("health").is_not_null() & (pl.col("health") <= 0))
    if dead.is_empty():
        return []
    first = dead.sort("tick").group_by(keys, maintain_order=True).first()

    out: list[Event] = []
    for row in first.sort("tick").iter_rows(named=True):
        label = names.objective(row.get("objective_type"))
        owner = row.get("team_num")

        # Neutral objectives (Mid Boss owns team 4) are not anybody's structure
        # falling; they belong to their own dataset, which says who took them.
        if owner not in md.team_nums:
            continue

        lane = row.get("lane")
        lane_txt = (
            f" ({names.lane(lane)} lane)"
            if lane not in (None, LANE_NONE, LANE_SENTINEL)
            else ""
        )
        # team_num on an objective is its *owner*; the destroyer is the other team.
        destroyer = _other_team(md, owner)
        who = names.team(destroyer) if destroyer is not None else "The enemy team"
        out.append(
            Event(
                tick=int(row["tick"]),
                kind="objective",
                text=f"{who} destroyed {names.team(owner)}'s {label}{lane_txt}",
                team=destroyer,
                detail={
                    "objective_type": row.get("objective_type"),
                    "owner_team": owner,
                    "lane": lane,
                    "label": label,
                },
            )
        )
    return out


def patron_phase_events(md: MatchData, names: Names) -> list[Event]:
    """Patron entering its final phase is the single loudest tactical signal in
    a Deadlock match — surface it explicitly."""
    df = md.objectives
    if df.is_empty() or df["phase"].null_count() == df.height:
        return []
    out: list[Event] = []
    for (entity_id,), group in df.sort("tick").group_by(["entity_id"], maintain_order=True):
        previous = None
        for row in group.iter_rows(named=True):
            phase = row.get("phase")
            if phase is None or phase == previous:
                continue
            if previous is not None and phase != 0:
                owner = row.get("team_num")
                out.append(
                    Event(
                        tick=int(row["tick"]),
                        kind="patron_phase",
                        text=(
                            f"{names.team(owner)}'s {names.objective(row.get('objective_type'))} "
                            f"entered phase '{PATRON_PHASES.get(phase, phase)}'"
                        ),
                        team=owner,
                        detail={"phase": phase, "entity_id": entity_id},
                    )
                )
            previous = phase
    return out


def mid_boss_events(md: MatchData, names: Names) -> list[Event]:
    if md.mid_boss.is_empty():
        return []
    out: list[Event] = []
    for row in md.mid_boss.sort("tick").iter_rows(named=True):
        team = row.get("team_num")
        # `killed` rows carry the boss's own (neutral) team, not the killer's;
        # only name a team when it is one of the two playing.
        who = f" — {names.team(team)}" if team in md.team_nums else ""
        out.append(
            Event(
                tick=int(row["tick"]),
                kind="mid_boss",
                text=f"Mid Boss {row.get('event')}{who}",
                team=team if team in md.team_nums else None,
                detail={"event": row.get("event")},
            )
        )
    return out


def urn_events(md: MatchData, names: Names) -> list[Event]:
    if md.urn.is_empty():
        return []
    out = []
    for row in md.urn.sort("tick").iter_rows(named=True):
        hero = row.get("hero_id")
        carrier = f" — {_hero_label(md, names, hero)}" if hero else ""
        out.append(
            Event(
                tick=int(row["tick"]),
                kind="urn",
                text=f"Urn: {row.get('event')}{carrier}",
                team=row.get("team_num"),
                actors=(hero,) if hero else (),
                detail={"event": row.get("event")},
            )
        )
    return out


def rift_events(md: MatchData, names: Names) -> list[Event]:
    """One row per rift; emit the moments that matter (spawn and resolution)."""
    if md.rift.is_empty():
        return []
    out: list[Event] = []
    for row in md.rift.sort("active_tick").iter_rows(named=True):
        num = row.get("rift_num")
        if row.get("active_tick") is not None:
            out.append(
                Event(
                    tick=int(row["active_tick"]),
                    kind="rift",
                    text=f"Rift #{num} went active in lane {row.get('lane')}",
                    detail={"rift_num": num, "stage": "active"},
                )
            )
        if row.get("capture_tick") is not None:
            winner = row.get("winning_team")
            out.append(
                Event(
                    tick=int(row["capture_tick"]),
                    kind="rift",
                    text=f"Rift #{num} captured by {names.team(winner)}",
                    team=winner,
                    detail={"rift_num": num, "stage": "capture"},
                )
            )
        elif row.get("expire_tick") is not None:
            out.append(
                Event(
                    tick=int(row["expire_tick"]),
                    kind="rift",
                    text=f"Rift #{num} expired uncaptured",
                    detail={"rift_num": num, "stage": "expire"},
                )
            )
    return out


def _other_team(md: MatchData, team: int | None) -> int | None:
    others = [t for t in md.team_nums if t != team]
    return others[0] if len(others) == 1 else None


def build_timeline(md: MatchData, names: Names | None = None) -> list[Event]:
    """The full stream, deterministically ordered."""
    names = names or Names.from_boon()
    events: list[Event] = []
    for builder in (
        objective_events,
        patron_phase_events,
        kill_events,
        mid_boss_events,
        urn_events,
        rift_events,
    ):
        events.extend(builder(md, names))
    return sorted(events, key=Event.sort_key)


def format_timeline(md: MatchData, events: Iterable[Event]) -> list[str]:
    """``[12:34] Infernus (player_one) killed Seven (player_two)`` — one line per event."""
    assert md.clock is not None
    return [f"[{md.clock.mmss(e.tick)}] {e.text}" for e in events]
