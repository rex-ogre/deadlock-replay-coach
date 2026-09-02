"""Where a player sits against their own rank, and against the best.

A replay can say what happened. It cannot say whether what happened was
normal, and "was that bad?" is the question every solo player actually has.
The Deadlock API publishes, per hero and per rank badge, the summed post-match
counters of every match in that bucket -- in exactly the field names Valve's
own record uses for a single player. So the same arithmetic applied to one
player's final line and to a population row produces two numbers that mean the
same thing, and the difference between them is a real measurement rather than
an impression.

Two rows are quoted for every metric: the player's own rank, which answers "is
this normal", and the highest-populated rank for that hero, which answers "what
do the best players do differently". The top row is deliberately the highest
bucket that clears :data:`~deadlock_coach.gamedata.MIN_BASELINE_MATCHES`
rather than a fixed rank -- an off-meta hero may have no Eternus sample at all,
and inventing one is worse than naming Ascendant honestly.

Three cautions are built into the output rather than left to the reader:

* Everything is a **ratio**. Per-match totals are not comparable across ranks
  because high-rank matches end sooner; souls, deaths and damage all scale with
  a clock the population table does not publish. Every metric here divides two
  counters from the same match, so match length cancels.
* Nothing is comparable **across heroes**. A shotgun and a bow do not share an
  accuracy scale. Every comparison is against the same hero's own rows.
* Some metrics have **no better direction**. Damage per soul separates a
  brawler from a farmer, not a good player from a bad one, and it is labelled
  as style rather than scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from .gamedata import GameConstants, HeroBaseline
from .skillstats import PlayerSkillStats

# Inside this relative band the player and the population are the same number:
# the buckets aggregate thousands of matches, but one player's single match is
# a sample of one, and pretending 3% is a finding would be false precision.
LEVEL_BAND = 0.08


class StatLine(Protocol):
    """The counters a player line and a population row both carry."""

    shots_hit: int
    shots_missed: int
    kills: int
    deaths: int
    assists: int
    net_worth: int
    last_hits: int
    denies: int
    player_damage: int
    player_damage_taken: int
    boss_damage: int
    creep_damage: int
    neutral_damage: int


def _ratio(numerator: float, denominator: float, scale: float = 1.0) -> float | None:
    return numerator / denominator * scale if denominator else None


@dataclass(frozen=True)
class MetricSpec:
    """One comparable number, and how to read it."""

    key: str
    label: str
    unit: str  # "%", "x", or a per-something phrase
    value: Callable[[StatLine], float | None]
    #: True when more is better, False when less is, None when the metric
    #: describes a style rather than a skill and must not be scored.
    higher_is_better: bool | None
    reads_as: str
    #: Smallest gap, in the metric's own unit, worth putting in front of a
    #: player. Relative gaps alone would rank a 0%-versus-2% deny rate as the
    #: single biggest hole in someone's game, which is arithmetic, not coaching.
    min_gap: float = 0.0


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec(
        key="accuracy",
        label="Weapon accuracy",
        unit="%",
        value=lambda s: _ratio(s.shots_hit, s.shots_hit + s.shots_missed),
        higher_is_better=True,
        reads_as="share of shots that landed on anything, creeps included",
        min_gap=0.02,
    ),
    MetricSpec(
        key="deaths_per_10k_souls",
        label="Deaths per 10k souls",
        unit="deaths",
        value=lambda s: _ratio(s.deaths, s.net_worth, 10_000),
        higher_is_better=False,
        reads_as="how often you died relative to how far the match got, so it "
        "does not punish a long game",
        min_gap=0.3,
    ),
    MetricSpec(
        key="kda",
        label="Kills + assists per death",
        unit="x",
        value=lambda s: _ratio(s.kills + s.assists, s.deaths),
        higher_is_better=True,
        reads_as="fight participation against fight cost",
        min_gap=0.3,
    ),
    MetricSpec(
        key="hero_damage_per_1k_souls",
        label="Hero damage per 1k souls",
        unit="damage",
        value=lambda s: _ratio(s.player_damage, s.net_worth, 1_000),
        higher_is_better=None,
        reads_as="how much of your build you turned into damage to heroes; a "
        "farm-first hero sits low here on purpose",
    ),
    MetricSpec(
        key="damage_taken_ratio",
        label="Damage taken per damage dealt",
        unit="x",
        value=lambda s: _ratio(s.player_damage_taken, s.player_damage),
        higher_is_better=False,
        reads_as="what every point of your damage cost you; a tank sits high "
        "here on purpose, but a large gap against your own hero's top rank is "
        "usually position, not role",
        min_gap=0.15,
    ),
    MetricSpec(
        key="deny_rate",
        label="Deny rate",
        unit="%",
        value=lambda s: _ratio(s.denies, s.last_hits + s.denies),
        higher_is_better=True,
        reads_as="share of contested souls you took off the opponent rather "
        "than only securing your own",
        min_gap=0.03,
    ),
    MetricSpec(
        key="objective_damage_per_1k_souls",
        label="Objective damage per 1k souls",
        unit="damage",
        value=lambda s: _ratio(s.boss_damage, s.net_worth, 1_000),
        higher_is_better=None,
        reads_as="how much of your build went into Walkers, Shrines, the Patron "
        "and the Mid Boss rather than into heroes and creeps",
    ),
    MetricSpec(
        key="jungle_share",
        label="Jungle share of farm damage",
        unit="%",
        value=lambda s: _ratio(s.neutral_damage, s.creep_damage + s.neutral_damage),
        higher_is_better=None,
        reads_as="how much of your farming happened off-lane; a style number, "
        "but a large gap explains where the souls came from",
    ),
)


@dataclass(frozen=True)
class MetricComparison:
    """One metric, for one player, against two population rows."""

    key: str
    label: str
    unit: str
    reads_as: str
    higher_is_better: bool | None
    player: float | None
    peer: float | None
    top: float | None
    peer_badge: int | None
    peer_matches: int
    top_badge: int | None
    top_matches: int
    min_gap: float = 0.0

    @property
    def gap_to_top(self) -> float | None:
        """Player minus the top rank, in the metric's own unit."""
        if self.player is None or self.top is None:
            return None
        return self.player - self.top

    @property
    def gap_to_peer(self) -> float | None:
        if self.player is None or self.peer is None:
            return None
        return self.player - self.peer

    @property
    def relative_to_top(self) -> float | None:
        """Gap as a share of the top rank's value, for ranking findings."""
        if self.gap_to_top is None or not self.top:
            return None
        return self.gap_to_top / self.top

    @property
    def verdict(self) -> str:
        """``behind`` / ``ahead`` / ``level`` / ``style`` against the top rank."""
        if self.higher_is_better is None:
            return "style"
        relative = self.relative_to_top
        if relative is None:
            return "unknown"
        if abs(relative) < LEVEL_BAND:
            return "level"
        better = relative > 0 if self.higher_is_better else relative < 0
        return "ahead" if better else "behind"

    @property
    def severity(self) -> float:
        """How far behind the top rank this is; 0 unless it is a deficit.

        Used only for ranking, so that a summary leads with the metric that is
        furthest from the top rather than the one that happens to be first. A
        deficit smaller than :attr:`min_gap` scores zero however large it looks
        in relative terms.
        """
        relative = self.relative_to_top
        gap = self.gap_to_top
        if relative is None or gap is None or self.verdict != "behind":
            return 0.0
        if abs(gap) < self.min_gap:
            return 0.0
        return abs(relative)

    def format(self, value: float | None) -> str:
        if value is None:
            return "—"
        if self.unit == "%":
            return f"{value:.0%}"
        if self.unit == "x":
            return f"{value:.2f}x"
        # A deaths-per-10k figure lives between 1 and 4; rounding it to a whole
        # number would erase the entire difference between ranks.
        return f"{value:,.1f}" if abs(value) < 20 else f"{value:,.0f}"


@dataclass(frozen=True)
class PlayerBenchmark:
    """Every comparison available for one player, plus what they were against."""

    hero_id: int
    badge: int | None
    metrics: tuple[MetricComparison, ...]
    peer: HeroBaseline | None
    top: HeroBaseline | None

    @property
    def available(self) -> bool:
        return any(m.player is not None and m.top is not None for m in self.metrics)

    def by_key(self, key: str) -> MetricComparison | None:
        return next((m for m in self.metrics if m.key == key), None)

    def deficits(self, limit: int | None = None) -> list[MetricComparison]:
        """Metrics where this player is measurably behind the top rank.

        Ordered worst first, and filtered to gaps large enough to be worth a
        player's practice time -- see :attr:`MetricSpec.min_gap`.
        """
        out = sorted(
            (m for m in self.metrics if m.verdict == "behind" and m.severity),
            key=lambda m: -m.severity,
        )
        return out[:limit] if limit else out

    def strengths(self, limit: int | None = None) -> list[MetricComparison]:
        out = sorted(
            (m for m in self.metrics if m.verdict == "ahead"),
            key=lambda m: -abs(m.relative_to_top or 0),
        )
        return out[:limit] if limit else out


def benchmark_player(
    stat: PlayerSkillStats,
    constants: GameConstants,
) -> PlayerBenchmark:
    """Compare one player's post-match line with their rank and with the top.

    Missing population rows are left as ``None`` rather than filled in: an
    unpopular hero at a rare rank genuinely has no baseline, and a comparison
    against a 40-match bucket would be noise dressed as coaching.
    """
    peer = constants.peer_baseline(stat.hero_id, stat.badge)
    top = constants.top_baseline(stat.hero_id)
    # An all-ranks row is not a peer row. It is kept when nothing better exists,
    # but it must never be presented as "the top", which would compare a player
    # against an average that includes themselves.
    if top is not None and peer is not None and top.badge == peer.badge:
        top = None

    metrics = tuple(
        MetricComparison(
            key=spec.key,
            label=spec.label,
            unit=spec.unit,
            reads_as=spec.reads_as,
            higher_is_better=spec.higher_is_better,
            player=spec.value(stat),
            peer=spec.value(peer) if peer else None,
            top=spec.value(top) if top else None,
            peer_badge=peer.badge if peer else None,
            peer_matches=peer.matches if peer else 0,
            top_badge=top.badge if top else None,
            top_matches=top.matches if top else 0,
            min_gap=spec.min_gap,
        )
        for spec in METRICS
    )
    return PlayerBenchmark(
        hero_id=stat.hero_id,
        badge=stat.badge,
        metrics=metrics,
        peer=peer,
        top=top,
    )
