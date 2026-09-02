"""Per-player mechanical stats from a replay or the network fallback.

The per-tick entity stream records landed damage but cannot identify every
miss. A completed replay also embeds Valve's cumulative ``PostMatchDetails``,
which *does* contain ``shots_hit`` and ``shots_missed``. The local replay is the
primary source; the Deadlock API remains a fallback for incomplete recordings.

Three numbers from it are worth coaching on, and they mean different things:

``accuracy``
    ``shots_hit / (shots_hit + shots_missed)`` over every target, creeps
    included. Rank-correlated and cleanly measured, but only comparable within
    one hero: a shotgun and a bow do not share a scale.
``hero_bullet_share``
    what fraction of landed shots hit a *hero*. Two players with identical
    accuracy can be farming or fighting, and this separates them.
``creep_efficiency``
    ``creep_kills / possible_creeps``. The API counts the creeps that were
    available to secure, so this is a real denominator rather than a per-minute
    rate that punishes short matches.

Every one of these is optional. A replay that ends before the post-match packet
may still need the API, which is rate limited to three Steam match lookups an
hour per IP without a key. If neither source has the match, the module returns
nothing without taking the report down.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .gamedata import (
    HTTP_TIMEOUT_SECONDS,
    MIN_BASELINE_MATCHES,
    GameConstants,
    cache_dir,
)

log = logging.getLogger(__name__)

API_BASE = "https://api.deadlock-api.com/v1"

# Steam's 64-bit IDs are an offset over the account IDs the match record uses.
STEAM_ID_OFFSET = 76_561_197_960_265_728


def account_id_of(steam_id: int | None) -> int | None:
    if steam_id is None:
        return None
    account = int(steam_id) - STEAM_ID_OFFSET
    return account if account > 0 else int(steam_id)


@dataclass(frozen=True)
class PlayerSkillStats:
    """One player's mechanical line for one match."""

    account_id: int
    hero_id: int
    shots_hit: int
    shots_missed: int
    hero_bullets_hit: int
    hero_bullets_hit_crit: int
    creep_kills: int
    possible_creeps: int
    badge: int | None = None
    baseline_accuracy: float | None = None
    baseline_matches: int = 0
    # The rest of the post-match line. Nothing downstream of the mechanics
    # table needs these individually -- they exist so that :mod:`benchmark` can
    # form the same ratios out of a player and a population row, from counters
    # that carry identical definitions on both sides.
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
    neutral_kills: int = 0
    duration_seconds: int = 0

    @property
    def shots_taken(self) -> int:
        return self.shots_hit + self.shots_missed

    @property
    def accuracy(self) -> float | None:
        return self.shots_hit / self.shots_taken if self.shots_taken else None

    @property
    def crit_rate(self) -> float | None:
        if not self.hero_bullets_hit:
            return None
        return self.hero_bullets_hit_crit / self.hero_bullets_hit

    @property
    def hero_bullet_share(self) -> float | None:
        return self.hero_bullets_hit / self.shots_hit if self.shots_hit else None

    @property
    def creep_efficiency(self) -> float | None:
        return self.creep_kills / self.possible_creeps if self.possible_creeps else None

    @property
    def accuracy_delta(self) -> float | None:
        """Accuracy minus the population's, for this hero at this rank."""
        if self.accuracy is None or self.baseline_accuracy is None:
            return None
        return self.accuracy - self.baseline_accuracy

    def summary(self) -> str | None:
        """One line a language model can reason over without the raw counts."""
        if self.accuracy is None:
            return None
        parts = [f"accuracy {self.accuracy:.0%} on {self.shots_taken} shots"]
        delta = self.accuracy_delta
        if delta is not None:
            direction = "above" if delta >= 0 else "below"
            parts.append(
                f"{abs(delta):.0%} {direction} the {self.baseline_matches:,}-match "
                f"median for this hero at this rank ({self.baseline_accuracy:.0%})"
            )
        if self.hero_bullet_share is not None:
            parts.append(f"{self.hero_bullet_share:.0%} of landed shots were on heroes")
        if self.crit_rate is not None:
            parts.append(f"{self.crit_rate:.0%} of those were crits")
        if self.creep_efficiency is not None:
            parts.append(
                f"secured {self.creep_kills}/{self.possible_creeps} available creeps "
                f"({self.creep_efficiency:.0%})"
            )
        return "; ".join(parts)


# ------------------------------------------------------------------ fetching


def _metadata_path(match_id: int) -> Path:
    return cache_dir() / "matches" / f"{match_id}.json"


def _headers() -> dict[str, str]:
    headers = {"User-Agent": "deadlock-replay-coach"}
    key = os.environ.get("DEADLOCK_API_KEY")
    if key:
        headers["X-API-KEY"] = key
    return headers


def _cache_metadata(path: Path, payload: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    except OSError as exc:
        log.warning("could not cache match metadata (%s)", exc)


def _has_skill_samples(payload: object) -> bool:
    """Whether a metadata-shaped payload contains the required denominator."""
    if not isinstance(payload, dict):
        return False
    info = payload.get("match_info")
    if not isinstance(info, dict):
        return False
    for player in info.get("players") or []:
        if not isinstance(player, dict):
            continue
        samples = player.get("stats") or []
        if not samples or not isinstance(samples[-1], dict):
            continue
        if "shots_hit" in samples[-1] or "shots_missed" in samples[-1]:
            return True
    return False


def _fetch_archived_metadata(match_id: int) -> dict | None:
    """Reconstruct the useful part of metadata from the API's match archive.

    Valve eventually removes a match's original metadata.  Deadlock API keeps
    many parsed matches in its database for longer, including the final
    cumulative counters used here.  Querying that archive prevents an expired
    Steam/S3 record from turning the whole mechanics table into ``unknown``.
    """
    params = urllib.parse.urlencode(
        {
            "match_ids": int(match_id),
            "include_player_info": "true",
            "include_player_final_stats": "true",
            # The endpoint otherwise defaults to ranked + unranked only.
            "match_mode": "",
            "limit": 1,
        }
    )
    url = f"{API_BASE}/matches/metadata?{params}"
    try:
        request = urllib.request.Request(url, headers=_headers())
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            matches = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        log.info("archived match stats unavailable for %s (HTTP %s)", match_id, exc.code)
        return None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        log.info("archived match stats unavailable for %s (%s)", match_id, type(exc).__name__)
        return None

    if not isinstance(matches, list) or not matches or not isinstance(matches[0], dict):
        return None

    match = matches[0]
    players = []
    for player in match.get("players") or []:
        if not isinstance(player, dict) or not isinstance(player.get("final_stats"), dict):
            continue
        try:
            account_id = int(player["account_id"])
            hero_id = int(player["hero_id"])
        except (KeyError, TypeError, ValueError):
            continue
        players.append(
            {
                "account_id": account_id,
                "hero_id": hero_id,
                # last_hits lives on the player row, not in final_stats, and the
                # population table counts the same field -- so it has to survive
                # the reshape or the farming comparison silently reads zero.
                "last_hits": player.get("last_hits"),
                "denies": player.get("denies"),
                "net_worth": player.get("net_worth"),
                "kills": player.get("kills"),
                "deaths": player.get("deaths"),
                "assists": player.get("assists"),
                "stats": [player["final_stats"]],
            }
        )
    if not players:
        return None
    return {
        "match_info": {
            "average_badge_team0": match.get("average_badge_team0"),
            "average_badge_team1": match.get("average_badge_team1"),
            "duration_s": match.get("duration_s"),
            "players": players,
        },
        "stats_source": "deadlock-api-archive",
    }


def _fetch_metadata(match_id: int) -> dict | None:
    """Pull one match's post-game record, cached forever once we have it.

    The cache is permanent on purpose: a finished match never changes, and the
    endpoint is capped at three lookups an hour per IP without an API key.
    """
    path = _metadata_path(match_id)
    if path.exists():
        try:
            cached = json.loads(path.read_text())
            if _has_skill_samples(cached):
                return cached
            log.info("cached metadata for %s has no shot counters; refreshing", match_id)
        except (OSError, json.JSONDecodeError):
            log.warning("discarding corrupt metadata cache for %s", match_id)

    url = f"{API_BASE}/matches/{match_id}/metadata"
    payload = None
    try:
        request = urllib.request.Request(url, headers=_headers())
        with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # 429 is the un-keyed rate limit, 503 is Valve having dropped the match.
        log.info("match metadata unavailable for %s (HTTP %s)", match_id, exc.code)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        log.info("match metadata unavailable for %s (%s)", match_id, type(exc).__name__)

    if _has_skill_samples(payload):
        _cache_metadata(path, payload)
        return payload

    archived = _fetch_archived_metadata(match_id)
    if archived is not None:
        _cache_metadata(path, archived)
    return archived


def _last_sample(player: dict) -> dict:
    """The final cumulative row of the per-minute series.

    The series is cumulative, so the last row is the whole match. Earlier rows
    are kept by the caller when a phase breakdown is wanted.
    """
    samples = player.get("stats") or []
    return samples[-1] if samples else {}


def _counter(player: dict, sample: dict, name: str) -> int:
    """Read a cumulative counter from either Valve metadata location."""
    value = player.get(name)
    if value is None:
        value = sample.get(name)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def stats_from_metadata(
    payload: dict | None,
    constants: GameConstants,
) -> dict[int, PlayerSkillStats]:
    """Normalize API- or replay-shaped metadata, keyed by account ID."""
    if not payload:
        return {}
    info = payload.get("match_info") or {}

    badges = [
        info.get("average_badge_team0") or 0,
        info.get("average_badge_team1") or 0,
    ]
    known = [b for b in badges if b]
    match_badge = round(sum(known) / len(known)) if known else None
    duration = info.get("duration_s") or 0

    out: dict[int, PlayerSkillStats] = {}
    for player in info.get("players") or []:
        sample = _last_sample(player)
        if not sample:
            continue
        try:
            account = int(player["account_id"])
            hero_id = int(player["hero_id"])
        except (KeyError, TypeError, ValueError):
            continue

        baseline = constants.baseline_accuracy(hero_id, match_badge)
        if baseline and baseline[1] < MIN_BASELINE_MATCHES:
            baseline = None

        # Valve reports some counters on the player row and some inside the
        # per-minute series. Reading the row first keeps both record shapes --
        # Steam's own and the API archive -- on the same definitions.
        out[account] = PlayerSkillStats(
            account_id=account,
            hero_id=hero_id,
            shots_hit=int(sample.get("shots_hit") or 0),
            shots_missed=int(sample.get("shots_missed") or 0),
            hero_bullets_hit=int(sample.get("hero_bullets_hit") or 0),
            hero_bullets_hit_crit=int(sample.get("hero_bullets_hit_crit") or 0),
            creep_kills=int(sample.get("creep_kills") or 0),
            possible_creeps=int(sample.get("possible_creeps") or 0),
            badge=match_badge,
            baseline_accuracy=baseline[0] if baseline else None,
            baseline_matches=baseline[1] if baseline else 0,
            kills=_counter(player, sample, "kills"),
            deaths=_counter(player, sample, "deaths"),
            assists=_counter(player, sample, "assists"),
            net_worth=_counter(player, sample, "net_worth"),
            last_hits=_counter(player, sample, "last_hits"),
            denies=_counter(player, sample, "denies"),
            player_damage=_counter(player, sample, "player_damage"),
            player_damage_taken=_counter(player, sample, "player_damage_taken"),
            boss_damage=_counter(player, sample, "boss_damage"),
            creep_damage=_counter(player, sample, "creep_damage"),
            neutral_damage=_counter(player, sample, "neutral_damage"),
            neutral_kills=_counter(player, sample, "neutral_kills"),
            duration_seconds=int(sample.get("time_stamp_s") or duration or 0),
        )
    return out


def fetch_match_stats(
    match_id: int | None,
    constants: GameConstants,
    *,
    offline: bool = False,
) -> dict[int, PlayerSkillStats]:
    """Mechanical stats from the network fallback, keyed by account ID.

    Returns an empty mapping whenever the record is unavailable. Callers must
    treat that as "unknown", never as "zero".
    """
    if match_id is None or offline:
        return {}
    return stats_from_metadata(_fetch_metadata(int(match_id)), constants)


def by_hero(
    stats: dict[int, PlayerSkillStats],
    steam_ids: dict[int, int | None],
) -> dict[int, PlayerSkillStats]:
    """Re-key account-ID stats onto the hero IDs the rest of the code uses."""
    out: dict[int, PlayerSkillStats] = {}
    # Hero IDs are unique inside one match and survive anonymised/private
    # replays.  Keep this as a fallback for demos where boon cannot recover a
    # Steam ID even though the post-match record itself is available.
    stats_by_hero = {stat.hero_id: stat for stat in stats.values()}
    for hero_id, steam_id in steam_ids.items():
        account = account_id_of(steam_id)
        if account is not None and account in stats:
            out[hero_id] = stats[account]
        elif hero_id in stats_by_hero:
            out[hero_id] = stats_by_hero[hero_id]
    return out
