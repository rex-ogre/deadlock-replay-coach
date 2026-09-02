"""Build the compact, persistent payload used by the browser replay viewer.

The coaching JSON is intentionally prose/analysis heavy.  A visual replay has
the opposite requirements: sampled positions plus small change-only streams
for inventory and abilities.  Keeping this as a separate artifact means the
model never pays for tens of thousands of map points, while the web UI can
still work after the uploaded ``.dem`` has been deleted.
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable

from opencc import OpenCC

from .events import build_timeline
from .gamedata import ASSETS_BASE, CACHE_TTL_SECONDS, HTTP_TIMEOUT_SECONDS, cache_dir
from .match import MatchData
from .names import Names

log = logging.getLogger(__name__)

VIEWER_SCHEMA_VERSION = 2
DEFAULT_MAP_RADIUS = 10_752
_CACHE_NAME = "visual_assets_v2.json"
_MAP_LANDMARKS = Path(__file__).parent / "data" / "map_landmarks.json"
_S2TW = OpenCC("s2twp")
_UNIVERSAL_ABILITIES = {
    "citadel_ability_climb_rope",
    "citadel_ability_dash",
    "citadel_ability_sprint",
    "citadel_ability_melee_parry",
    "citadel_ability_jump",
    "citadel_ability_mantle",
    "citadel_ability_slide",
    "citadel_ability_zip_line",
    "citadel_ability_zipline_boost",
}


class _PlainText(HTMLParser):
    """Reduce game tooltip HTML/SVG to safe, readable plain text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("svg", "style", "script"):
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "br":
            self.parts.append("\n")
        elif tag == "img":
            alt = dict(attrs).get("alt")
            if alt:
                self.parts.append(f" {alt} ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("svg", "style", "script") and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


def plain_text(value: object, *, limit: int = 1_200) -> str:
    if not value:
        return ""
    parser = _PlainText()
    try:
        parser.feed(str(value))
        parser.close()
        text = "".join(parser.parts)
    except Exception:
        text = re.sub(r"<[^>]+>", " ", str(value))
    text = html.unescape(text).replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text).strip()
    return _S2TW.convert(text[:limit])


def _cache_path() -> Path:
    return cache_dir() / _CACHE_NAME


def _read_cache(*, allow_stale: bool) -> dict | None:
    path = _cache_path()
    try:
        age = time.time() - path.stat().st_mtime
        if not allow_stale and age > CACHE_TTL_SECONDS:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _write_cache(value: dict) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        log.warning("could not cache visual assets (%s)", exc)


def _get_json(path: str, **query: object) -> Any:
    url = f"{ASSETS_BASE}/{path.lstrip('/')}"
    cleaned = {key: value for key, value in query.items() if value is not None}
    if cleaned:
        url += "?" + urllib.parse.urlencode(cleaned)
    request = urllib.request.Request(url, headers={"User-Agent": "deadlock-replay-coach/0.1"})
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_visual_assets() -> dict:
    """Fetch one internally consistent asset snapshot from the static API."""
    versions = _get_json("client-versions")
    version = max(int(value) for value in versions)
    common = {"client_version": version}
    return {
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "client_version": version,
        "source": "live",
        "map": _get_json("map", **common),
        "heroes": _get_json("heroes", language="english", only_active="true", **common),
        "items": _get_json("items", language="english", **common),
        # The API currently has Simplified Chinese but not Traditional Chinese.
        # Convert its display strings when the compact viewer payload is built.
        "heroes_zh": _get_json("heroes", language="schinese", only_active="true", **common),
        "items_zh": _get_json("items", language="schinese", **common),
    }


def load_visual_assets(*, offline: bool = False, refresh: bool = False) -> dict:
    if not refresh:
        cached = _read_cache(allow_stale=offline)
        if cached:
            return {**cached, "source": "cache"}
    if not offline:
        try:
            payload = fetch_visual_assets()
            _write_cache(payload)
            return payload
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError) as exc:
            log.warning("visual assets unavailable; viewer will use fallbacks (%s: %s)", type(exc).__name__, exc)
    stale = _read_cache(allow_stale=True)
    if stale:
        return {**stale, "source": "stale-cache"}
    return {
        "fetched_at": "",
        "client_version": None,
        "source": "unavailable",
        "map": {
            "radius": DEFAULT_MAP_RADIUS,
            "images": {},
            "objective_positions": {},
            "zipline_paths": [],
        },
        "heroes": [],
        "items": [],
        "heroes_zh": [],
        "items_zh": [],
    }


def _image(asset: dict, *, item: bool = False) -> str:
    keys = (
        ("shop_image_webp", "shop_image", "image_webp", "image")
        if item
        else ("image_webp", "image", "shop_image_webp", "shop_image")
    )
    for key in keys:
        if asset.get(key):
            return str(asset[key])
    return ""


def _hero_image(hero: dict) -> str:
    images = hero.get("images") or {}
    for key in ("minimap_image_webp", "minimap_image", "icon_image_small_webp", "icon_image_small"):
        if images.get(key):
            return str(images[key])
    return ""


def _description(asset: dict) -> dict[str, str]:
    raw = asset.get("description") or {}
    if not isinstance(raw, dict):
        return {"summary": plain_text(raw)}
    fields = {
        "summary": raw.get("quip") or raw.get("desc") or "",
        "details": raw.get("desc") or "",
        "t1": raw.get("t1_desc") or "",
        "t2": raw.get("t2_desc") or "",
        "t3": raw.get("t3_desc") or "",
    }
    return {key: plain_text(value) for key, value in fields.items() if value}


def _bonuses(asset: dict) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    for prop in (asset.get("properties") or {}).values():
        if not isinstance(prop, dict) or not prop.get("label"):
            continue
        if not (
            prop.get("tooltip_section")
            or prop.get("tooltip_is_elevated")
            or prop.get("tooltip_is_important")
        ):
            continue
        raw = prop.get("value")
        if raw in (None, "", "0", "0.0"):
            continue
        value = f"{prop.get('prefix') or ''}{raw}{prop.get('postfix') or ''}"
        values.append({"label": plain_text(prop["label"], limit=80), "value": plain_text(value, limit=80)})
        if len(values) >= 12:
            break
    return values


def _asset_row(asset: dict) -> dict:
    kind = str(asset.get("type") or "item")
    class_name = str(asset.get("class_name") or "")
    return {
        "id": asset.get("id"),
        "class_name": class_name,
        "name": plain_text(asset.get("name") or asset.get("class_name") or asset.get("id") or "Unknown", limit=120),
        "kind": kind,
        "image": _image(asset, item=kind == "upgrade"),
        "cost": asset.get("cost"),
        "tier": asset.get("item_tier"),
        "slot_type": asset.get("item_slot_type"),
        "heroes": [int(value) for value in (asset.get("heroes") or []) if value is not None],
        "start_trained": bool(asset.get("start_trained", False)),
        "core_skill": kind == "ability" and _is_core_skill(class_name),
        "description": _description(asset),
        "bonuses": _bonuses(asset),
    }


def _localized_asset(asset: dict) -> dict:
    """The display-only fields that differ between English and Chinese."""
    return {
        "name": plain_text(
            asset.get("name") or asset.get("class_name") or asset.get("id") or "Unknown",
            limit=120,
        ),
        "description": _description(asset),
        "bonuses": _bonuses(asset),
    }


def _is_core_skill(class_name: str) -> bool:
    """Exclude universal movement/melee entities from the four-skill bar."""
    return bool(class_name) and not (
        class_name in _UNIVERSAL_ABILITIES or class_name.startswith("ability_melee_")
    )


def _rows(frame, columns: Iterable[str]) -> list[dict]:
    wanted = [column for column in columns if column in frame.columns]
    if frame.is_empty() or not wanted:
        return []
    return [
        {key: value for key, value in row.items() if value is not None}
        for row in frame.select(wanted).sort("tick").iter_rows(named=True)
    ]


def _map_landmarks() -> dict:
    """Load the versioned overlay points bundled with the replay viewer."""
    try:
        value = json.loads(_MAP_LANDMARKS.read_text(encoding="utf-8"))
        if isinstance(value, dict) and isinstance(value.get("landmarks"), list):
            return value
    except (OSError, ValueError) as exc:
        log.warning("could not load bundled map landmarks (%s)", exc)
    return {"schema_version": 1, "coordinate_size": 1024, "landmarks": []}


def upgrade_viewer_payload(payload: dict, *, assets: dict | None = None) -> dict:
    """Add map overlays to a persisted schema-v1 viewer without its demo.

    Old jobs commonly outlive their uploaded replay, so asking the user to
    decode them again is not a viable migration path. The position/event rows
    remain untouched; only versioned static map data is filled in.
    """
    if int(payload.get("schema_version") or 0) >= VIEWER_SCHEMA_VERSION:
        return payload
    landmark_data = _map_landmarks()
    visual_assets = assets or load_visual_assets(offline=True)
    asset_map = visual_assets.get("map") or {}
    map_data = dict(payload.get("map") or {})
    map_data.setdefault("radius", asset_map.get("radius") or DEFAULT_MAP_RADIUS)
    map_data.setdefault("image", (asset_map.get("images") or {}).get("minimap", ""))
    map_data.setdefault("objective_positions", asset_map.get("objective_positions") or {})
    map_data.setdefault("zipline_paths", asset_map.get("zipline_paths") or [])
    map_data.setdefault("landmark_coordinate_size", landmark_data.get("coordinate_size") or 1024)
    map_data.setdefault(
        "landmark_snapshot",
        {
            "schema_version": landmark_data.get("schema_version") or 1,
            "updated_at": landmark_data.get("updated_at") or "",
        },
    )
    map_data.setdefault("landmarks", landmark_data.get("landmarks") or [])
    return {**payload, "schema_version": VIEWER_SCHEMA_VERSION, "map": map_data}


def render_viewer_json(
    md: MatchData,
    names: Names | None = None,
    *,
    assets: dict | None = None,
    offline: bool = False,
) -> str:
    """Serialize everything needed for map/inventory/skill time travel."""
    names = names or Names.from_boon()
    assets = assets or load_visual_assets(offline=offline)
    assert md.clock is not None

    hero_ids = set(md.hero_ids)
    purchase_ids = set(md.item_purchases["ability_id"].drop_nulls().to_list())
    upgrade_ids = {
        int(value)
        for value in (
            md.ability_upgrades["ability_id"].drop_nulls().to_list()
            if not md.ability_upgrades.is_empty()
            else []
        )
    }
    state_ids = {
        int(value)
        for value in (
            md.ability_ticks["ability_id"].drop_nulls().to_list()
            if not md.ability_ticks.is_empty()
            else []
        )
    }
    used_classes = {
        str(value)
        for value in (
            md.ability_uses["ability"].drop_nulls().to_list()
            if not md.ability_uses.is_empty()
            else []
        )
    }

    hero_assets = {
        int(row["id"]): row
        for row in assets.get("heroes", [])
        if isinstance(row, dict) and row.get("id") in hero_ids
    }
    hero_assets_zh = {
        int(row["id"]): row
        for row in assets.get("heroes_zh", [])
        if isinstance(row, dict) and row.get("id") in hero_ids
    }
    item_assets_zh = {
        int(row["id"]): row
        for row in assets.get("items_zh", [])
        if isinstance(row, dict) and row.get("id") is not None
    }
    relevant_assets: dict[str, dict] = {}
    for raw in assets.get("items", []):
        if not isinstance(raw, dict) or raw.get("id") is None:
            continue
        raw_id = int(raw["id"])
        owners = {int(value) for value in (raw.get("heroes") or []) if value is not None}
        core_skill = str(raw.get("type")) == "ability" and _is_core_skill(
            str(raw.get("class_name") or "")
        )
        relevant = (
            raw_id in purchase_ids
            or (
                core_skill
                and (
                    raw_id in upgrade_ids
                    or raw_id in state_ids
                    or str(raw.get("class_name") or "") in used_classes
                    or bool(owners & hero_ids)
                )
            )
        )
        if relevant:
            packed = _asset_row(raw)
            if raw_id in item_assets_zh:
                packed["translations"] = {"zh-TW": _localized_asset(item_assets_zh[raw_id])}
            relevant_assets[str(raw_id)] = packed

    roster = []
    for row in md.players.iter_rows(named=True):
        hero_id = row.get("hero_id")
        if not hero_id:
            continue
        hero_asset = hero_assets.get(int(hero_id), {})
        packed_hero = {
                "hero_id": int(hero_id),
                "hero": plain_text(hero_asset.get("name") or names.hero(hero_id), limit=120),
                "player_name": row.get("player_name") or "",
                "team_num": row.get("team_num"),
                "team": names.team(row.get("team_num")),
                "image": _hero_image(hero_asset),
            }
        localized_hero = hero_assets_zh.get(int(hero_id))
        if localized_hero:
            packed_hero["translations"] = {
                "zh-TW": {"hero": plain_text(localized_hero.get("name"), limit=120)}
            }
        roster.append(packed_hero)

    positions = []
    clock_ticks: set[int] = {0, md.end_tick}
    if not md.player_samples.is_empty():
        for row in md.player_samples.sort(["tick", "hero_id"]).iter_rows(named=True):
            tick = row.get("tick")
            hero_id = row.get("hero_id")
            if tick is None or hero_id is None or row.get("x") is None or row.get("y") is None:
                continue
            clock_ticks.add(int(tick))
            positions.append(
                [
                    int(tick), int(hero_id), round(float(row["x"]), 1), round(float(row["y"]), 1),
                    round(float(row.get("z") or 0), 1), bool(row.get("is_alive", True)),
                    row.get("health"), row.get("max_health"),
                    (
                        int(row.get("souls") or 0) + int(row.get("spent_souls") or 0)
                        if row.get("souls") is not None or row.get("spent_souls") is not None
                        else None
                    ),
                    row.get("kills"), row.get("deaths"), row.get("assists"),
                ]
            )

    map_data = assets.get("map") or {}
    landmark_data = _map_landmarks()
    timeline = [
        {"tick": event.tick, "clock": md.clock.mmss(event.tick), "kind": event.kind, "text": event.text}
        for event in build_timeline(md, names)
    ]
    inventory_events = _rows(md.item_purchases, ("tick", "hero_id", "ability_id", "change"))
    ability_upgrades = _rows(md.ability_upgrades, ("tick", "hero_id", "ability_id", "tier"))
    core_ability_ids = {
        int(raw_id) for raw_id, asset in relevant_assets.items() if asset.get("core_skill")
    } | upgrade_ids
    ability_states = [
        event
        for event in _rows(
            md.ability_ticks,
            (
                "tick", "hero_id", "ability_id", "slot", "cooldown_start", "cooldown_end",
                "remaining_charges", "charge_recharge_start", "charge_recharge_end",
            ),
        )
        if int(event.get("ability_id", -1)) in core_ability_ids
    ]
    hero_labels = {row["hero_id"]: row["hero"] for row in roster}
    for event in inventory_events:
        asset = relevant_assets.get(str(event.get("ability_id")), {})
        localized = (asset.get("translations") or {}).get("zh-TW") or {}
        hero = hero_labels.get(event.get("hero_id"), names.hero(event.get("hero_id")))
        timeline.append(
            {
                "tick": event["tick"],
                "clock": md.clock.mmss(event["tick"]),
                "kind": "purchase",
                "text": f"{hero} inventory: {asset.get('name') or names.ability(event.get('ability_id'))} "
                f"({event.get('change') or 'changed'})",
                "translations": {
                    "zh-TW": f"{hero} 裝備變更："
                    f"{localized.get('name') or asset.get('name') or names.ability(event.get('ability_id'))} "
                    f"({event.get('change') or 'changed'})"
                },
            }
        )
    for event in ability_upgrades:
        asset = relevant_assets.get(str(event.get("ability_id")), {})
        localized = (asset.get("translations") or {}).get("zh-TW") or {}
        hero = hero_labels.get(event.get("hero_id"), names.hero(event.get("hero_id")))
        timeline.append(
            {
                "tick": event["tick"],
                "clock": md.clock.mmss(event["tick"]),
                "kind": "ability_upgrade",
                "text": f"{hero} upgraded {asset.get('name') or names.ability(event.get('ability_id'))} "
                f"to T{event.get('tier')}",
                "translations": {
                    "zh-TW": f"{hero} 升級 "
                    f"{localized.get('name') or asset.get('name') or names.ability(event.get('ability_id'))} "
                    f"至 T{event.get('tier')}"
                },
            }
        )
    timeline.sort(key=lambda event: (event["tick"], event["kind"], event["text"]))
    payload = {
        "schema_version": VIEWER_SCHEMA_VERSION,
        "match": {
            "match_id": md.match_id,
            "map_name": md.map_name,
            "build": md.build,
            "tick_rate": md.tick_rate,
            "end_tick": md.end_tick,
            "duration": md.clock.mmss(md.end_tick),
            "teams": {str(team): names.team(team) for team in md.team_nums},
        },
        "asset_snapshot": {
            "client_version": assets.get("client_version"),
            "fetched_at": assets.get("fetched_at"),
            "source": assets.get("source"),
        },
        "map": {
            "radius": map_data.get("radius") or DEFAULT_MAP_RADIUS,
            "image": (map_data.get("images") or {}).get("minimap", ""),
            "objective_positions": map_data.get("objective_positions") or {},
            "zipline_paths": map_data.get("zipline_paths") or [],
            "landmark_coordinate_size": landmark_data.get("coordinate_size") or 1024,
            "landmark_snapshot": {
                "schema_version": landmark_data.get("schema_version") or 1,
                "updated_at": landmark_data.get("updated_at") or "",
            },
            "landmarks": landmark_data.get("landmarks") or [],
        },
        "roster": roster,
        "assets": relevant_assets,
        # Compact row: positional state plus the scoreboard counters shown by
        # the map marker tooltip at that same sample.
        "position_columns": [
            "tick", "hero_id", "x", "y", "z", "alive", "health", "max_health",
            "net_worth", "kills", "deaths", "assists",
        ],
        "positions": positions,
        "objective_states": _rows(
            md.objectives,
            (
                "tick", "entity_id", "objective_type", "team_num", "lane", "health",
                "max_health", "phase", "x", "y", "z",
            ),
        ),
        "neutral_states": _rows(
            md.neutrals,
            ("tick", "entity_id", "team_num", "health", "max_health", "x", "y", "z"),
        ),
        "clock": [[tick, round(md.clock.seconds(tick), 3)] for tick in sorted(clock_ticks)],
        "inventory_events": inventory_events,
        "ability_upgrades": ability_upgrades,
        "ability_states": ability_states,
        "ability_uses": _rows(md.ability_uses, ("tick", "hero_id", "ability")),
        "timeline": timeline,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
