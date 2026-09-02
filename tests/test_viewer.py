from __future__ import annotations

import json

from deadlock_coach import viewer
from deadlock_coach.viewer import (
    load_visual_assets,
    plain_text,
    render_viewer_json,
    upgrade_viewer_payload,
)

from .conftest import MINUTE, frame


def visual_assets() -> dict:
    return {
        "source": "test",
        "client_version": 6684,
        "fetched_at": "2026-08-22T10:38:36+00:00",
        "map": {
            "radius": 10_752,
            "images": {"minimap": "https://assets.example/minimap.webp"},
            "objective_positions": {"team0_core": {"left_relative": 0.4, "top_relative": 0.9}},
            "zipline_paths": [{
                "origin": [0, -10_752, 0],
                "color": "#29b1cc",
                "P0_points": [[0, 0, 0], [0, 21_504, 0]],
            }],
        },
        "heroes": [
            {
                "id": 1,
                "name": "炽焱",
                "images": {"minimap_image_webp": "https://assets.example/infernus.webp"},
            }
        ],
        "items": [
            {
                "id": 101,
                "class_name": "upgrade_clip_size",
                "name": "扩容弹匣",
                "type": "upgrade",
                "shop_image_webp": "https://assets.example/mag.webp",
                "cost": 800,
                "properties": {
                    "clip": {
                        "label": "最大弹药量",
                        "value": "30",
                        "prefix": "+",
                        "postfix": "%",
                        "tooltip_section": "innate",
                    }
                },
            },
            {
                "id": 201,
                "class_name": "ability_flame_dash",
                "name": "烈焰冲刺",
                "type": "ability",
                "heroes": [1],
                "image_webp": "https://assets.example/dash.webp",
                "description": {
                    "quip": "向前冲刺",
                    "desc": "向前冲刺并留下<strong>火焰</strong>。",
                    "t2_desc": "+1秒持续时间",
                },
            },
            {
                "id": 301,
                "class_name": "citadel_ability_dash",
                "name": "冲刺",
                "type": "ability",
                "heroes": [1],
                "start_trained": True,
            },
            {"id": 999, "class_name": "unused", "name": "无关物品", "type": "upgrade"},
        ],
    }


def test_tooltip_html_becomes_traditional_plain_text():
    source = '向前冲刺<svg><path></path>忽略</svg><br><span>留下火焰</span>'
    assert plain_text(source) == "向前衝刺\n留下火焰"


def test_viewer_persists_map_positions_inventory_and_skill_state(match, names):
    enriched = match.with_frames(
        item_purchases=frame(
            "item_purchases",
            [{"tick": 4 * MINUTE, "hero_id": 1, "ability_id": 101, "change": "purchased"}],
        ),
        ability_upgrades=frame(
            "ability_upgrades",
            [{"tick": 3 * MINUTE, "hero_id": 1, "ability_id": 201, "tier": 2}],
        ),
        ability_ticks=frame(
            "ability_ticks",
            [{
                "tick": 5 * MINUTE,
                "hero_id": 1,
                "ability_id": 201,
                "slot": 1,
                "cooldown_start": 300.0,
                "cooldown_end": 312.0,
                "remaining_charges": 0,
            }],
        ),
        ability_uses=frame(
            "ability_uses",
            [{"tick": 5 * MINUTE, "hero_id": 1, "ability": "ability_flame_dash"}],
        ),
        neutrals=frame(
            "neutrals",
            [{
                "tick": 5 * MINUTE,
                "entity_id": 9001,
                "team_num": 4,
                "health": 750,
                "max_health": 1000,
                "x": 250.0,
                "y": -400.0,
                "z": 0.0,
            }],
        ),
    )

    payload = json.loads(render_viewer_json(enriched, names, assets=visual_assets()))

    assert payload["schema_version"] == 2
    assert payload["map"]["radius"] == 10_752
    assert len(payload["map"]["zipline_paths"]) == 1
    assert len([row for row in payload["map"]["landmarks"] if row["kind"] == "camp"]) == 38
    assert len([row for row in payload["map"]["landmarks"] if row["kind"] == "buff"]) == 2
    assert all(row["label"].isascii() for row in payload["map"]["landmarks"])
    assert payload["positions"][0][0:2] == [5 * MINUTE, 1]
    assert payload["position_columns"][-4:] == ["net_worth", "kills", "deaths", "assists"]
    assert payload["positions"][0][8] == 9_000
    assert payload["positions"][0][9:12] == [0, 0, 0]
    assert payload["objective_states"]
    assert payload["neutral_states"][0]["entity_id"] == 9001
    assert payload["inventory_events"][0]["ability_id"] == 101
    assert payload["ability_upgrades"][0]["tier"] == 2
    assert payload["ability_states"][0]["cooldown_end"] == 312.0
    assert payload["ability_uses"][0]["ability"] == "ability_flame_dash"
    assert any(event["kind"] == "purchase" and "擴容彈匣" in event["text"] for event in payload["timeline"])
    assert any(event["kind"] == "ability_upgrade" and "烈焰衝刺" in event["text"] for event in payload["timeline"])
    assert payload["assets"]["101"]["name"] == "擴容彈匣"
    assert payload["assets"]["201"]["description"]["details"] == "向前衝刺並留下火焰。"
    assert "999" not in payload["assets"]
    assert "301" not in payload["assets"], "universal movement must not flood the four-skill bar"
    assert payload["roster"][0]["image"] == "https://assets.example/infernus.webp"


def test_viewer_packs_english_and_chinese_asset_names_once(match, names):
    assets = visual_assets()
    chinese_heroes = assets["heroes"]
    chinese_items = assets["items"]
    assets["heroes"] = [{**chinese_heroes[0], "name": "Infernus"}]
    english_names = {
        101: "Basic Magazine",
        201: "Flame Dash",
        301: "Dash",
        999: "Unused Item",
    }
    assets["items"] = [{**row, "name": english_names[row["id"]]} for row in chinese_items]
    assets["heroes_zh"] = chinese_heroes
    assets["items_zh"] = chinese_items
    enriched = match.with_frames(
        item_purchases=frame(
            "item_purchases",
            [{"tick": 4 * MINUTE, "hero_id": 1, "ability_id": 101, "change": "purchased"}],
        ),
        ability_upgrades=frame(
            "ability_upgrades",
            [{"tick": 3 * MINUTE, "hero_id": 1, "ability_id": 201, "tier": 2}],
        ),
    )

    payload = json.loads(render_viewer_json(enriched, names, assets=assets))

    assert payload["roster"][0]["hero"] == "Infernus"
    assert payload["roster"][0]["translations"]["zh-TW"]["hero"] == "熾焱"
    assert payload["assets"]["101"]["name"] == "Basic Magazine"
    assert payload["assets"]["101"]["translations"]["zh-TW"]["name"] == "擴容彈匣"
    assert any("inventory: Basic Magazine" in row["text"] for row in payload["timeline"])
    assert any("擴容彈匣" in row["translations"]["zh-TW"] for row in payload["timeline"] if row["kind"] == "purchase")


def test_viewer_still_works_without_network_assets(match, names):
    payload = json.loads(
        render_viewer_json(
            match,
            names,
            assets={
                "source": "unavailable",
                "map": {"radius": 10_752, "images": {}},
                "heroes": [],
                "items": [],
            },
        )
    )
    assert payload["positions"]
    assert payload["map"]["image"] == ""
    assert payload["roster"][0]["hero"] == "Infernus"


def test_old_viewer_payload_gets_static_map_layers_without_a_demo():
    payload = upgrade_viewer_payload(
        {"schema_version": 1, "map": {"radius": 9_999}, "positions": [[1, 2, 3]]},
        assets=visual_assets(),
    )

    assert payload["schema_version"] == 2
    assert payload["positions"] == [[1, 2, 3]]
    assert payload["map"]["radius"] == 9_999
    assert len(payload["map"]["zipline_paths"]) == 1
    assert len(payload["map"]["landmarks"]) == 40


def test_offline_visual_assets_never_touch_the_network(tmp_path, monkeypatch):
    monkeypatch.setenv("DEADLOCK_COACH_CACHE", str(tmp_path))
    monkeypatch.setattr(viewer, "_get_json", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    assets = load_visual_assets(offline=True)
    assert assets["source"] == "unavailable"
    assert assets["map"]["radius"] == 10_752
