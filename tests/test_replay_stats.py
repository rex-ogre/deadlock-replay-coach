from __future__ import annotations

import cramjam
import pytest

from deadlock_coach import gamedata
from deadlock_coach.replay_stats import (
    decode_post_match_details,
    read_replay_metadata,
)
from deadlock_coach.skillstats import stats_from_metadata


def _varint(value: int) -> bytes:
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _number(field: int, value: int) -> bytes:
    return _varint(field << 3) + _varint(value)


def _message(field: int, value: bytes) -> bytes:
    return _varint((field << 3) | 2) + _varint(len(value)) + value


def _post_match_payload() -> bytes:
    early = b"".join(
        [_number(1, 60), _number(32, 10), _number(33, 10)]
    )
    final = b"".join(
        [
            _number(1, 1800),
            _number(14, 9),
            _number(15, 4),
            _number(16, 12),
            _number(17, 80),
            _number(19, 100),
            _number(21, 50_000),
            _number(28, 40_000),
            _number(32, 600),
            _number(33, 400),
            _number(36, 180),
            _number(37, 18),
        ]
    )
    player = b"".join(
        [
            _number(1, 492_740_207),
            _message(5, early),
            _message(5, final),
            _number(8, 9),
            _number(9, 4),
            _number(10, 12),
            _number(11, 41_766),
            _number(12, 1),
            _number(13, 199),
            _number(14, 7),
        ]
    )
    match_info = b"".join(
        [
            _number(1, 1800),
            _message(4, player),
            _number(23, 62),
            _number(24, 64),
        ]
    )
    contents = _message(2, match_info)
    return _message(1, contents)


class _BitWriter:
    def __init__(self):
        self.bits: list[int] = []

    def write(self, value: int, count: int) -> None:
        self.bits.extend((value >> bit) & 1 for bit in range(count))

    def ubitvar(self, value: int) -> None:
        if value < 16:
            self.write(value, 6)
        elif value < 256:
            self.write((value & 15) | 16, 6)
            self.write(value >> 4, 4)
        elif value < 4096:
            self.write((value & 15) | 32, 6)
            self.write(value >> 4, 8)
        else:
            self.write((value & 15) | 48, 6)
            self.write(value >> 4, 28)

    def varint(self, value: int) -> None:
        for byte in _varint(value):
            self.write(byte, 8)

    def bytes(self, value: bytes) -> None:
        for byte in value:
            self.write(byte, 8)

    def finish(self) -> bytes:
        out = bytearray((len(self.bits) + 7) // 8)
        for index, bit in enumerate(self.bits):
            out[index // 8] |= bit << (index % 8)
        return bytes(out)


def _demo_bytes(*, compressed: bool) -> bytes:
    payload = _post_match_payload()
    writer = _BitWriter()
    writer.ubitvar(316)
    writer.varint(len(payload))
    writer.bytes(payload)
    packet_body = _message(3, writer.finish())
    stored = bytes(cramjam.snappy.compress_raw(packet_body)) if compressed else packet_body
    command = 7 | (64 if compressed else 0)
    header = _varint(command) + _varint(1234) + _varint(len(stored))
    stop = _varint(0) + _varint(1235) + _varint(0)
    return b"PBDEMS2\0" + b"\0" * 8 + header + stored + stop


def test_decodes_the_replay_post_match_counters():
    payload = decode_post_match_details(_post_match_payload())
    stat = stats_from_metadata(payload, gamedata.load_constants(offline=True))[492_740_207]

    assert payload["stats_source"] == "replay-post-match-details"
    assert stat.hero_id == 1
    assert stat.shots_hit == 600
    assert stat.shots_missed == 400
    assert stat.accuracy == pytest.approx(0.6)
    assert stat.hero_bullets_hit == 180
    assert stat.hero_bullets_hit_crit == 18
    assert stat.net_worth == 41_766
    assert stat.last_hits == 199


@pytest.mark.parametrize("compressed", [False, True])
def test_scans_direct_post_match_message_from_demo(tmp_path, compressed):
    path = tmp_path / "match.dem"
    path.write_bytes(_demo_bytes(compressed=compressed))

    payload = read_replay_metadata(path)

    final = payload["match_info"]["players"][0]["stats"][-1]
    assert final["shots_hit"] == 600
    assert final["shots_missed"] == 400


def test_incomplete_replay_returns_none(tmp_path):
    path = tmp_path / "partial.dem"
    path.write_bytes(b"PBDEMS2\0" + b"\0" * 8 + b"\0\0\0")
    assert read_replay_metadata(path) is None
