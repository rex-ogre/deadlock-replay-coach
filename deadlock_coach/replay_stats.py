"""Read the post-match counters embedded in a completed Deadlock replay.

Boon decodes ``PostMatchDetails`` for :meth:`boon.Demo.summary`, but version
0.7.0 deliberately selects only a subset of ``PlayerStats`` when it builds the
returned DataFrame.  In particular, it drops ``shots_hit`` and
``shots_missed`` even though the protobuf payload contains both.  This module
does one narrow, read-only pass over the PBDEMS2 command stream and extracts
that single message without attempting to duplicate Boon's entity parser.

The returned object intentionally has the same shape as Deadlock API metadata,
so the existing skill-stat normalization and population benchmarks remain the
single source of truth.
"""

from __future__ import annotations

import logging
import mmap
from collections.abc import Iterator
from pathlib import Path

import cramjam

log = logging.getLogger(__name__)

_MAGIC = b"PBDEMS2\0"
_HEADER_SIZE = 16
_COMPRESSED = 64
_PACKET_COMMANDS = {7, 8}
_FULL_PACKET = 13
_SVC_USER_MESSAGE = 72
_POST_MATCH_DETAILS = 316


class ReplayStatsError(ValueError):
    """The narrow post-match scan could not safely decode the replay."""


def _read_varint(data, pos: int, *, limit: int | None = None) -> tuple[int, int]:
    end = len(data) if limit is None else min(len(data), limit)
    value = 0
    for shift in range(0, 70, 7):
        if pos >= end:
            raise ReplayStatsError("truncated varint")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
    raise ReplayStatsError("varint is too long")


def _fields(data: bytes) -> Iterator[tuple[int, int, int | bytes]]:
    """Yield protobuf fields without generated message classes."""
    pos = 0
    while pos < len(data):
        key, pos = _read_varint(data, pos)
        number, wire = key >> 3, key & 7
        if not number:
            raise ReplayStatsError("invalid protobuf field number")
        if wire == 0:
            value, pos = _read_varint(data, pos)
            yield number, wire, value
        elif wire == 1:
            if pos + 8 > len(data):
                raise ReplayStatsError("truncated fixed64 field")
            yield number, wire, data[pos : pos + 8]
            pos += 8
        elif wire == 2:
            size, pos = _read_varint(data, pos)
            if size < 0 or pos + size > len(data):
                raise ReplayStatsError("truncated length-delimited field")
            yield number, wire, data[pos : pos + size]
            pos += size
        elif wire == 5:
            if pos + 4 > len(data):
                raise ReplayStatsError("truncated fixed32 field")
            yield number, wire, data[pos : pos + 4]
            pos += 4
        else:
            raise ReplayStatsError(f"unsupported protobuf wire type {wire}")


def _message(data: bytes, number: int) -> bytes | None:
    return next(
        (value for field, wire, value in _fields(data) if field == number and wire == 2),
        None,
    )


def _messages(data: bytes, number: int) -> list[bytes]:
    return [
        value
        for field, wire, value in _fields(data)
        if field == number and wire == 2 and isinstance(value, bytes)
    ]


def _varints(data: bytes) -> dict[int, int]:
    return {
        field: int(value)
        for field, wire, value in _fields(data)
        if wire == 0 and isinstance(value, int)
    }


class _BitReader:
    """Only the three packet-stream primitives required by Source 2 messages."""

    def __init__(self, data: bytes):
        self.data = data
        self.position = 0
        self.total_bits = len(data) * 8

    @property
    def remaining(self) -> int:
        return self.total_bits - self.position

    def read_bits(self, count: int) -> int:
        if count < 0 or self.position + count > self.total_bits:
            raise ReplayStatsError("packet bitstream is truncated")
        if not count:
            return 0
        start = self.position // 8
        offset = self.position % 8
        byte_count = (offset + count + 7) // 8
        word = int.from_bytes(self.data[start : start + byte_count], "little")
        self.position += count
        return (word >> offset) & ((1 << count) - 1)

    def read_u8(self) -> int:
        return self.read_bits(8)

    def read_varint(self) -> int:
        value = 0
        for index in range(5):
            byte = self.read_u8()
            value |= (byte & 0x7F) << (7 * index)
            if not byte & 0x80:
                return value
        return value

    def read_ubitvar(self) -> int:
        value = self.read_bits(6)
        selector = value & 48
        if selector == 16:
            return (value & 15) | self.read_bits(4) << 4
        if selector == 32:
            return (value & 15) | self.read_bits(8) << 4
        if selector == 48:
            return (value & 15) | self.read_bits(28) << 4
        return value

    def read_bytes(self, size: int) -> bytes:
        if size < 0 or self.position + size * 8 > self.total_bits:
            raise ReplayStatsError("packet message is truncated")
        if self.position % 8 == 0:
            start = self.position // 8
            self.position += size * 8
            return self.data[start : start + size]
        return bytes(self.read_u8() for _ in range(size))

    def skip_bytes(self, size: int) -> None:
        if size < 0 or self.position + size * 8 > self.total_bits:
            raise ReplayStatsError("packet message is truncated")
        self.position += size * 8


def _post_match_payload(packet: bytes) -> bytes | None:
    reader = _BitReader(packet)
    while reader.remaining > 8:
        message_type = reader.read_ubitvar()
        size = reader.read_varint()
        if message_type not in (_POST_MATCH_DETAILS, _SVC_USER_MESSAGE):
            reader.skip_bytes(size)
            continue
        payload = reader.read_bytes(size)
        if message_type == _POST_MATCH_DETAILS:
            return payload
        envelope = _varints(payload)
        if envelope.get(1) == _POST_MATCH_DETAILS:
            return _message(payload, 2)
    return None


_PLAYER_ROW_FIELDS = {
    1: "account_id",
    8: "kills",
    9: "deaths",
    10: "assists",
    11: "net_worth",
    12: "hero_id",
    13: "last_hits",
    14: "denies",
}

_STAT_FIELDS = {
    1: "time_stamp_s",
    2: "net_worth",
    14: "kills",
    15: "deaths",
    16: "assists",
    17: "creep_kills",
    18: "neutral_kills",
    19: "possible_creeps",
    20: "creep_damage",
    21: "player_damage",
    22: "neutral_damage",
    23: "boss_damage",
    24: "denies",
    28: "player_damage_taken",
    32: "shots_hit",
    33: "shots_missed",
    36: "hero_bullets_hit",
    37: "hero_bullets_hit_crit",
}


def decode_post_match_details(payload: bytes) -> dict | None:
    """Decode one ``CCitadelUserMsgPostMatchDetails`` payload."""
    details = _message(payload, 1)
    contents = _message(details, 2) if details else None
    if not contents:
        return None

    match_values = _varints(contents)
    players = []
    for raw_player in _messages(contents, 4):
        player_values = _varints(raw_player)
        player = {
            name: player_values[field]
            for field, name in _PLAYER_ROW_FIELDS.items()
            if field in player_values
        }
        samples = []
        for raw_sample in _messages(raw_player, 5):
            values = _varints(raw_sample)
            samples.append(
                {
                    name: values[field]
                    for field, name in _STAT_FIELDS.items()
                    if field in values
                }
            )
        if samples:
            samples.sort(key=lambda row: int(row.get("time_stamp_s") or 0))
            player["stats"] = samples
        if player.get("account_id") is not None and player.get("hero_id") is not None:
            players.append(player)
    if not players:
        return None
    return {
        "match_info": {
            "duration_s": match_values.get(1),
            "average_badge_team0": match_values.get(23),
            "average_badge_team1": match_values.get(24),
            "players": players,
        },
        "stats_source": "replay-post-match-details",
    }


def _packet_data(command: int, body: bytes) -> bytes | None:
    if command in _PACKET_COMMANDS:
        return _message(body, 3)
    if command == _FULL_PACKET:
        packet = _message(body, 2)
        return _message(packet, 3) if packet else None
    return None


def read_replay_metadata(path: str | Path) -> dict | None:
    """Return API-shaped post-match metadata from a completed ``.dem``.

    An incomplete or older replay without ``PostMatchDetails`` returns ``None``
    and lets the caller use its existing network fallback.
    """
    try:
        with Path(path).open("rb") as handle, mmap.mmap(
            handle.fileno(), 0, access=mmap.ACCESS_READ
        ) as data:
            if len(data) < _HEADER_SIZE or data[:8] != _MAGIC:
                raise ReplayStatsError("not a PBDEMS2 replay")
            pos = _HEADER_SIZE
            while pos < len(data):
                raw_command, pos = _read_varint(data, pos)
                command = raw_command & ~_COMPRESSED
                _tick, pos = _read_varint(data, pos)
                body_size, pos = _read_varint(data, pos)
                if command == 0:
                    break
                if body_size < 0 or pos + body_size > len(data):
                    raise ReplayStatsError("demo command body is truncated")
                if command not in _PACKET_COMMANDS and command != _FULL_PACKET:
                    pos += body_size
                    continue
                raw_body = data[pos : pos + body_size]
                pos += body_size
                body = (
                    bytes(cramjam.snappy.decompress_raw(raw_body))
                    if raw_command & _COMPRESSED
                    else raw_body
                )
                packet = _packet_data(command, body)
                if not packet:
                    continue
                payload = _post_match_payload(packet)
                if payload:
                    return decode_post_match_details(payload)
    except (OSError, ValueError, RuntimeError, ReplayStatsError) as exc:
        log.info("replay post-match stats unavailable (%s: %s)", type(exc).__name__, exc)
    return None
