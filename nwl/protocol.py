"""NWL yayın mesajının biçimi.

Baz istasyonu her 15 dakikada bir önümüzdeki 24 saatin tahminini yayınlar:

    Başlık (6 bayt):  issued_minute (uint32), entry_count (uint16)
    Kayıt (5 bayt):   time_slot (uint16), predicted_load (uint16), confidence (uint8)

predicted_load binde bir, confidence yüzde bir birimiyle saklanır.

confidence, tahminin beklenen mutlak hatasını taşır: 1 hatasız, 0 ise hatanın en az
CONFIDENCE_SCALE olduğu anlamına gelir. Böylece alıcı hatayı yükle aynı birime çevirebilir.
"""

import struct
from dataclasses import dataclass

SLOT_MINUTES = 15
HORIZON_SLOTS = 96
CONFIDENCE_SCALE = 0.5

_HEADER = struct.Struct("!IH")
_ENTRY = struct.Struct("!HHB")
_LOAD_UNIT = 1000
_CONFIDENCE_UNIT = 100


@dataclass(frozen=True)
class ForecastEntry:
    time_slot: int
    predicted_load: float
    confidence: float


@dataclass(frozen=True)
class ForecastMessage:
    issued_minute: int
    entries: tuple[ForecastEntry, ...]


def encode(message: ForecastMessage) -> bytes:
    payload = bytearray(_HEADER.pack(message.issued_minute, len(message.entries)))
    for entry in message.entries:
        payload += _ENTRY.pack(
            entry.time_slot,
            min(round(entry.predicted_load * _LOAD_UNIT), 65535),
            round(entry.confidence * _CONFIDENCE_UNIT),
        )
    return bytes(payload)


def decode(payload: bytes) -> ForecastMessage:
    issued_minute, count = _HEADER.unpack_from(payload)
    entries = tuple(
        ForecastEntry(slot, load / _LOAD_UNIT, confidence / _CONFIDENCE_UNIT)
        for slot, load, confidence in _ENTRY.iter_unpack(payload[_HEADER.size:])
    )
    if len(entries) != count:
        raise ValueError(f"{count} kayıt bekleniyordu, {len(entries)} geldi")
    return ForecastMessage(issued_minute, entries)


def broadcast_bits_per_second() -> float:
    """Tam boy yayınların hücreye getirdiği ek yük."""
    message_bytes = _HEADER.size + HORIZON_SLOTS * _ENTRY.size
    return message_bytes * 8 / (SLOT_MINUTES * 60)
