"""Milano şebeke verisini (Telecom Italia, 2 ay) tahmin modellerine hazırlar.

Veri, CESNET TS-Zoo'nun HDF5 kopyasından okunur. İlk okuma dosyanın tamamını taradığı
için uzun sürer; sonuç `data/milan_internet.npz` içinde saklanır.
"""

from pathlib import Path

import numpy as np
import tables

from nwl.protocol import HORIZON_SLOTS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
H5_PATH = (
    DATA_DIR
    / "tszoo/databases/TelecomItalia"
    / "TelecomItalia-telecommunications-service_provider-10_minutes.h5"
)
CACHE_PATH = DATA_DIR / "milan_internet.npz"
GRID_SIDE = 100
SQUARES = GRID_SIDE**2
READ_ROWS = 5_000_000
STEPS_PER_DAY = 144
BLOCK = 5  # bir kare yaklaşık 235 m, yani hücre yaklaşık 1,2 km'lik bir bölge
REFERENCE_LOAD = 0.75


def extract_internet(h5_path: Path = H5_PATH) -> dict[str, np.ndarray]:
    """Kare ve zaman başına internet trafiğini toplar.

    Az görülen (kare, ülke) çiftleri kaynakta tek seride birleştirilmiştir. Bu serinin
    hangi kareye ait olduğu bilinmediği için ayrı tutulur.
    """
    with tables.open_file(h5_path) as h5:
        times = h5.root.times.times_10_minutes.col("time")
        info = h5.root.ts_ids_info.read()
        rows = h5.root.service_provider.agg_10_minutes

        merged_id = np.bincount(info["ts_id"]).argmax()
        square_of_series = np.zeros(merged_id + 1, dtype=np.int64)
        single = info["ts_id"] != merged_id
        square_of_series[info["ts_id"][single]] = info["square_id"][single] - 1

        by_square = np.zeros(SQUARES * len(times))
        merged = np.zeros(len(times))
        for start in range(0, rows.nrows, READ_ROWS):
            chunk = rows.read(start, start + READ_ROWS)
            traffic = np.nan_to_num(chunk["internet_traffic"])
            is_merged = chunk["ts_id"] == merged_id
            merged += np.bincount(
                chunk["id_time"][is_merged], traffic[is_merged], minlength=len(times)
            )
            squares = square_of_series[chunk["ts_id"][~is_merged]]
            cells = squares * len(times) + chunk["id_time"][~is_merged]
            by_square += np.bincount(cells, traffic[~is_merged], minlength=by_square.size)

    return {
        "times": times,
        "by_square": by_square.reshape(SQUARES, len(times)),
        "merged": merged,
    }


def load_traffic() -> dict[str, np.ndarray]:
    if not CACHE_PATH.exists():
        np.savez_compressed(CACHE_PATH, **extract_internet())
    with np.load(CACHE_PATH) as cache:
        return dict(cache)


def sum_blocks(by_square: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """`block`x`block` karelik komşu grupları tek hücre gibi toplar."""
    side = GRID_SIDE // block
    grid = by_square.reshape(side, block, side, block, -1)
    return grid.sum(axis=(1, 3)).reshape(side * side, -1)


def to_slots(series: np.ndarray) -> np.ndarray:
    """10 dakikalık değerleri ortalamayı koruyarak 15 dakikalık slotlara çevirir."""
    five_minutes = np.repeat(series, 2)
    return five_minutes.reshape(-1, 3).mean(axis=1)


def to_load(slots: np.ndarray, train_slots: int) -> np.ndarray:
    """Eğitim döneminin yoğun saatleri REFERENCE_LOAD olacak şekilde ölçekler.

    Veri göreli birimlerdedir; kapasite bilinmediği için bu ölçek bir varsayımdır.
    """
    return slots / np.quantile(slots[:train_slots], 0.99) * REFERENCE_LOAD


def choose_cells(blocks: np.ndarray, train_steps: int) -> dict[str, int]:
    """Eğitim dönemine bakarak üç farklı bölge tipi seçer."""
    train = blocks[:, :train_steps]
    volume = train.sum(axis=1)
    by_time_of_day = train.reshape(len(train), -1, STEPS_PER_DAY).mean(axis=1)
    evening = by_time_of_day[:, 20 * 6 : 23 * 6].mean(axis=1)
    midday = by_time_of_day[:, 12 * 6 : 16 * 6].mean(axis=1)

    busy = np.flatnonzero(volume >= np.quantile(volume, 0.75))
    return {
        "Merkez": int(volume.argmax()),
        "Ortalama": int(volume.argsort()[len(volume) // 2]),
        "Akşam yoğun": int(busy[(evening[busy] / midday[busy]).argmax()]),
    }


def load_cells(train_days: int) -> dict[str, np.ndarray]:
    """Seçilen üç hücrenin yük serisi. Seriler 1 Kasım 2013 gece yarısı başlar."""
    blocks = sum_blocks(load_traffic()["by_square"])
    chosen = choose_cells(blocks, train_days * STEPS_PER_DAY)
    train_slots = train_days * HORIZON_SLOTS
    return {
        name: to_load(to_slots(blocks[index]), train_slots)
        for name, index in chosen.items()
    }
