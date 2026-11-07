"""Tarayıcı simülasyonunun kullandığı verileri `simulation/data.js` dosyasına yazar.

Çalıştırmak için: python -m simulation.export_data
"""

import json
from pathlib import Path

from nwl.cell import CALIBRATED_CAPACITY_MBPS
from nwl.evaluate_forecasts import MILAN_NORMAL_END_DAY, TRAIN_DAYS
from nwl.milan import load_cells
from nwl.protocol import HORIZON_SLOTS

HERE = Path(__file__).resolve().parent
CALIBRATION = HERE.parent / "calibration"
TRACE_NAMES = {"Merkez": "centre", "Ortalama": "average", "Akşam yoğun": "evening"}


def main() -> None:
    cells = load_cells(TRAIN_DAYS)
    last_slot = MILAN_NORMAL_END_DAY * HORIZON_SLOTS
    traces = {
        TRACE_NAMES[name]: [round(float(value), 4) for value in series[:last_slot]]
        for name, series in cells.items()
    }
    cell = json.loads((CALIBRATION / "cell.json").read_text())
    broadcast = json.loads((CALIBRATION / "broadcast.json").read_text())
    data = {
        "capacityMbps": CALIBRATED_CAPACITY_MBPS,
        "sinrDbPercentiles": cell["sinr_db_percentiles"],
        "snrDb": broadcast["snr_db"],
        "messageErrorRate": broadcast["message_error_rate"],
        "traces": traces,
    }
    text = "globalThis.NWL_DATA = " + json.dumps(data, separators=(",", ":")) + ";\n"
    (HERE / "data.js").write_text(text)


if __name__ == "__main__":
    main()
