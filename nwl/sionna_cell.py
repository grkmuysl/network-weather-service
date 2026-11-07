"""Hücre kapasitesini ve kullanıcı SINR dağılımını Sionna ile hesaplar.

Çalıştırmak için: python -m nwl.sionna_cell
Sonuç `calibration/cell.json` dosyasına yazılır. Simülatör yalnızca bu dosyayı okur,
yani Sionna'yı bir kez çalıştırmak yeterlidir.

Adımlar: 3GPP UMa hücreleri kurulur, her kullanıcının ortalama SINR'ı bulunur, sonra
oransal adil zamanlayıcı ve bağlantı uyarlamasıyla slot slot taşınan veri ölçülür.
Hızlı sönümleme kaynak bloğu başına bağımsız Rayleigh olarak eklenir. MIMO yoktur,
bu yüzden sonuç temkinli bir kapasitedir.
"""

import json
from pathlib import Path

import numpy as np
import torch
from sionna.phy import config
from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies
from sionna.phy.channel.tr38901 import UMa, PanelArray
from sionna.sys import (
    EESM,
    InnerLoopLinkAdaptation,
    PFSchedulerSUMIMO,
    PHYAbstraction,
    gen_hexgrid_topology,
)

CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "calibration" / "cell.json"
SEED = 0

CARRIER_HZ = 3.5e9
SUBCARRIER_SPACING_HZ = 30e3
RESOURCE_BLOCKS = 273  # 100 MHz, n78 bandının yaygın genişliği
BANDWIDTH_HZ = RESOURCE_BLOCKS * 12 * SUBCARRIER_SPACING_HZ
SLOT_SECONDS = 0.5e-3
DOWNLINK_SHARE = 0.75  # TDD deseninin aşağı yön payı
RE_PER_BLOCK = 12 * 12  # 2 sembol kontrol ve referans sinyali için ayrılır

TX_POWER_DBM = 49.0
NOISE_FIGURE_DB = 7.0
THERMAL_NOISE_DBM_PER_HZ = -174.0

DROPS = 8
RINGS = 1
USERS_PER_SECTOR = 10
SLOTS = 400
ACTIVE_USERS = (1, 5, 10, 20)
CELLS_IN_BATCH = 32
MIN_SERVED_SINR_DB = -5.0  # en düşük MCS bunun altında çözülemez
MCS_REFINEMENTS = 2  # etkin SINR seçilen MCS'e bağlı olduğu için tekrar hesaplanır


def user_sinr_db() -> np.ndarray:
    """Her kullanıcının tüm komşu hücreler dolu iken ortalama SINR'ı (dB)."""
    array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="38.901",
        carrier_frequency=CARRIER_HZ,
    )
    user_array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=CARRIER_HZ,
    )
    channel = UMa(
        carrier_frequency=CARRIER_HZ,
        o2i_model="low",
        ut_array=user_array,
        bs_array=array,
        direction="downlink",
    )
    channel.set_topology(*gen_hexgrid_topology(DROPS, RINGS, USERS_PER_SECTOR, "uma"))
    a, tau = channel(1, 1.0)
    frequencies = subcarrier_frequencies(12, SUBCARRIER_SPACING_HZ * 12)
    h = cir_to_ofdm_channel(frequencies, a, tau)
    gain = (h.abs() ** 2).mean(dim=(2, 4, 5, 6)).numpy()  # parti, kullanıcı, baz istasyonu

    noise_dbm = THERMAL_NOISE_DBM_PER_HZ + 10 * np.log10(BANDWIDTH_HZ) + NOISE_FIGURE_DB
    received = 10 ** ((TX_POWER_DBM - noise_dbm) / 10) * gain
    serving = received.max(axis=-1)
    interference = received.sum(axis=-1) - serving
    return 10 * np.log10(serving / (1 + interference)).ravel()


def effective_capacity_sinr(sinr: torch.Tensor, scheduled: torch.Tensor) -> torch.Tensor:
    """Kullanıcının aldığı blokların Shannon hızına denk gelen tek SINR değeri."""
    used = scheduled.sum(dim=(1, 2))
    rate = (torch.log2(1 + sinr) * scheduled).sum(dim=(1, 2)) / used.clamp(min=1)
    return torch.where(used > 0, 2**rate - 1, torch.zeros_like(rate))


def cell_throughput_mbps(sinr_db: np.ndarray, active_users: int, rng: np.random.Generator) -> float:
    """`active_users` kullanıcılı hücrenin ortalama toplam aşağı yön hızı.

    Kapsama dışındaki kullanıcılar sayılmaz. Hiç veri alamadıkları için oransal adil
    zamanlayıcı bütün kaynağı onlara verir ve hücrenin hızı anlamsız biçimde düşer.
    """
    served = sinr_db[sinr_db >= MIN_SERVED_SINR_DB]
    picked = rng.choice(served, (CELLS_IN_BATCH, active_users))
    mean_sinr = torch.tensor(10 ** (picked / 10), dtype=torch.float32)
    shape = (CELLS_IN_BATCH, 1, RESOURCE_BLOCKS, active_users)

    scheduler = PFSchedulerSUMIMO(active_users, RESOURCE_BLOCKS, 1, batch_size=CELLS_IN_BATCH)
    abstraction = PHYAbstraction()
    link_adaptation = InnerLoopLinkAdaptation(abstraction, bler_target=0.1)
    eesm = EESM()

    last_rate = torch.full((CELLS_IN_BATCH, active_users), 1e-3)
    delivered_bits = 0.0
    for _ in range(SLOTS):
        fading = torch.distributions.Exponential(1.0).sample(shape)
        sinr = mean_sinr[:, None, None, :] * fading
        scheduled = scheduler(last_rate, torch.log2(1 + sinr))[..., 0]
        allocated_re = scheduled.sum(dim=(1, 2)).int() * RE_PER_BLOCK

        sinr_used = (sinr * scheduled)[..., None]
        mcs = link_adaptation(
            sinr_eff=effective_capacity_sinr(sinr, scheduled), num_allocated_re=allocated_re
        )
        for _ in range(MCS_REFINEMENTS):
            effective = eesm(sinr_used, mcs)
            mcs = link_adaptation(sinr_eff=effective, num_allocated_re=allocated_re)
        effective = eesm(sinr_used, mcs)
        bits, *_ = abstraction(
            mcs, sinr_eff=effective, num_allocated_re=allocated_re,
            check_mcs_index_validity=False,
        )
        last_rate = bits.float() / SLOT_SECONDS + 1e-3
        delivered_bits += bits.sum().item()

    seconds = SLOTS * SLOT_SECONDS * CELLS_IN_BATCH
    return delivered_bits / seconds * DOWNLINK_SHARE / 1e6


def main() -> None:
    config.seed = SEED
    rng = np.random.default_rng(SEED)
    sinr_db = user_sinr_db()
    capacity = {users: cell_throughput_mbps(sinr_db, users, rng) for users in ACTIVE_USERS}

    percentiles = np.arange(1, 100)
    CALIBRATION_PATH.parent.mkdir(exist_ok=True)
    CALIBRATION_PATH.write_text(
        json.dumps(
            {
                "capacity_mbps_by_active_users": {str(u): round(c, 1) for u, c in capacity.items()},
                "users_out_of_coverage": round(float((sinr_db < MIN_SERVED_SINR_DB).mean()), 3),
                "sinr_db_percentiles": np.percentile(sinr_db, percentiles).round(2).tolist(),
            },
            indent=2,
        )
    )
    print(f"Kullanıcı SINR (dB): %5 {np.percentile(sinr_db, 5):.1f}, "
          f"ortanca {np.median(sinr_db):.1f}, %95 {np.percentile(sinr_db, 95):.1f}")
    for users, value in capacity.items():
        print(f"{users:3d} etkin kullanıcı: {value:6.1f} Mbps")


if __name__ == "__main__":
    main()
