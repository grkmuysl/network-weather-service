"""Tahmin mesajının kaybolma olasılığını Sionna ile ölçer.

Çalıştırmak için: python -m nwl.sionna_broadcast
Önce `python -m nwl.sionna_cell` çalışmış olmalıdır. Sonuç `calibration/broadcast.json`
dosyasına yazılır.

486 baytlık mesaj 5G LDPC ve QPSK ile gönderilir. Aynı mesajın alıcıda birleştirilen
kopyaları kaybı azaltır, ama yayının kapladığı kaynağı aynı oranda artırır.
"""

import json

import numpy as np
import torch
from sionna.phy import config
from sionna.phy.channel import AWGN
from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
from sionna.phy.mapping import BinarySource, Constellation, Demapper, Mapper

from nwl.sionna_cell import CALIBRATION_PATH, SEED

BROADCAST_PATH = CALIBRATION_PATH.with_name("broadcast.json")
MESSAGE_BITS = 486 * 8
CODE_RATE = 1 / 3
BITS_PER_SYMBOL = 2
COPIES = (1, 2, 4)
SNR_DB = np.arange(-14.0, 6.5, 1.0)
BLOCKS_PER_POINT = 400
BATCH = 100
FADING_GROUP_SYMBOLS = 48  # bir kaynak bloğundaki alt taşıyıcılar sönümlemeyi paylaşır


def message_error_rate(snr_db: float, copies: int) -> float:
    """Kopyalar birleştirildikten sonra mesajın çözülemediği oran."""
    coded_bits = int(MESSAGE_BITS / CODE_RATE)
    encoder = LDPC5GEncoder(MESSAGE_BITS, coded_bits)
    decoder = LDPC5GDecoder(encoder, hard_out=True)
    constellation = Constellation("qam", BITS_PER_SYMBOL)
    mapper = Mapper(constellation=constellation)
    demapper = Demapper("app", constellation=constellation)
    source = BinarySource()
    awgn = AWGN()
    noise = torch.tensor(10 ** (-snr_db / 10), dtype=torch.float32)

    errors = 0
    for _ in range(BLOCKS_PER_POINT // BATCH):
        bits = source([BATCH, MESSAGE_BITS])
        symbols = mapper(encoder(bits))
        llr = 0.0
        for _ in range(copies):
            groups = symbols.shape[-1] // FADING_GROUP_SYMBOLS + 1
            real, imag = torch.randn(BATCH, groups), torch.randn(BATCH, groups)
            fading = torch.complex(real, imag) / 2**0.5
            fading = fading.repeat_interleave(FADING_GROUP_SYMBOLS, dim=-1)
            fading = fading[:, : symbols.shape[-1]]
            order = torch.randperm(symbols.shape[-1])
            fading = fading[:, order]  # kodlu bitler frekansa yayılır

            received = awgn(fading * symbols, noise)
            equalized = received / fading
            noise_per_symbol = noise / fading.abs() ** 2
            llr = llr + demapper(equalized, noise_per_symbol)
        errors += (decoder(llr) != bits).any(dim=-1).sum().item()
    return errors / (BLOCKS_PER_POINT // BATCH * BATCH)


def main() -> None:
    config.seed = SEED
    curves = {
        str(copies): [round(message_error_rate(snr, copies), 4) for snr in SNR_DB]
        for copies in COPIES
    }
    with CALIBRATION_PATH.open() as file:
        sinr_db = np.array(json.load(file)["sinr_db_percentiles"])

    loss_by_copies = {
        copies: float(np.interp(sinr_db, SNR_DB, curve).mean()) for copies, curve in curves.items()
    }
    BROADCAST_PATH.write_text(
        json.dumps(
            {"snr_db": SNR_DB.tolist(), "message_error_rate": curves, "mean_loss": loss_by_copies},
            indent=2,
        )
    )
    for copies, loss in loss_by_copies.items():
        print(f"{copies} kopya: ortalama mesaj kaybı %{100 * loss:.1f}")


if __name__ == "__main__":
    main()
