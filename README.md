# Network Weather Service

Yapay zekâ ajanları için yük tahmini katmanı. CSE476/575 dönem projesi.

Baz istasyonu kendi yükünü tahmin edip hücredeki ajanlara yayınlar. Ajanlar ertelenebilir işlerini sessiz saatlere kaydırır, böylece insanların gördüğü yoğun saat tepesi düşer. Bu depo, fikri tek bir hücrenin Python simülasyonuyla sınar.

## Kurulum

Python 3.14 ile test edildi. Depo klasöründe şu komutları çalıştırın:

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python -m nwl
```

Komut senaryoların sonuç tablosunu yazdırır ve grafikleri `results/` klasörüne kaydeder.

Tahmin modellerini karşılaştırmak için:

```bash
python -m nwl.evaluate_forecasts
```

Bu komut Milano verisini `data/` altında bekler (aşağıya bakın).

## Veri

Gerçek veri olarak Milano'nun Telecom Italia kayıtlarının (1 Kasım 2013 - 1 Ocak 2014) CESNET TS-Zoo dönüşümü kullanılır. `TelecomItalia-telecommunications-service_provider-10_minutes.h5` dosyasını [Zenodo](https://doi.org/10.5281/zenodo.22721975) üzerinden indirip `data/tszoo/databases/TelecomItalia/` klasörüne koyun. Dosya 9,3 GB'tır.

Veri kaynağı: BigDataChallenge yarışması. Barlacchi vd., "A multi-source dataset of urban life in the city of Milan and the Province of Trentino", *Scientific Data* 2:150055 (2015), [doi:10.1038/sdata.2015.55](https://doi.org/10.1038/sdata.2015.55). Veri Open Database License (ODbL 1.0) ile dağıtılır.

Veri değerleri ölçeklidir ve kareler baz istasyonu değildir. Bu projede 5×5 karelik bloklar (yaklaşık 1,2 km) tek hücre sayılır ve yük ölçeği varsayımdır.

## Testler

```bash
python -m pytest
```

## Klasörler

| Klasör | İçerik |
|---|---|
| `nwl/` | Simülasyon kodu |
| `tests/` | Otomatik testler |
| `data/` | Ham veri, depoya girmez |
| `results/` | Üretilen grafik ve tablolar, depoya girmez |
