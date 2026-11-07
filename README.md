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

Komut özet değerleri yazdırır ve günlük yük grafiğini `results/human_load.png` dosyasına kaydeder.

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
