"""Simülasyonu tek bir HTML dosyasında birleştirir.

Çalıştırmak için: python -m simulation.build
`dist/network_weather.html` çift tıklayarak açılabilir, sunucu gerekmez.
`dist/network_weather_page.html` ise yayın için başlıksız halidir.
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"


def main() -> None:
    page = (HERE / "page.html").read_text()
    for marker, source in (("DATA", "data.js"), ("ENGINE", "engine.js"), ("APP", "app.js")):
        page = page.replace(f"/*{marker}*/", (HERE / source).read_text())

    DIST.mkdir(exist_ok=True)
    (DIST / "network_weather_page.html").write_text(page)
    head = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    head += '<meta name="viewport" content="width=device-width, initial-scale=1">\n</head>\n<body>\n'
    (DIST / "network_weather.html").write_text(head + page + "\n</body>\n</html>\n")


if __name__ == "__main__":
    main()
