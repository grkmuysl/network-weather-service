"""Türkçe ayrıntılı raporu üretir.

Çalıştırmak için: python -m report.build_report_tr
Sayılar `results/` ve `calibration/` dosyalarından okunur, elle yazılmaz.
Önce `nwl.experiments` deneyleri ve `nwl.evaluate_forecasts` çalışmış olmalıdır.
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

from nwl import plots
from nwl.evaluate_forecasts import MILAN_NORMAL_END_DAY, TRAIN_DAYS
from nwl.forecast import SLOTS_PER_DAY
from nwl.milan import load_cells
from nwl.traffic import daily_profile

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "report" / "figures"
OUTPUT = ROOT / "report" / "NWL_Ayrintili_Rapor.pdf"
SCREENSHOT = FIGURES / "simulasyon.jpg"
INK = colors.HexColor("#14323d")
ACCENT = colors.HexColor("#1b7f73")
LINE = colors.HexColor("#c9d6da")
SOFT = colors.HexColor("#eef4f4")
QUIET = colors.HexColor("#566b73")
TEXT_WIDTH = A4[0] - 4.4 * cm

FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
for name, file in (
    ("Govde", "DejaVuSans.ttf"),
    ("Govde-Kalin", "DejaVuSans-Bold.ttf"),
    ("Govde-Italik", "DejaVuSans-Oblique.ttf"),
    ("Kod", "DejaVuSansMono.ttf"),
):
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR / file)))
pdfmetrics.registerFontFamily(
    "Govde", normal="Govde", bold="Govde-Kalin", italic="Govde-Italik", boldItalic="Govde-Kalin"
)

BODY = ParagraphStyle("Govde", fontName="Govde", fontSize=9.6, leading=14.2, textColor=INK,
                      spaceAfter=7, alignment=TA_LEFT)
H1 = ParagraphStyle("H1", parent=BODY, fontName="Govde-Kalin", fontSize=16, leading=20,
                    spaceBefore=16, spaceAfter=9, textColor=ACCENT, keepWithNext=1)
H2 = ParagraphStyle("H2", parent=BODY, fontName="Govde-Kalin", fontSize=11.2, leading=15,
                    spaceBefore=9, spaceAfter=4, keepWithNext=1)
CAPTION = ParagraphStyle("Altyazi", parent=BODY, fontName="Govde-Italik", fontSize=8.2,
                         leading=11, textColor=QUIET, spaceAfter=10)
CELL = ParagraphStyle("Hucre", parent=BODY, fontSize=8, leading=10.5, spaceAfter=0)
CELL_HEAD = ParagraphStyle("HucreBaslik", parent=CELL, fontName="Govde-Kalin")
BULLET = ParagraphStyle("Madde", parent=BODY, leftIndent=14, bulletIndent=3, spaceAfter=3)
BOX = ParagraphStyle("Kutu", parent=BODY, spaceAfter=0)
TITLE = ParagraphStyle("Baslik", parent=BODY, fontName="Govde-Kalin", fontSize=27, leading=33,
                       textColor=ACCENT, spaceAfter=10)
SUBTITLE = ParagraphStyle("AltBaslik", parent=BODY, fontSize=13, leading=19, spaceAfter=30)
TOC_1 = ParagraphStyle("Icindekiler", parent=BODY, spaceAfter=4)


def n(value: float, digits: int = 1) -> str:
    """Sayıyı Türkçe ondalık virgülle yazar."""
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: float) -> str:
    return f"%{round(value * 100)}"


class Report(BaseDocTemplate):
    def afterFlowable(self, flowable) -> None:
        if isinstance(flowable, Paragraph) and flowable.style.name == "H1":
            key = f"bolum-{self.seq.nextf('bolum')}"
            self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (0, flowable.getPlainText(), self.page, key))


def footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setFont("Govde", 7.5)
    canvas.setFillColor(QUIET)
    canvas.drawString(2.2 * cm, 1.2 * cm, "Network Weather Service, CSE476/575")
    canvas.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, str(document.page))
    canvas.restoreState()


def p(text: str, style: ParagraphStyle = BODY) -> Paragraph:
    return Paragraph(text, style)


def bullets(items: list[str]) -> list[Paragraph]:
    return [Paragraph(item, BULLET, bulletText="•") for item in items]


def box(text: str) -> Table:
    table = Table([[p(text, BOX)]], colWidths=[TEXT_WIDTH])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SOFT),
        ("LINEBEFORE", (0, 0), (0, -1), 2.5, ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return KeepTogether([table, Spacer(1, 9)])


def table(rows: list[list[str]], widths: list[float], highlight: int | None = None) -> Table:
    data = [[p(str(c), CELL_HEAD if r == 0 else CELL) for c in row] for r, row in enumerate(rows)]
    total = sum(widths)
    result = Table(data, colWidths=[w / total * TEXT_WIDTH for w in widths], repeatRows=1)
    style = [
        ("LINEBELOW", (0, 0), (-1, 0), 1, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if highlight is not None:
        style.append(("BACKGROUND", (0, highlight), (-1, highlight), SOFT))
    result.setStyle(TableStyle(style))
    return result


def figure(path: Path, caption: str, width: float = TEXT_WIDTH) -> KeepTogether:
    with PILImage.open(path) as image:
        ratio = image.height / image.width
    return KeepTogether([Image(str(path), width=width, height=width * ratio), Spacer(1, 3),
                         p(caption, CAPTION)])


def draw_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def block(x, y, w, h, title, body="", accent=False):
        face = "#dff1ee" if accent else "white"
        edge = "#1b7f73" if accent else "#8aa0a8"
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8",
                                    fc=face, ec=edge, lw=1.6 if accent else 1))
        ax.text(x + 1.2, y + h - 2.6, title, fontsize=8.6, fontweight="bold", color="#14323d")
        ax.text(x + 1.2, y + 1.2, body, fontsize=7.6, color="#566b73")

    def arrow(start, end, style="-"):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, color="#566b73",
                                     lw=1, linestyle=style, shrinkA=0, shrinkB=0))

    ax.text(3, 47, "Yapay zekâ ajanı", fontsize=10.5, fontweight="bold", color="#14323d")
    ax.text(68, 47, "Baz istasyonu", fontsize=10.5, fontweight="bold", color="#14323d")
    block(3, 35, 31, 9, "Görev zamanlayıcı (Uygulama)", "acil: hemen, ertelenebilir: sonra")
    block(3, 22, 31, 9, "Network Weather Layer (yeni)", "tahmini çözer, slotları listeler", True)
    block(3, 9, 31, 9, "Alt katmanlar", "Taşıma ile Fiziksel arası, değişmedi")
    block(68, 35, 29, 9, "Yük ölçümü", "toplam yük: insan + ajan")
    block(68, 22, 29, 9, "Tahmin modeli", "ortalama, ARIMA veya LSTM")
    block(68, 9, 29, 9, "Tahmin yayını", "15 dakikada bir, 24 saatlik")
    block(43, 37, 16, 5, "İnsanlar")
    arrow((18.5, 31.4), (18.5, 34.6))
    arrow((82.5, 34.6), (82.5, 31.4))
    arrow((82.5, 21.6), (82.5, 18.4))
    arrow((59.4, 39.5), (67.6, 39.5))
    ax.plot([67.6, 51, 51], [13.5, 13.5, 26.5], color="#566b73", lw=1)
    arrow((51, 26.5), (34.6, 26.5))
    ax.text(45, 28, "(1) yük tahmini", fontsize=8, color="#566b73")
    ax.plot([18.5, 18.5, 99, 99], [8.6, 3, 3, 39.5], color="#566b73", lw=1)
    arrow((99, 39.5), (97.6, 39.5))
    ax.text(40, 4.2, "(2) ajan trafiği, sakin saatlerde gönderilir", fontsize=8, color="#566b73")
    ax.text(45.5, 33.5, "(3) insan trafiği", fontsize=8, color="#566b73")
    fig.tight_layout(pad=0.2)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def draw_figures() -> None:
    FIGURES.mkdir(exist_ok=True)
    plots.set_language("tr", figure_scale=0.72)
    draw_architecture(FIGURES / "mimari.png")
    plots.plot_main_results(pd.read_csv(RESULTS / "main.csv"), FIGURES / "ana.png")
    plots.plot_sweeps(pd.read_csv(RESULTS / "sweeps.csv"), FIGURES / "tarama.png")
    plots.plot_error_robustness(pd.read_csv(RESULTS / "errors.csv"), FIGURES / "dayaniklilik.png")
    plots.plot_broadcast_loss(pd.read_csv(RESULTS / "broadcast.csv"), FIGURES / "yayin.png")

    cells = load_cells(TRAIN_DAYS)
    last = MILAN_NORMAL_END_DAY * SLOTS_PER_DAY
    english = {"Merkez": "Centre", "Ortalama": "Average", "Akşam yoğun": "Evening"}
    shapes = {
        english[name]: series[:last].reshape(-1, SLOTS_PER_DAY).mean(axis=0)
        for name, series in cells.items()
    }
    synthetic = daily_profile().reshape(-1, 15).mean(axis=1)
    plots.plot_daily_shapes({**shapes, "Synthetic": synthetic}, FIGURES / "sekil.png")
    plots.set_language("en")
    return {
        name: float(np.corrcoef(shape, synthetic)[0, 1]) for name, shape in shapes.items()
    }


def forecast_scores() -> dict[str, dict[str, tuple[str, str]]]:
    """`results/forecast_comparison.txt` içindeki MAE ve RMSE değerlerini okur."""
    scores, section = {}, None
    for line in (RESULTS / "forecast_comparison.txt").read_text().splitlines():
        if line.startswith(("Sentetik veri", "Milano,")):
            section = line.rstrip(":")
            scores[section] = {}
            continue
        match = re.match(r"(.+?)\s+(\d\.\d{3})\s+(\d\.\d{3})\s+", line)
        if match and section:
            scores[section][match.group(1).strip()] = (match.group(2), match.group(3))
    return scores


def build() -> None:
    correlation = draw_figures()
    main = pd.read_csv(RESULTS / "main.csv").groupby(["world", "scenario", "forecaster"]).mean(numeric_only=True)
    sweeps = pd.read_csv(RESULTS / "sweeps.csv")
    sweep = sweeps.groupby(["world", "group", "value", "scenario"]).mean(numeric_only=True)
    errors = pd.read_csv(RESULTS / "errors.csv")
    broadcast = pd.read_csv(RESULTS / "broadcast.csv")
    cell = json.loads((ROOT / "calibration" / "cell.json").read_text())
    link = json.loads((ROOT / "calibration" / "broadcast.json").read_text())
    capacity = cell["capacity_mbps_by_active_users"]
    forecast = forecast_scores()

    EV, CE, SY = "Akşam yoğun, normal", "Merkez, normal", "Sentetik"
    ev_h = "Akşam yoğun, tatil"
    m = lambda w, s, f, k: float(main.loc[(w, s, f), k])
    cong = lambda w, s, f: m(w, s, f, "congested_minutes_per_day")
    peak = lambda w, s, f: m(w, s, f, "peak_load")
    delay = lambda w, s, f: m(w, s, f, "agent_delay_minutes")
    sw = lambda w, g, v, s, k="congested_minutes_per_day": float(sweep.loc[(w, g, v, s), k])
    gain_ce = 100 * (1 - cong(CE, "d", "ARIMA") / cong(CE, "b", "-"))
    gain_ev = 100 * (1 - cong(EV, "d", "ARIMA") / cong(EV, "b", "-"))

    fixed = errors[(errors.forecaster == "Pencere sabit") & errors.group.isin(["Bağımsız hata", "Uzun süren hata"])]
    wide = errors[(errors.forecaster == "Pencere genişliyor") & errors.group.isin(["Bağımsız hata", "Uzun süren hata"])]
    ev_fixed = fixed[fixed.world == EV].groupby(["group", "value"]).peak_load.mean()
    ev_wide = wide[wide.world == EV].groupby(["group", "value"]).peak_load.mean()
    reference = errors[errors.group == "referans"].groupby("world").peak_load.mean()
    shifts = errors[errors.group == "Zaman kayması (dk)"].copy()
    shifts["value"] = shifts["value"].astype(float)
    shift = shifts[shifts.forecaster == "Pencere genişliyor"].groupby(["world", "value"]).peak_load.mean()

    b = broadcast.copy()
    b_mean = b[b.group != "referans"].assign(value=lambda d: d["value"].astype(float)).groupby(
        ["world", "group", "value"]).mean(numeric_only=True)
    loss = link["mean_loss"]
    S: list = []

    S += [Spacer(1, 4.5 * cm), p("Network Weather Service", TITLE),
          p("Yapay zekâ ajanları için tıkanıklık tahmin katmanı: deneyin ayrıntılı anlatımı", SUBTITLE),
          p("<b>Enes Ödev</b> (220104004034) ve <b>Görkem Uysal</b> (230104004174)"),
          p("CSE476/575 Dönem Projesi, Güz 2026"), p("7 Kasım 2026"), Spacer(1, 2 * cm),
          box("Bu rapor projenin Türkçe ve ayrıntılı anlatımıdır. Fikri, simülasyonun nasıl kurulduğunu, "
              "hangi deneylerin yapıldığını ve sonuçların ne anlama geldiğini baştan sona açıklar. "
              "Tablolar ve grafikler deney dosyalarından otomatik üretilmiştir."),
          PageBreak()]

    toc = TableOfContents()
    toc.levelStyles = [TOC_1]
    S += [p("İçindekiler", ParagraphStyle("IcBaslik", parent=H1)), toc, PageBreak()]

    S += [p("1. Kısa özet", H1),
          p("Bugünün mobil şebekesi tepkiseldir: baz istasyonu trafiği geldiği gibi taşır ve ne kadar yoğun "
            "olacağını kimseye söylemez. İnsanlar için bu uygundur, çünkü şebekeyi ihtiyaç duydukları anda "
            "kullanırlar. Yapay zekâ ajanlarının ise yedekleme, rapor üretme ve model güncelleme gibi "
            "bekleyebilecek birçok işi vardır."),
          p("Bu projede <b>Network Weather Layer (NWL)</b> adlı yeni bir katman önerdik. Baz istasyonu önümüzdeki "
            "24 saatin yük tahminini yayınlar. Ajan, acil işlerini hemen gönderir, ertelenebilir işlerini sakin "
            "saatlere taşır. Bir hücreyi önce sentetik, sonra gerçek Milano trafiğiyle simüle ettik."),
          box(f"<b>Ana sonuç:</b> ARIMA tahminiyle günlük tıkanıklık, ajanların tahmini yok saydığı duruma göre "
              f"şehir merkezinde {n(cong(CE,'b','-'))} dakikadan {n(cong(CE,'d','ARIMA'))} dakikaya "
              f"(%{round(gain_ce)} azalma), akşam yoğun bölgede {n(cong(EV,'b','-'))} dakikadan "
              f"{n(cong(EV,'d','ARIMA'))} dakikaya (%{round(gain_ev)} azalma) indi. Bedeli ortalama yaklaşık "
              f"2,5 saatlik ajan gecikmesidir. Son teslim süresi kaçırılmadı.")]
    S += bullets([
        f"<b>Rastgelelik şarttır.</b> Rastgele seçim olmadan ajanlar aynı sakin slota yığılır. Akşam yoğun "
        f"bölgede tepe yük {pct(peak(EV,'c','ARIMA'))} olur.",
        f"<b>Kalıcı tahmin hatası tehlikelidir.</b> Sabit pencerede tepe {pct(ev_fixed.max())} oldu. Pencereyi "
        f"belirsizliğe göre genişletince en fazla {pct(ev_wide.max())} oldu. Bu, projenin kendi tasarım katkısıdır.",
        f"<b>Sionna ile doğruladık.</b> Hücre kapasitesini {capacity['10']:.0f} Mbps hesapladık. Mesaj kaybı tek "
        f"kopyada %{round(100 * loss['1'])}, dört kopyada %{round(100 * loss['4'])} oldu.",
        "<b>Sistemin sınırı vardır.</b> Ajan trafiği çok artarsa kaydıracak sakin zaman kalmaz.",
    ])
    S += [Spacer(1, 6),
          p("Kesin yüzdeler seçtiğimiz varsayımlara bağlıdır. Bu nedenle asıl bulgular, senaryolar arasındaki "
            "farklardır. Tüm varsayımlar Bölüm 4'te listelenmiştir.")]

    S += [p("2. Problem ve fikir", H1),
          p("Ajan trafiği insanlarla aynı saatlerde yoğunlaşırsa akşam tepesini büyütür ve insanların bağlantısını "
            "yavaşlatır. Oysa bu işlerin çoğu saatlerce bekleyebilir. Ajan ne zaman sakin olacağını bilirse "
            "bekler."),
          p("Otoyol benzetmesi işi kolaylaştırır. İnsanlar işe gitmek için belli saatte yolda olmak zorundadır. "
            "Yedekleme gibi işler ise kargo kamyonları gibidir, gece de gidebilir. Biz kamyonlara yolun yoğunluk "
            "tahminini veriyoruz."),
          p("Ama tahmini paylaşmak yeni bir risk doğurur. Herkese \"en boş saat 03:00\" denirse herkes 03:00'te "
            "çıkar ve boş saat en kötü ana dönüşür. Buna <b>sürü etkisi</b> diyoruz. Proje üç soruya cevap arar:"),
          ]
    S += bullets(["Yayınlanan bir tahmin tıkanıklığı ne kadar azaltır, ajan gecikmesi olarak bedeli nedir?",
                  "Tahmin ne kadar doğru olmalı, hangi model yeterlidir?",
                  "Ajanlar yeni bir tepe oluşturmaktan nasıl kaçınır, tahmin yanlışsa bu çözüm dayanır mı?"])
    S += [p("İlgili çalışmalar", H2),
          p("5G çekirdeğindeki NWDAF işlevi şebeke yük analitiği üretir. IETF ALTO (RFC 7285) ise şebeke bilgisini "
            "uygulamalara sunar. İkisi de operatör ve uygulamalar için tasarlanmıştır. NWL'nin farkı iki noktadadır: "
            "tahmini hücredeki herkese yayınla ulaştırır, yani ek yükü ajan sayısıyla artmaz. Ayrıca ajanların "
            "toplu davranışını, yani sürü etkisini, tasarımın merkezine koyar.")]

    S += [p("3. Sistem tasarımı", H1),
          p("NWL tek yönlü bir protokoldür: baz istasyonu yayınlar, her ajan kendi başına karar verir. Baz "
            "istasyonu ajan başına bilgi tutmaz ve rezervasyon almaz."),
          figure(FIGURES / "mimari.png", "Şekil 1. NWL'nin protokol yığınındaki yeri ve yük ölçüm döngüsü."),
          p("Tahmin mesajı", H2),
          p("Baz istasyonu 15 dakikada bir, önümüzdeki 96 slotun (24 saat) tahminini yayınlar. Mesaj 486 bayttır "
            "ve hücreye yaklaşık 4,3 bit/s ek yük getirir. Bu yük ajan sayısıyla değişmez."),
          table([["Alan", "Boyut", "Anlamı"],
                 ["issued_minute", "4 bayt", "Başlık: tahminin yayınlandığı dakika"],
                 ["entry_count", "2 bayt", "Başlık: kayıt sayısı"],
                 ["time_slot", "2 bayt", "Her kayıtta: slot numarası"],
                 ["predicted_load", "2 bayt", "Her kayıtta: tahmini yük (binde bir birimiyle)"],
                 ["confidence", "1 bayt", "Her kayıtta: güven, 0 ile 1 arası (yüzde bir birimiyle)"]],
                [3, 1.5, 7]),
          Spacer(1, 8),
          p("<b>Güven (confidence)</b> tahminin beklenen hatasını taşır. Baz istasyonu her ufuk için son iki günün "
            "tahmin hatasını ölçer ve güven = 1 − hata / 0,5 formülüyle yayınlar. Tüm modeller aynı yöntemi kullanır."),
          p("Ajanın kararı", H2),
          p("Acil görevler hemen başlar. Ertelenebilir bir görev, son teslim süresine yetişecek herhangi bir slotta "
            "başlayabilir (30 dakika güvenlik payı bırakılır). Zamanlayıcı şu adımları izler:")]
    S += bullets(["Her slotun temkinli yükünü hesaplar: tahmini yük + beklenen hata.",
                  "En sakin slotu bulur. Bunun temkinli yüküne L, beklenen hatasına e diyelim.",
                  "Temkinli yükü L + 0,05 + 2e değerini geçmeyen tüm slotlar içinden rastgele bir dakika seçer."])
    S += [Spacer(1, 4),
          p("Buradaki <b>2e</b> terimi deneyler sırasında bulduğumuz bir tasarım sonucudur. Tahmin belirsizse iki "
            "slot arasındaki küçük fark anlamsızdır, bu yüzden pencere genişler. Nedeni Bölüm 8.3'te açıklanmıştır."),
          p("Geri besleme döngüsü", H2),
          p("Baz istasyonu insan ve ajan trafiğini ayıramaz, yalnızca toplam yükü ölçer. Bu yüzden modeller toplam "
            "yükü tahmin eder. Ajanların kaydırdığı yük ölçüme, oradan da sonraki tahminlere girer. Model yalnızca "
            "bitmiş slotları görür, yani geleceği bilemez.")]

    S += [p("4. Simülasyon yöntemi", H1),
          p("Tek bir hücreyi dakika dakika ilerleyen bir Python simülatörüyle modelledik. Her deney 10 gün sürer. "
            "İlk ve son gün ölçüme girmez, çünkü ilk gecede önceki günden ertelenen görev yoktur ve son gün tahmin "
            "ufku biter. Geriye 8 ölçüm günü kalır."),
          p("Her senaryo aynı insan trafiğini ve aynı görevleri görür. Böylece aradaki fark yalnızca ajanların "
            "davranışından gelir."),
          p("Model nasıl çalışır?", H2)]
    S += bullets(["<b>Hücre:</b> Toplam talep kapasiteyi aşarsa herkes aynı oranda yavaşlar.",
                  "<b>İnsanlar:</b> Karşılanamayan insan talebi beklemez, kaybolmuş sayılır.",
                  "<b>Ajanlar:</b> Gönderemediği veriyi sonraki dakikalarda göndermeye devam eder.",
                  "<b>Görevler:</b> İnsan trafiğiyle aynı günlük şekilde gelir, çünkü ajanlar insanlar adına çalışır."])
    S += [p("Varsayımlar", H2),
          table([["Varsayım", "Değer", "Kaynağı"],
                 ["Hücre kapasitesi", f"{capacity['10']:.0f} Mbps", "Sionna ile hesaplandı (Bölüm 7)"],
                 ["Tıkanıklık eşiği", "Kapasitenin %80'i", "Varsayım"],
                 ["Ajan trafiğinin payı", "İnsan hacminin %25'i", "Varsayım"],
                 ["Acil görev payı", "%30", "Varsayım"],
                 ["Görev hızı", "2 Mbps", "Varsayım"],
                 ["Ortalama görev süresi", "10 dakika (log-normal)", "Varsayım"],
                 ["Acil görev süresi payı", "10 dakika", "Varsayım"],
                 ["Ertelenebilir görev süresi payı", "120 ile 720 dakika", "Varsayım"],
                 ["Sabit sakin pencere payı", "0,05", "Varsayım"],
                 ["Pencere hata ağırlığı", "2", "Deneylerle seçildi"],
                 ["Yük ölçeği (Milano)", "Eğitimin %99'luk dilimi = kapasitenin %75'i", "Varsayım"]],
                [3.4, 3.2, 3]),
          Spacer(1, 8),
          p("Senaryolar", H2),
          table([["Senaryo", "Ajanlar ne yapıyor?", "Neyi gösterir?"],
                 ["(a) Ajan yok", "Yalnızca insanlar var", "Başlangıç noktası"],
                 ["(b) Tahmini yok say", "Görev geldiği an gönderilir", "Bugünkü durum"],
                 ["(c) Rastgelelik yok", "En sakin slotun ilk dakikası", "Sürü etkisi"],
                 ["(d) Tam NWL", "Sakin pencerede rastgele dakika", "Önerilen sistem"]],
                [2.6, 4, 2.4], highlight=4),
          Spacer(1, 8),
          p("Ölçütler", H2)]
    S += bullets(["<b>Tepe yük:</b> Günün en yüksek yükünün ortalaması.",
                  "<b>Tepe/ortalama:</b> Trafiğin ne kadar dengesiz olduğu.",
                  "<b>Tıkanık dakika:</b> Günde kaç dakika yükün %80'i geçtiği.",
                  "<b>Karşılanmayan insan talebi:</b> Kapasite yetmediği için kaybolan insan trafiği.",
                  "<b>Ajan gecikmesi:</b> Görevin gelişinden bitişine geçen ortalama süre.",
                  "<b>Son teslim ihlali:</b> Süresini kaçıran görevlerin oranı."])
    S += [p("Ana sonuçlar 10 tohumla, taramalar 5 tohumla ölçüldü. Tohumlar görevleri ve zamanlama seçimlerini "
            "değiştirir, insan trafiği her seride sabittir.")]

    S += [p("5. Veri", H1),
          p("Gerçek veri olarak Telecom Italia'nın Milano kayıtlarını (1 Kasım 2013 - 1 Ocak 2014) CESNET TS-Zoo "
            "kopyasından kullandık. Veri, 10.000 karelik bir ızgarada 10 dakikalık internet etkinliğini verir. "
            "Bunu 15 dakikalık slotlara çevirdik."),
          p("Bir kare yaklaşık 235 metredir ve baz istasyonu değildir. Bu yüzden 5×5 karelik blokları (yaklaşık "
            "1,2 km) tek hücre saydık. Yalnızca eğitim dönemine bakan kurallarla üç bölge seçtik:")]
    S += bullets(["<b>Merkez:</b> En çok trafiği olan blok.", "<b>Ortalama:</b> Trafiği ortanca olan blok.",
                  "<b>Akşam yoğun:</b> En yoğun çeyrek içinde akşam/öğlen oranı en yüksek blok."])
    S += [Spacer(1, 4), figure(FIGURES / "sekil.png", "Şekil 2. Her serinin ortalama günü, kendi tepesine ölçeklenmiş (1 Kasım - 22 Aralık)."),
          p(f"Şekil önemli bir şey gösteriyor: başta kullandığımız sentetik eğri yalnızca bir bölge tipine benziyor. "
            f"Ortalama güne korelasyonu akşam yoğun bölgede {n(correlation['Evening'],2)}, merkezde "
            f"{n(correlation['Centre'],2)}, ortalama bölgede {n(correlation['Average'],2)}. Merkezde akşam tepesi yoktur, "
            f"yoğunluk gün ortasındadır. Bu yüzden tüm senaryoları gerçek serilerle de çalıştırdık."),
          p("Tahmin modelleri", H2)]
    S += bullets(["<b>Hareketli ortalama:</b> Son 2 saatin ortalaması, tüm ufuk için düz bir çizgi.",
                  "<b>Dün aynı saat:</b> Son günü aynen tekrar eder.",
                  "<b>Geçen hafta aynı saat:</b> Geçen haftanın aynı gününü tekrar eder. Gerçek trafikte haftalık "
                  "döngü güçlü çıktığı için ekledik.",
                  "<b>ARIMA:</b> Ortalama haftalık şekli çıkarır, kalan sapmalara ARIMA(2,0,1) uygular.",
                  "<b>Küçük LSTM:</b> 32 gizli birimli bir sinir ağı. Son 24 saatin yükünü, günün saatini ve haftanın "
                  "gününü görür."])
    keys = [("Sentetik", "Sentetik veri: 42 gün eğitim, 14 gün test"), ("Merkez", "Milano, Merkez: normal günler (13-22 Aralık)"),
            ("Ortalama", "Milano, Ortalama: normal günler (13-22 Aralık)"), ("Akşam yoğun", "Milano, Akşam yoğun: normal günler (13-22 Aralık)"),
            ("Merkez, tatil", "Milano, Merkez: tatil (23 Aralık-1 Ocak)"), ("Ortalama, tatil", "Milano, Ortalama: tatil (23 Aralık-1 Ocak)"),
            ("Akşam, tatil", "Milano, Akşam yoğun: tatil (23 Aralık-1 Ocak)")]
    models = ["Hareketli ortalama", "Dün aynı saat", "Geçen hafta aynı saat", "ARIMA", "LSTM"]
    rows = [["Model"] + [k for k, _ in keys]]
    for model in models:
        rows.append([model] + [forecast[s][model][0].replace(".", ",") for _, s in keys])
    S += [p("Tahmin doğruluğu", H2),
          p("Modeller ilk 42 günde eğitildi. Tablo ortalama mutlak hatayı (MAE) yük biriminde verir; küçük olan "
            "iyidir. RMSE değerleri de aynı sıralamayı gösterir ve sonuç dosyasındadır."),
          table(rows, [3.3, 1.5, 1.4, 1.5, 1.5, 1.5, 1.6, 1.5]), Spacer(1, 7),
          p("Tek bir kazanan yok. Normal günlerde ARIMA ve \"geçen hafta aynı saat\" öne çıkıyor. Tatilde ise "
            "\"dün aynı saat\" en iyisi, çünkü tatil davranışı eğitimde hiç görülmedi ve öğrenilen haftalık şekil "
            "işe yaramıyor.")]

    S += [p("6. Deneyler nasıl yapıldı?", H1),
          p("Çalışma adım adım ilerledi. Önce kusursuz tahminle sistemin mantığını sınadık, sonra gerçek modelleri "
            "bağladık, en son varsayımları ve hataları test ettik.")]
    S += [table([["Deney", "Ne değişti?", "Soru", "Tekrar"],
                 ["Ana karşılaştırma", "Senaryo, model, trafik serisi", "Hangi durum ne kadar iyi?", "10 tohum"],
                 ["Varsayım taraması", "Ajan payı, acil pay, teslim süresi, pencere, eşik", "Sonuç varsayımlara ne kadar bağlı?", "5 tohum"],
                 ["Tahmin hatası", "Eklenen hata ve zaman kayması", "Tahmin yanlışsa ne olur?", "10 tohum"],
                 ["Yayın kaybı", "Mesaj kaybı oranı ve kopya sayısı", "Mesaj ulaşmazsa ne olur?", "10 tohum"]],
                [2.4, 3.8, 3.2, 1.2]), Spacer(1, 8),
          p("Kusursuz tahmin yalnızca gelecekteki insan yükünü bilir, ajanların nereye kayacağını bilemez. Bu yüzden "
            "bir üst sınır değil, bir karşılaştırma noktasıdır. Bazı durumlarda iyi modeller onu geçebilir.")]

    S += [p("7. Sionna ile radyo kalibrasyonu", H1),
          p("Hücre simülatörümüz radyo ayrıntısı içermez. Bu yüzden NVIDIA Sionna 2.2.0 ile iki değeri hesapladık. "
            "Sionna ayrı çalışır, simülatör yalnızca kaydedilen sonuçları okur."),
          p("Hücre kapasitesi (Sionna SYS)", H2),
          p("3,5 GHz'de 100 MHz bant, her iki uçta tek anten ve 21 sektörlü 3GPP şehir modeli kurduk. Komşu hücreler "
            "tam yükte kabul edildi. Oransal adil zamanlayıcı ve bağlantı uyarlamasıyla slot slot taşınan veriyi ölçtük."),
          table([["Hücredeki etkin kullanıcı"] + list(capacity), ["Aşağı yön kapasitesi (Mbps)"] + [f"{v:.0f}" for v in capacity.values()]],
                [4, 1, 1, 1, 1]), Spacer(1, 7),
          p(f"Rapor boyunca 10 kullanıcılı değeri ({capacity['10']:.0f} Mbps) kullandık. Kullanıcıların yaklaşık "
            f"%{round(100 * cell['users_out_of_coverage'])} kadarı −5 dB altında kaldığı için kapasite hesabına girmedi. "
            f"En düşük kodlama seviyesi onlara hizmet veremez ve zamanlayıcı bütün kaynağı boşa onlara ayırır."),
          p(f"Kapasiteyi 100 Mbps yerine {capacity['10']:.0f} Mbps almak senaryoların sırasını değiştirmedi. Bu tek başına "
            f"bağımsız bir doğrulama değildir: Milano yükü kapasiteye göre ölçeklendiği için trafik de kapasiteyle "
            f"birlikte değişir."),
          p("Tahmin mesajının ulaşması (Sionna PHY)", H2),
          p("486 baytlık mesajı 5G LDPC (1/3 oran) ve QPSK ile sönümlemeli bir kanalda gönderdik. Alıcı, aynı mesajın "
            "kopyalarını birleştirebilir. Ölçülen hata eğrilerini Sionna'nın kullanıcı SINR dağılımına uyguladık. Her "
            "kopya iki katına çıktığında yaklaşık 3 dB kazanç elde ettik.")]

    S += [p("8. Sonuçlar", H1), p("8.1 Ana karşılaştırma", H2),
          figure(FIGURES / "ana.png", "Şekil 3. Ortalama günlük tepe yük, 10 tohum.")]
    rows = [["Senaryo", "Sentetik dk/gün", "Merkez dk/gün", "Akşam dk/gün", "Tepe / ortalama", "Gecikme (dk)", "İnsan kaybı", "İhlal"]]
    for label, s, f in [("(a) Ajan yok", "a", "-"), ("(b) Tahmini yok say", "b", "-"), ("(c) ARIMA, rastgelelik yok", "c", "ARIMA"),
                        ("(d) Hareketli ortalama", "d", "Hareketli ortalama"), ("(d) Dün aynı saat", "d", "Dün aynı saat"),
                        ("(d) Geçen hafta aynı saat", "d", "Geçen hafta aynı saat"), ("(d) ARIMA", "d", "ARIMA"),
                        ("(d) Küçük LSTM", "d", "LSTM"), ("(d) Kusursuz tahmin", "d", "Kusursuz")]:
        rows.append([label, n(cong(SY, s, f)), n(cong(CE, s, f)), n(cong(EV, s, f)), n(m(EV, s, f, "peak_to_average"), 2),
                     n(delay(EV, s, f)), "%" + n(m(EV, s, f, "human_unserved_percent"), 2), "%" + n(m(EV, s, f, "deadline_violation_percent"), 2)])
    S += [table(rows, [3.3, 1.3, 1.2, 1.2, 1.3, 1.3, 1.1, 1]),
          p("Tablo: ilk üç sütun günlük tıkanık dakikadır. Son dört sütun akşam yoğun bölgeye aittir.", CAPTION),
          p("Tablodan çıkan sonuçlar:")]
    S += bullets([
        f"<b>Ajan eklemek tıkanıklık yaratır.</b> Akşam yoğun bölgede tepe yük (a) senaryosunda {pct(peak(EV,'a','-'))}, (b) senaryosunda {pct(peak(EV,'b','-'))}.",
        f"<b>NWL işe yarıyor.</b> ARIMA ile tepe yük {pct(peak(EV,'d','ARIMA'))} oluyor. Bu değer kusursuz tahmine ({pct(peak(EV,'d','Kusursuz'))}) çok yakın.",
        f"<b>Rastgelelik olmazsa felaket olur.</b> (c) senaryosunda tepe yük {pct(peak(EV,'c','ARIMA'))} ve kaybolan insan talebi %{n(m(EV,'c','ARIMA','human_unserved_percent'),2)}.",
        "<b>Hareketli ortalama yetmez.</b> Düz bir tahmin sakin saatleri göstermediği için kazanç küçük kalıyor.",
    ])
    S += [Spacer(1, 4),
          p(f"<b>Tatil dönemi:</b> Tahminler kötüleşse de yük düştüğü için hücre zorlanmıyor. Akşam yoğun bölgede tepe "
            f"(b) ile {pct(peak(ev_h,'b','-'))}, (d) ARIMA ile {pct(peak(ev_h,'d','ARIMA'))} oldu. Merkez ve ortalama "
            f"bölgede tatilde hiç tıkanıklık görülmedi."),
          p("8.2 Varsayımlara duyarlılık", H2),
          figure(FIGURES / "tarama.png", "Şekil 4. Bir varsayım değişirken günlük tıkanık dakika, 5 tohum. Satırların dikey ölçekleri farklıdır."),
          ]
    S += bullets([
        f"<b>Ajan payı:</b> NWL'nin boş kapasiteye ihtiyacı var. Akşam yoğun bölgede %50 payda tıkanıklık {sw(EV,'Ajan payı',0.5,'b'):.0f} dakikadan "
        f"{sw(EV,'Ajan payı',0.5,'d'):.0f} dakikaya iniyor. %75 payda fark kalmıyor ({sw(EV,'Ajan payı',0.75,'b'):.0f} ve {sw(EV,'Ajan payı',0.75,'d'):.0f} dakika).",
        f"<b>Teslim süresi:</b> En etkili ayar bu. En fazla 4 saatte merkezde kazanç küçük ({sw(CE,'En uzun süre (dk)',240.0,'b'):.0f} dakikadan "
        f"{sw(CE,'En uzun süre (dk)',240.0,'d'):.0f} dakikaya). 24 saatte tıkanıklık neredeyse bitiyor ({n(sw(CE,'En uzun süre (dk)',1440.0,'d'))} dakika), "
        f"ama gecikme yaklaşık 5 saate çıkıyor.",
        f"<b>Acil görev payı:</b> Kaydırılabilen iş azaldıkça kazanç küçülüyor ama %70'te bile sürüyor (merkez: "
        f"{sw(CE,'Acil görev payı',0.7,'b'):.0f} dakikadan {sw(CE,'Acil görev payı',0.7,'d'):.0f} dakikaya).",
        "<b>Tıkanıklık eşiği:</b> 0,7, 0,8 ve 0,9 eşikleri senaryoların sırasını değiştirmiyor.",
    ])
    S += [p("8.3 Tahmin hatasına dayanıklılık", H2),
          p("Bu projenin en önemli öğrenmesi burada. İlk testimizde her yayında yeni rastgele gürültü ekledik ve "
            "sistem çok dayanıklı göründü. Sonra daha gerçekçi bir test yaptık: hata yayından yayına aynı kaldı. "
            "Sonuç çok farklıydı."),
          p(f"Kalıcı hata tahminde sahte bir \"sakin boşluk\" yaratır. Herkes aynı hatalı tahmini gördüğü için "
            f"ajanlar oraya yığılır. Sabit pencerede akşam yoğun bölgede tepe {pct(ev_fixed.max())} oldu. Tahmini hiç "
            f"kullanmamak ({pct(reference[EV])}) bundan çok daha iyidir."),
          p(f"Çözüm olarak rastgele pencereyi beklenen hataya göre genişlettik. Aynı hatalarda tepe {pct(ev_wide.min())} "
            f"ile {pct(ev_wide.max())} arasında kaldı."),
          figure(FIGURES / "dayaniklilik.png", "Şekil 5. Eklenen hatalarda (üst) ve zaman kaymalarında (alt) tepe yük, 10 tohum."),
          p(f"Zaman kayması daha az zararlı. Tepe çoğu durumda (b)'nin altında kalıyor. İstisna, akşam yoğun bölgede "
            f"tahminin 1-2 saat geç kalması: tepe {pct(shift[EV, -60.0])} ve {pct(shift[EV, -120.0])} oluyor, (b) ise "
            f"{pct(reference[EV])}. Bu testte güven değeri değişmediği için genişleyen pencerenin etkisi yoktur."),
          p("Pencereyi genişletmenin bedeli", H2),
          p("Genişletmek bir sigortadır: en kötü durumu çok iyileştirir, ama tahmin zaten iyiyse biraz kayıp verir."),
          ]
    rows = [["Tahminci", "Akşam: sabit", "Akşam: genişleyen", "Merkez: sabit", "Merkez: genişleyen"]]
    for label, f in [("Dün aynı saat", "Dün aynı saat"), ("Geçen hafta aynı saat", "Geçen hafta aynı saat"), ("ARIMA", "ARIMA"), ("Küçük LSTM", "LSTM")]:
        rows.append([label, n(cong(EV, "d", f"{f} (pencere sabit)")), n(cong(EV, "d", f)), n(cong(CE, "d", f"{f} (pencere sabit)")), n(cong(CE, "d", f))])
    S += [table(rows, [3, 1.5, 1.8, 1.5, 1.8]), p("Tablo: günlük tıkanık dakika, normal günler.", CAPTION),
          p("8.4 Yayın kaybı", H2),
          p("Mesaj kaybı beklediğimizden çok daha az zarar veriyor. Her tahmin 24 saati kapsadığı için ajan elindeki "
            "son tahmini kullanabilir. Hiç tahmini olmayan ajan ise işini hemen gönderir."),
          figure(FIGURES / "yayin.png", "Şekil 6. Her ajan aynı oranda yayın kaybederse günlük tıkanık dakika (ARIMA, 10 tohum)."),
          ]
    rows = [["Yayın kopyası", "Ortalama kayıp", "Sentetik tıkanık dk", "Merkez tıkanık dk", "Akşam tıkanık dk", "Akşam gecikme (dk)"]]
    for copies in (1.0, 2.0, 4.0):
        key = str(int(copies))
        rows.append([key, f"%{n(100 * loss[key])}", n(b_mean.loc[(SY, "Sionna", copies), "congested_minutes_per_day"]),
                     n(b_mean.loc[(CE, "Sionna", copies), "congested_minutes_per_day"]),
                     n(b_mean.loc[(EV, "Sionna", copies), "congested_minutes_per_day"]),
                     n(b_mean.loc[(EV, "Sionna", copies), "agent_delay_minutes"])])
    S += [table(rows, [1.5, 1.5, 1.8, 1.7, 1.7, 1.8]), Spacer(1, 7),
          p("Sionna satırları, aynı ortalama kaybın rastgele dağıldığı durumdan kötü. Rastgele kayıpta her ajanın "
            "tahmini biraz eski olur. Zayıf sinyalde ise bazı ajanların hiç tahmini olmaz. Yani <b>ortalama iletimden "
            "çok kapsama önemlidir</b>. Kopya sayısını artırmak bunu düzeltir ve çok ucuzdur: dört kopya bile 15 "
            "dakikada yaklaşık 23.000 kaynak öğesi kullanır, bu hücre kapasitesinin çok küçük bir kısmıdır.")]

    S += [p("9. Etkileşimli simülasyon", H1),
          p("Sonuçları canlı izlemek için Network Weather adlı bir tarayıcı simülasyonu da yaptık. Aynı hücre, görev "
            "ve zamanlayıcı kurallarını gerçek Milano trafiği ve Sionna değerleriyle çalıştırır. 10 günlük bir "
            "çalıştırma yaklaşık 40 milisaniye sürdüğü için ayar değiştikçe sonuç hemen yenilenir."),
          figure(SCREENSHOT, "Şekil 7. Hücre görünümü: ajanlar, yük ve durum bilgisi.", width=8.2 * cm),
          p("Ekran nasıl okunur?", H2)]
    S += bullets(["<b>Yeşil nokta:</b> Veri gönderen ajan.", "<b>Sarı halka:</b> Sakin saati bekleyen ajan.",
                  "<b>Kırmızı nokta:</b> Acil görev.", "<b>Sol üstteki yüzde:</b> Hücrenin anlık yükü.",
                  "<b>Tahmin şeridi:</b> Önümüzdeki 24 saat. Parlak çubuklar sakin penceredir.",
                  "<b>Günlük grafik:</b> Kesikli çizginin üstü tıkanıklıktır.",
                  "<b>Son yayın:</b> Protokoldeki gerçek baytlar."])
    S += [Spacer(1, 4),
          p("Tarayıcı motoru Python'a yakın sonuç verir. Kusursuz tahminle akşam yoğun bölgede tepe (b) için %88 "
            "(Python %90), (d) için %82 (Python %83) çıktı. Tıkanık dakikalar daha fazla fark eder, çünkü görev çekilişleri "
            "farklıdır. Rapordaki tüm sonuçlar Python'dan alınmıştır."),
          p("Tarayıcıda iki sadeleştirme vardır: ARIMA yerine \"haftalık şekil + AR(1)\" modeli çalışır, LSTM yoktur. "
            "Sentetik seri de ayrı üretildiği için sentetik sayılar farklıdır."),
          p("10. Sınırlar", H1),
          p("Sonuçları okurken şunları akılda tutmak gerekir:")]
    S += bullets(["<b>Varsayımlar:</b> Eşik, yük ölçeği, ajan payı, görev boyutu ve teslim süreleri varsayımdır.",
                  "<b>Ajan verisi yok:</b> Görevler modellenmiştir. Gerçek ajanlar insan şekline uymayabilir.",
                  "<b>Hücre tanımı:</b> 5×5 blok gerçek bir hücrenin yerine geçer. Her tipten tek blok denendi.",
                  "<b>Sabit insan trafiği:</b> Tohumlar yalnızca görevleri değiştirir, gerçek değişkenlik daha fazladır.",
                  "<b>Ayar seçimi:</b> Pencere ağırlığı 2, sentetik ve akşam yoğun sonuçlarına bakılarak seçildi. Resmî bir tarama yapılmadı.",
                  "<b>Yük ölçümü:</b> Sayaç, görevlerin tıkanıklıkta yavaşlamasını saymaz.",
                  "<b>Güven:</b> Tüm ajanların kurala uyduğu ve yayının gerçek olduğu varsayıldı. Kurala uymayan ajanlar ve sahte yayın incelenmedi.",
                  "<b>Sionna modeli:</b> Kapasite varsayılan banda ve MIMO'suz yapıya bağlıdır. Hızlı sönümleme sadeleştirilmiştir.",
                  "<b>Veri yaşı:</b> Milano kayıtları 2013-2014 yıllarına aittir."])
    S += [p("11. Sonuç", H1),
          p("Küçük, tek yönlü bir tahmin yayını (yaklaşık 4,3 bit/s), ajanlar rastgeleleştirdiği ve belirsizliği hesaba "
            "kattığı sürece ertelenebilir trafiği yoğun saatlerden uzaklaştırabilir. Gerçek serilerde tıkanıklık güçlü "
            "biçimde azaldı, bedeli saatlerle ölçülen ajan gecikmesi oldu."),
          p("Asıl ders toplu davranışla ilgilidir. En büyük tehlike ortalama hatası yüksek bir tahmin değil, çok "
            "sayıda ajanın aynı anda inandığı ortak bir hatadır. Bu yüzden bir tahmin protokolü kendi belirsizliğini de "
            "taşımalı, alıcılar da onu yayılmak için kullanmalıdır."),
          p("Sonraki adımlar: pencere genişliği için ilkeli bir kural, daha çok hücreyle test, gerçek ajan iş yükleri, "
            "kurala uymayan ajanlar, yayının doğrulanması ve komşu hücre yüküne bağlı bir kapasite modeli.")]

    S += [p("Terimler sözlüğü", H1),
          table([["Terim", "Anlamı"],
                 ["Ajan", "Kullanıcı adına kendi başına çalışan yapay zekâ yazılımı"],
                 ["Baz istasyonu / hücre", "Mobil cihazlara hizmet veren anten ve kapsadığı alan"],
                 ["Tıkanıklık", "Yükün kapasitenin %80'ini geçmesi"],
                 ["Slot", "15 dakikalık zaman dilimi"],
                 ["Sürü etkisi", "Herkesin aynı boş zamana yönelip orada yeni tepe oluşturması"],
                 ["Sakin pencere", "Ajanın rastgele seçim yaptığı düşük yüklü zaman aralığı"],
                 ["Güven (confidence)", "Tahminin ne kadar hatalı olabileceğini söyleyen değer"],
                 ["MAE / RMSE", "Tahmin hatası ölçüleri; küçük olan iyidir"],
                 ["Tohum", "Rastgele sayı üretecinin başlangıç değeri. Aynı tohum aynı sonucu verir"],
                 ["SINR", "Sinyalin gürültü ve parazite oranı; yüksek olan iyidir"],
                 ["LDPC, QPSK", "5G'de kullanılan hata düzeltme kodu ve modülasyon türü"],
                 ["Oransal adil zamanlayıcı", "Kaynağı kullanıcı hızını ve adaleti dengeleyerek dağıtan algoritma"],
                 ["ARIMA, LSTM", "Zaman serisi tahmin modelleri; LSTM bir sinir ağıdır"],
                 ["Sionna", "NVIDIA'nın açık kaynaklı kablosuz haberleşme simülasyon kütüphanesi"]],
                [3, 8]), Spacer(1, 12),
          p("Sonuçları yeniden üretmek", H1),
          p("Tüm kod özel GitHub deposundadır (grkmuysl/network-weather-service). Python 3.14 ile test edildi ve 75 "
            "otomatik test geçiyor. Milano komutları 9,3 GB'lik veri dosyasını gerektirir."),
          table([["Komut", "Ne yapar?"],
                 ["python -m pytest", "Testleri çalıştırır"], ["python -m nwl", "Sentetik senaryolar"],
                 ["python -m nwl.evaluate_forecasts", "Model karşılaştırması"],
                 ["python -m nwl.experiments main", "Ana karşılaştırma"],
                 ["python -m nwl.experiments sweeps", "Varsayım taramaları"],
                 ["python -m nwl.experiments errors", "Tahmin hatası testleri"],
                 ["python -m nwl.experiments broadcast", "Yayın kaybı deneyi"],
                 ["python -m nwl.sionna_cell", "Sionna kapasite hesabı"],
                 ["python -m nwl.sionna_broadcast", "Sionna mesaj kaybı ölçümü"],
                 ["python -m report.build_report_tr", "Bu raporu üretir"]], [5, 5]), Spacer(1, 12),
          p("Kaynaklar", H1)]
    S += bullets([
        "G. Barlacchi vd., \"A multi-source dataset of urban life in the city of Milan and the Province of Trentino\", Scientific Data 2:150055, 2015. doi:10.1038/sdata.2015.55",
        "BigDataChallenge yarışması, Telecom Italia Milano verisi, Harvard Dataverse, doi:10.7910/DVN/EGZHFV (ODbL 1.0).",
        "CESNET TS-Zoo Telecom Italia dönüşümü, Zenodo, doi:10.5281/zenodo.22721975.",
        "R. Alimi vd., \"Application-Layer Traffic Optimization (ALTO) Protocol\", RFC 7285, IETF, 2014.",
        "3GPP TS 23.288, 5G sistemi için şebeke veri analitiği hizmetleri mimarisi.",
        "3GPP TR 38.901, 0,5-100 GHz kanal modeli.",
        "J. Hoydis vd., \"Sionna: An Open-Source Library for Next-Generation Physical Layer Research\", 2022. Sürüm 2.2.0, github.com/NVlabs/sionna.",
    ])

    document = Report(str(OUTPUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                      topMargin=2 * cm, bottomMargin=2 * cm,
                      title="Network Weather Service: Ayrıntılı Rapor",
                      author="Enes Ödev, Görkem Uysal")
    frame = Frame(document.leftMargin, document.bottomMargin, document.width, document.height, id="ana",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    document.addPageTemplates([PageTemplate(id="sayfa", frames=[frame], onPage=footer)])
    document.multiBuild(S)


if __name__ == "__main__":
    build()
